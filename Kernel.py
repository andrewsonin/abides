import os
import queue
from abc import ABC, abstractmethod
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional, Any, Type

import numpy as np
import pandas as pd

from agent.Agent import Agent
from message.Message import MessageType
from util.util import log_print

__all__ = (
    "Kernel",
)


class Oracle(ABC):  # TODO
    pass


class AgentLatencyModelBase(ABC):
    __slots__ = ()

    @abstractmethod
    def get_latency_and_noise(self, sender_id: int, recipient_id: int) -> Tuple[int, int]:
        pass


class DefaultAgentLatencyModel(AgentLatencyModelBase):
    __slots__ = ("default_latency",)

    def __init__(self, default_latency: int):
        self.default_latency = default_latency

    def get_latency_and_noise(self, sender_id: int, recipient_id: int) -> Tuple[int, int]:
        return self.default_latency, 0


class AgentLatencyModel(AgentLatencyModelBase):
    __slots__ = ("latency_matrix", "noise_probs", "random_state")

    def __init__(self,
                 latency_matrix: List[List[int]],
                 noise_probs: List[float],
                 random_state: np.random.RandomState):
        """
        :param latency_matrix:  Defines the communication delay between every pair of agents.
                                The first dimension refers to the sender ID, the second — to the recipient one
        :param noise_probs:     list with list index = ns extra delay, value = probability of this delay.
        :param random_state:    np.random.RandomState used for to sample noise
        """
        noise_probs = np.asarray(noise_probs)
        if not (noise_probs >= 0).all() or not np.isclose(noise_probs.sum(), 1):  # type: ignore
            raise ValueError("Parameter 'noise_probs' should define the array of probabilities that add up to 1")
        self.latency_matrix = latency_matrix
        # There is a noise model for latency, intended to be a one-sided
        # distribution with the peak at zero.  By default there is no noise
        # (100% chance to add zero ns extra delay).
        self.noise_probs = noise_probs
        self.random_state = random_state

    def get_latency_and_noise(self, sender_id: int, recipient_id: int) -> Tuple[int, int]:
        latency = self.latency_matrix[sender_id][recipient_id]
        noise = self.random_state.choice(len(self.noise_probs), p=self.noise_probs)
        return latency, noise


class CustomState(dict):
    __slots__ = ()


class AgentID(int):
    __slots__ = ()
    pass


class Kernel:
    __slots__ = (
        "name",
        "random_state",
        "messages",
        "current_time",
        "kernel_wallclock_start",
        "mean_result_by_agent_type",
        "agent_count_by_type",
        "summary_log",
        "agents",
        "custom_state",
        "startTime",
        "stopTime",
        "seed",
        "skip_log",
        "oracle",
        "log_dir",
        "agent_current_times",
        "agent_computation_delays",
        "agent_latency_model",
        "current_agent_additional_delay"
    )

    def __init__(self,
                 kernel_name: str,
                 random_state: np.random.RandomState,
                 *,
                 start_time: datetime,
                 stop_time: datetime,
                 agents: Optional[List[Agent]] = None,
                 default_computation_delay: int = 1,
                 agent_latency_model: Optional[AgentLatencyModelBase] = None,
                 default_latency: int = 1,
                 skip_log: bool = False,
                 oracle: Optional[Oracle] = None,
                 log_dir: Optional[Union[str, PathLike, Path]] = None) -> None:

        # kernel_name is for human readers only.
        self.name = kernel_name
        if not isinstance(random_state, np.random.RandomState):
            raise ValueError(
                "A valid, seeded np.random.RandomState object is required for the Kernel",
                kernel_name
            )

        self.random_state = random_state

        # A single message queue to keep everything organized by increasing
        # delivery timestamp.
        self.messages: queue.PriorityQueue[Tuple[datetime, Tuple[int, MessageType, Any]]] = queue.PriorityQueue()

        # currentTime is None until after kernelStarting() event completes
        # for all agents.  This is a pd.Timestamp that includes the date.
        self.current_time: Optional[pd.Timestamp] = None

        # Timestamp at which the Kernel was created.  Primarily used to
        # create a unique log directory for this run.  Also used to
        # print some elapsed time and messages per second statistics.
        self.kernel_wallclock_start = pd.Timestamp('now')

        # TODO: This is financial, and so probably should not be here...
        self.mean_result_by_agent_type: Dict[str, float] = {}
        self.agent_count_by_type: Dict[str, int] = {}

        # The Kernel maintains a summary log to which agents can write
        # information that should be centralized for very fast access
        # by separate statistical summary programs.  Detailed event
        # logging should go only to the agent's individual log.  This
        # is for things like "final position value" and such.
        self.summary_log: List[Dict[str, Any]] = []

        # agents must be a list of agents for the simulation,
        #        based on class agent.Agent
        if agents is not None:
            self.agents: List[Agent] = agents
        else:
            self.agents = []

        # The kernel start and stop time (first and last timestamp in
        # the simulation, separate from anything like exchange open/close).
        self.startTime = start_time
        self.stopTime = stop_time

        # The kernel maintains a current time for each agent to allow
        # simulation of per-agent computation delays.  The agent's time
        # is pushed forward (see below) each time it awakens, and it
        # cannot receive new messages/wakeups until the global time
        # reaches the agent's time.  (i.e. it cannot act again while
        # it is still "in the future")

        # This also nicely enforces agents being unable to act before
        # the simulation startTime.
        self.agent_current_times = [self.startTime] * len(self.agents)

        # agentComputationDelays is in nanoseconds, starts with a default
        # value from config, and can be changed by any agent at any time
        # (for itself only).  It represents the time penalty applied to
        # an agent each time it is awakened  (wakeup or recvMsg).  The
        # penalty applies _after_ the agent acts, before it may act again.
        # TODO: this might someday change to pd.Timedelta objects.
        self.agent_computation_delays = [default_computation_delay] * len(self.agents)

        if agent_latency_model is None:
            self.agent_latency_model: AgentLatencyModelBase = DefaultAgentLatencyModel(default_latency)
        elif isinstance(agent_latency_model, AgentLatencyModelBase):
            if isinstance(agent_latency_model, AgentLatencyModel):
                num_agents = len(self.agents)
                if any(len(row) != num_agents for row in agent_latency_model.latency_matrix):
                    raise ValueError(
                        "Attribute 'latency_matrix' of 'agent_latency_model' "
                        "should be a square matrix of size equals to the number of agents"
                    )
            self.agent_latency_model = agent_latency_model
        else:
            raise TypeError("Parameter 'agent_latency_model' should be an instance of 'AgentLatencyModelBase' or None")

        # The kernel maintains an accumulating additional delay parameter
        # for the current agent.  This is applied to each message sent
        # and upon return from wakeup/receiveMessage, in addition to the
        # agent's standard computation delay.  However, it never carries
        # over to future wakeup/receiveMessage calls.  It is useful for
        # staggering of sent messages.
        self.current_agent_additional_delay = 0

        # If a log directory was not specified, use the initial wallclock.
        self.log_dir: Union[str, PathLike, Path] = log_dir or str(int(self.kernel_wallclock_start.timestamp()))

        self.custom_state = CustomState()

        # Should the Kernel skip writing agent logs?
        self.skip_log = skip_log

        # The data oracle for the simulation, if needed.
        self.oracle = oracle

        log_print("Kernel initialized: {}", self.name)

    # This is called to actually start the simulation, once all agent
    # configuration is done.
    def runner(self, num_simulations: int = 1) -> CustomState:

        agents = self.agents
        # Simulation custom state in a freeform dictionary.  Allows config files
        # that drive multiple simulations, or require the ability to generate
        # special logs after simulation, to obtain needed output without special
        # case code in the Kernel.  Per-agent state should be handled using the
        # provided updateAgentState() method.
        self.custom_state.clear()

        self.current_agent_additional_delay = 0

        log_print("Kernel started: {}", self.name)
        log_print("Simulation started!")

        # Note that num_simulations has not yet been really used or tested
        # for anything.  Instead we have been running multiple simulations
        # with coarse parallelization from a shell script.
        for sim in range(num_simulations):
            log_print("Starting sim {}", sim)

            # Event notification for kernel init (agents should not try to
            # communicate with other agents, as order is unknown).  Agents
            # should initialize any internal resources that may be needed
            # to communicate with other agents during agent.kernelStarting().
            # Kernel passes self-reference for agents to retain, so they can
            # communicate with the kernel in the future (as it does not have
            # an agentID).
            log_print("\n--- Agent.kernelInitializing() ---")
            for agent in agents:
                agent.kernelInitializing(self)

            # Event notification for kernel start (agents may set up
            # communications or references to other agents, as all agents
            # are guaranteed to exist now).  Agents should obtain references
            # to other agents they require for proper operation (exchanges,
            # brokers, subscription services...).  Note that we generally
            # don't (and shouldn't) permit agents to get direct references
            # to other agents (like the exchange) as they could then bypass
            # the Kernel, and therefore simulation "physics" to send messages
            # directly and instantly or to perform disallowed direct inspection
            # of the other agent's state.  Agents should instead obtain the
            # agent ID of other agents, and communicate with them only via
            # the Kernel.  Direct references to utility objects that are not
            # agents are acceptable (e.g. oracles).
            log_print("\n--- Agent.kernelStarting() ---")
            for agent in agents:
                agent.kernelStarting(self.startTime)

            # Set the kernel to its startTime.
            self.current_time = self.startTime
            log_print("\n--- Kernel Clock started ---")
            log_print(
                "Kernel.currentTime is now {}",
                self.current_time
            )

            # Start processing the Event Queue.
            log_print("\n--- Kernel Event Queue begins ---")
            log_print(
                "Kernel will start processing messages.  Queue length: {}",
                len(self.messages.queue)
            )

            # Track starting wallclock time and total message count for stats at the end.
            eventQueueWallClockStart = pd.Timestamp('now')
            ttl_messages = 0

            # Process messages until there aren't any (at which point there never can
            # be again, because agents only "wake" in response to messages), or until
            # the kernel stop time is reached.
            while not self.messages.empty() and self.current_time and self.current_time <= self.stopTime:
                # Get the next message in timestamp order (delivery time) and extract it.
                self.current_time, event = self.messages.get()
                msg_recipient_id, msg_type, msg = event

                # Periodically print the simulation time and total messages, even if muted.
                if ttl_messages % 100000 == 0:
                    print(
                        f"\n--- Simulation time: {self.fmtTime(self.current_time)}, "
                        f"messages processed: {ttl_messages}, "
                        f"wallclock elapsed: {pd.Timestamp('now') - eventQueueWallClockStart} ---\n"
                    )

                log_print("\n--- Kernel Event Queue pop ---")
                log_print(
                    "Kernel handling {} message for agent {} at time {}",
                    msg_type,
                    msg_recipient_id,
                    self.fmtTime(self.current_time)
                )

                ttl_messages += 1

                # In between messages, always reset the currentAgentAdditionalDelay.
                self.current_agent_additional_delay = 0

                # Dispatch message to agent.
                if msg_type is MessageType.WAKEUP:

                    # Who requested this wakeup call?
                    agent_id = msg_recipient_id

                    # Test to see if the agent is already in the future.  If so,
                    # delay the wakeup until the agent can act again.
                    if self.agent_current_times[agent_id] > self.current_time:
                        # Push the wakeup call back into the PQ with a new time.
                        self.messages.put(
                            (self.agent_current_times[agent_id], (msg_recipient_id, msg_type, msg))
                        )
                        log_print(
                            "Agent in future: wakeup requeued for {}",
                            self.fmtTime(self.agent_current_times[agent_id])
                        )
                        continue

                    # Set agent's current time to global current time for start
                    # of processing.
                    self.agent_current_times[agent_id] = self.current_time

                    # Wake the agent.
                    agents[agent_id].wakeup(self.current_time)

                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agent_current_times[agent_id] += pd.Timedelta(
                        self.agent_computation_delays[agent_id] + self.current_agent_additional_delay
                    )

                    log_print(
                        "After wakeup return, agent {} delayed from {} to {}",
                        agent_id,
                        self.fmtTime(self.current_time),
                        self.fmtTime(self.agent_current_times[agent_id])
                    )

                elif msg_type is MessageType.MESSAGE:

                    # Who is receiving this message?
                    agent_id = msg_recipient_id

                    # Test to see if the agent is already in the future.  If so,
                    # delay the message until the agent can act again.
                    if self.agent_current_times[agent_id] > self.current_time:
                        # Push the message back into the PQ with a new time.
                        self.messages.put(
                            (self.agent_current_times[agent_id], (msg_recipient_id, msg_type, msg))
                        )
                        log_print(
                            "Agent in future: message requeued for {}",
                            self.fmtTime(self.agent_current_times[agent_id])
                        )
                        continue

                    # Set agent's current time to global current time for start
                    # of processing.
                    self.agent_current_times[agent_id] = self.current_time

                    # Deliver the message.
                    agents[agent_id].receiveMessage(self.current_time, msg)

                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agent_current_times[agent_id] += pd.Timedelta(
                        self.agent_computation_delays[agent_id] + self.current_agent_additional_delay
                    )

                    log_print(
                        "After receiveMessage return, agent {} delayed from {} to {}",
                        agent_id,
                        self.fmtTime(self.current_time),
                        self.fmtTime(self.agent_current_times[agent_id])
                    )

                else:
                    raise ValueError(
                        "Unknown message type found in queue",
                        "currentTime:",
                        self.current_time,
                        "messageType:",
                        msg_type
                    )

            if self.messages.empty():
                log_print("\n--- Kernel Event Queue empty ---")

            if self.current_time and (self.current_time > self.stopTime):
                log_print("\n--- Kernel Stop Time surpassed ---")

            # Record wall clock stop time and elapsed time for stats at the end.
            eventQueueWallClockStop = pd.Timestamp('now')
            eventQueueWallClockElapsed = eventQueueWallClockStop - eventQueueWallClockStart

            # Event notification for kernel end (agents may communicate with
            # other agents, as all agents are still guaranteed to exist).
            # Agents should not destroy resources they may need to respond
            # to final communications from other agents.
            log_print("\n--- Agent.kernelStopping() ---")
            for agent in agents:
                agent.kernelStopping()

            # Event notification for kernel termination (agents should not
            # attempt communication with other agents, as order of termination
            # is unknown).  Agents should clean up all used resources as the
            # simulation program may not actually terminate if num_simulations > 1.
            log_print("\n--- Agent.kernelTerminating() ---")
            for agent in agents:
                agent.kernelTerminating()

            print(
                f"Event Queue elapsed: {eventQueueWallClockElapsed}, "
                f"messages: {ttl_messages}, "
                f"messages per second: {ttl_messages / (eventQueueWallClockElapsed / (np.timedelta64(1, 's'))):0.1f}"
            )
            log_print("Ending sim {}", sim)

        # The Kernel adds a handful of custom state results for all simulations,
        # which configurations may use, print, log, or discard.
        self.custom_state['kernel_event_queue_elapsed_wallclock'] = eventQueueWallClockElapsed
        self.custom_state['kernel_slowest_agent_finish_time'] = max(self.agent_current_times)

        # Agents will request the Kernel to serialize their agent logs, usually
        # during kernelTerminating, but the Kernel must write out the summary
        # log itself.
        self.writeSummaryLog()

        # This should perhaps be elsewhere, as it is explicitly financial, but it
        # is convenient to have a quick summary of the results for now.
        print("Mean ending value by agent type:")
        for a in self.mean_result_by_agent_type:
            value = self.mean_result_by_agent_type[a]
            count = self.agent_count_by_type[a]
            print(f"{a}: {round(value / count)}")

        print("Simulation ending!")

        return self.custom_state

    def sendMessage(self,
                    sender: Optional[int] = None,
                    recipient: Optional[int] = None,
                    msg: Optional[str] = None,
                    delay: int = 0):
        # Called by an agent to send a message to another agent.  The kernel
        # supplies its own currentTime (i.e. "now") to prevent possible
        # abuse by agents.  The kernel will handle computational delay penalties
        # and/or network latency.  The message must derive from the message.Message class.
        # The optional delay parameter represents an agent's request for ADDITIONAL
        # delay (beyond the Kernel's mandatory computation + latency delays) to represent
        # parallel pipeline processing delays (that should delay the transmission of messages
        # but do not make the agent "busy" and unable to respond to new messages).

        if sender is None:
            raise ValueError(
                "sendMessage() called without valid sender ID",
                "sender:",
                sender,
                "recipient:",
                recipient,
                "msg:",
                msg
            )

        if recipient is None:
            raise ValueError(
                "sendMessage() called without valid recipient ID",
                "sender:",
                sender,
                "recipient:",
                recipient,
                "msg:",
                msg
            )

        if msg is None:
            raise ValueError(
                "sendMessage() called with message == None",
                "sender:",
                sender,
                "recipient:",
                recipient,
                "msg:",
                msg
            )

        # Apply the agent's current computation delay to effectively "send" the message
        # at the END of the agent's current computation period when it is done "thinking".
        # NOTE: sending multiple messages on a single wake will transmit all at the same
        # time, at the end of computation.  To avoid this, use Agent.delay() to accumulate
        # a temporary delay (current cycle only) that will also stagger messages.

        # The optional pipeline delay parameter DOES push the send time forward, since it
        # represents "thinking" time before the message would be sent.  We don't use this
        # for much yet, but it could be important later.

        # This means message delay (before latency) is the agent's standard computation delay
        # PLUS any accumulated delay for this wake cycle PLUS any one-time requested delay
        # for this specific message only.
        sentTime = self.current_time + pd.Timedelta(
            self.agent_computation_delays[sender] + self.current_agent_additional_delay + delay
        )

        # Apply communication delay per the agentLatencyModel, if defined, or the
        # agentLatency matrix [sender][recipient] otherwise.

        latency, noise = self.agent_latency_model.get_latency_and_noise(sender, recipient)
        deliverAt = sentTime + pd.Timedelta(latency + noise)
        log_print(
            "Kernel applied latency {}, noise {}, accumulated delay {}, "
            "one-time delay {} on sendMessage from: {} to {}, scheduled for {}",
            latency,
            noise,
            self.current_agent_additional_delay,
            delay,
            self.agents[sender].name,
            self.agents[recipient].name,
            self.fmtTime(deliverAt)
        )

        # Finally drop the message in the queue with priority == delivery time.
        self.messages.put(
            (deliverAt, (recipient, MessageType.MESSAGE, msg))
        )

        log_print(
            "Sent time: {}, current time {}, computation delay {}",
            sentTime,
            self.current_time,
            self.agent_computation_delays[sender]
        )
        log_print("Message queued: {}", msg)

    def setWakeup(self, sender: Optional[int] = None, requested_time: Optional[datetime] = None):
        # Called by an agent to receive a "wakeup call" from the kernel
        # at some requested future time.  Defaults to the next possible
        # timestamp.  Wakeup time cannot be the current time or a past time.
        # Sender is required and should be the ID of the agent making the call.
        # The agent is responsible for maintaining any required state; the
        # kernel will not supply any parameters to the wakeup() call.

        if requested_time is None:
            requested_time = self.current_time + pd.TimeDelta(1)

        if sender is None:
            raise ValueError(
                "setWakeup() called without valid sender ID",
                "sender:",
                sender,
                "requestedTime:",
                requested_time
            )

        if self.current_time and requested_time < self.current_time:
            raise ValueError(
                "setWakeup() called with requested time not in future",
                "currentTime:",
                self.current_time,
                "requestedTime:",
                requested_time
            )

        log_print(
            "Kernel adding wakeup for agent {} at time {}",
            sender,
            self.fmtTime(requested_time)
        )

        self.messages.put(
            (requested_time, (sender, MessageType.WAKEUP, None))
        )

    def getAgentComputeDelay(self, sender: int):
        # Allows an agent to query its current computation delay.
        return self.agent_computation_delays[sender]

    def setAgentComputeDelay(self, sender: int, requested_delay: Optional[int] = None):
        # Called by an agent to update its computation delay.  This does
        # not initiate a global delay, nor an immediate delay for the
        # agent.  Rather it sets the new default delay for the calling
        # agent.  The delay will be applied upon every return from wakeup
        # or recvMsg.  Note that this delay IS applied to any messages
        # sent by the agent during the current wake cycle (simulating the
        # messages popping out at the end of its "thinking" time).

        # Also note that we DO permit a computation delay of zero, but this should
        # really only be used for special or massively parallel agents.

        # requestedDelay should be in whole nanoseconds.
        if not isinstance(requested_delay, int):
            raise ValueError(
                "Requested computation delay must be whole nanoseconds.",
                "requestedDelay:",
                requested_delay
            )

        # requestedDelay must be non-negative.
        if requested_delay < 0:
            raise ValueError(
                "Requested computation delay must be non-negative nanoseconds.",
                "requestedDelay:",
                requested_delay
            )

        self.agent_computation_delays[sender] = requested_delay

    def delayAgent(self, sender: Optional[int] = None, additional_delay: Optional[int] = None):
        # Called by an agent to accumulate temporary delay for the current wake cycle.
        # This will apply the total delay (at time of sendMessage) to each message,
        # and will modify the agent's next available time slot.  These happen on top
        # of the agent's compute delay BUT DO NOT ALTER IT.  (i.e. effects are transient)
        # Mostly useful for staggering outbound messages.

        # additionalDelay should be in whole nanoseconds.
        if not isinstance(additional_delay, int):
            raise ValueError(
                "Additional delay must be whole nanoseconds.",
                "additionalDelay:",
                additional_delay
            )

        # additionalDelay must be non-negative.
        if additional_delay < 0:
            raise ValueError(
                "Additional delay must be non-negative nanoseconds.",
                "additionalDelay:",
                additional_delay
            )

        self.current_agent_additional_delay += additional_delay

    def findAgentByType(self, agent_type: Type[Agent]) -> Optional[int]:
        # Called to request an arbitrary agent ID that matches the class or base class
        # passed as "type".  For example, any ExchangeAgent, or any NasdaqExchangeAgent.
        # This method is rather expensive, so the results should be cached by the caller!

        for agent in self.agents:
            if isinstance(agent, agent_type):
                return agent.id
        return None

    def writeLog(self, sender: int, log_df: pd.DataFrame, filename: Optional[Union[str, PathLike, Path]] = None):
        # Called by any agent, usually at the very end of the simulation just before
        # kernel shutdown, to write to disk any log dataframe it has been accumulating
        # during simulation.  The format can be decided by the agent, although changes
        # will require a special tool to read and parse the logs.  The Kernel places
        # the log in a unique directory per run, with one filename per agent, also
        # decided by the Kernel using agent type, id, etc.

        # If there are too many agents, placing all these files in a directory might
        # be unfortunate.  Also if there are too many agents, or if the logs are too
        # large, memory could become an issue.  In this case, we might have to take
        # a speed hit to write logs incrementally.

        # If filename is not None, it will be used as the filename.  Otherwise,
        # the Kernel will construct a filename based on the name of the Agent
        # requesting log archival.

        if self.skip_log:
            return

        path = os.path.join(".", "log", self.log_dir)
        file = f"{filename or self.agents[sender].name.replace(' ', '')}.bz2"

        os.makedirs(path, exist_ok=True)
        log_df.to_pickle(os.path.join(path, file), compression='bz2')

    def appendSummaryLog(self, sender, event_type, event):
        # We don't even include a timestamp, because this log is for one-time-only
        # summary reporting, like starting cash, or ending cash.
        self.summary_log.append(
            {
                'AgentID': sender,
                'AgentStrategy': self.agents[sender].type,
                'EventType': event_type,
                'Event': event
            }
        )

    def writeSummaryLog(self):
        path = os.path.join(".", "log", self.log_dir)
        file = "summary_log.bz2"

        os.makedirs(path, exist_ok=True)
        dfLog = pd.DataFrame(self.summary_log)
        dfLog.to_pickle(os.path.join(path, file), compression='bz2')

    def updateAgentState(self, agent_id: int, state: Any) -> None:
        """ Called by an agent that wishes to replace its custom state in the dictionary
            the Kernel will return at the end of simulation.  Shared state must be set directly,
            and agents should coordinate that non-destructively.

            Note that it is never necessary to use this kernel state dictionary for an agent
            to remember information about itself, only to report it back to the config file.
        """
        if 'agent_state' not in self.custom_state:
            self.custom_state['agent_state'] = {}
        self.custom_state['agent_state'][agent_id] = state

    @staticmethod
    def fmtTime(simulationTime):
        # The Kernel class knows how to pretty-print time.  It is assumed simulationTime
        # is in nanoseconds since midnight.  Note this is a static method which can be
        # called either on the class or an instance.

        # Try just returning the pd.Timestamp now.
        return simulationTime

        # ns = simulationTime
        # hr = int(ns / (1000000000 * 60 * 60))
        # ns -= (hr * 1000000000 * 60 * 60)
        # m = int(ns / (1000000000 * 60))
        # ns -= (m * 1000000000 * 60)
        # s = int(ns / 1000000000)
        # ns = int(ns - (s * 1000000000))
        #
        # return "{:02d}:{:02d}:{:02d}.{:09d}".format(hr, m, s, ns)
