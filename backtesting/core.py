from copy import deepcopy
from os import PathLike, makedirs
from os.path import join as joinpath
from pathlib import Path
from typing import Tuple, List, Dict, Sequence, Union, Optional, Any, Generic, Type

import numpy as np
import pandas as pd

from backtesting.latency.base import AgentLatencyModelBase
from backtesting.latency.types import DefaultAgentLatencyModel, AgentLatencyModel
from backtesting.message.base import MessageAbstractBase, WakeUp, Message
from backtesting.typing import Event, FileName
from backtesting.typing.core import KernelSummaryLogEntry, AgentEventLogEntry, KernelCustomState
from backtesting.typing.vars import OracleType
from backtesting.utils.constants import one_ns_timedelta, one_s_timedelta
from backtesting.utils.structures import PriorityQueue
from backtesting.utils.util import log_print, is_integer

__all__ = (
    "Kernel",
    "Agent"
)


class Kernel(Generic[OracleType]):
    __slots__ = (
        "name",
        "random_state",
        "start_time",
        "stop_time",
        "agents",
        "message_queue",
        "current_time",
        "kernel_wallclock_start",
        "mean_result_by_agent_type",
        "agent_count_by_type",
        "summary_log",
        "custom_state",
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
                 *,
                 name: str,
                 random_state: np.random.RandomState,
                 start_time: pd.Timestamp,
                 stop_time: pd.Timestamp,
                 agents: Sequence['Agent'],
                 default_computation_delay: int = 1,
                 agent_latency_model: Union[int, AgentLatencyModelBase] = 1,
                 skip_log: bool = False,
                 oracle: OracleType = None,
                 log_dir: Optional[FileName] = None) -> None:
        """

        Args:
            name:                       Kernel name (is for human readers only)
            random_state:               Random state used by Kernel
            start_time:                 Simulation start time
            stop_time:                  Simulation stop time
            agents:                     Sequence of agents involved in simulation
            default_computation_delay:  Default computational delay
            agent_latency_model:        Single number (in nanoseconds)
                                        [OR]
                                        AgentLatencyModelBase instance for modelling agent-to-agent delays
                                        caused by network topology characteristics and random noise
            skip_log:                   Whether to skip logging
            oracle:                     Instance of Oracle
            log_dir:                    Log directory
        """

        # kernel_name is for human readers only.
        if not isinstance(name, str):
            raise TypeError("Parameter name should be of type str")
        self.name = name

        if not isinstance(random_state, np.random.RandomState):
            raise ValueError(
                "A valid, seeded np.random.RandomState object is required for the Kernel",
                name
            )
        self.random_state = random_state

        # A single message queue to keep everything organized by increasing
        # delivery timestamp.
        self.message_queue: PriorityQueue[Tuple[pd.Timestamp, Tuple[int, MessageAbstractBase]]] = PriorityQueue()

        # Timestamp at which the Kernel was created. Primarily used to
        # create a unique log directory for this run. Also used to
        # print some elapsed time and messages per second statistics.
        self.kernel_wallclock_start = pd.Timestamp('now')

        # TODO: This is financial, and so probably should not be here...
        self.mean_result_by_agent_type: Dict[str, float] = {}
        self.agent_count_by_type: Dict[str, int] = {}

        # The Kernel maintains a summary log to which agents can write
        # information that should be centralized for very fast access
        # by separate statistical summary programs. Detailed event
        # logging should go only to the agent's individual log. This
        # is for things like "final position value" and such.
        self.summary_log: List[KernelSummaryLogEntry] = []

        # agents must be a list of agents for the simulation,
        #        based on class agent.Agent
        if not all(isinstance(x, Agent) for x in agents):
            raise TypeError("'agents' must be a sequence containing only instances of the class Agent")
        self.agents = agents
        num_agents = len(agents)

        # The kernel start and stop time (first and last timestamp in
        # the simulation, separate from anything like exchange open/close).
        if not isinstance(start_time, pd.Timestamp) or not isinstance(stop_time, pd.Timestamp):
            raise TypeError("'start_time' and 'stop_time' must be of type pd.Timestamp")
        if stop_time < start_time:
            raise ValueError("'stop_time' should be larger than 'start_time'")
        self.current_time = self.start_time = start_time
        self.stop_time = stop_time

        # The kernel maintains a current time for each agent to allow
        # simulation of per-agent computation delays. The agent's time
        # is pushed forward (see below) each time it awakens, and it
        # cannot receive new messages/wakeups until the global time
        # reaches the agent's time. (i.e. it cannot act again while
        # it is still "in the future")

        # This also nicely enforces agents being unable to act before
        # the simulation startTime.
        self.agent_current_times = [self.start_time] * num_agents

        # agentComputationDelays is in nanoseconds, starts with a default
        # value from config, and can be changed by any agent at any time
        # (for itself only). It represents the time penalty applied to
        # an agent each time it is awakened  (wakeup or recvMsg). The
        # penalty applies _after_ the agent acts, before it may act again.
        # TODO: this might someday change to pd.Timedelta objects.
        if not isinstance(default_computation_delay, int):
            raise TypeError("'default_computation_delay' must be of type int")
        self.agent_computation_delays = [default_computation_delay] * num_agents

        if isinstance(agent_latency_model, int):
            self.agent_latency_model: AgentLatencyModelBase = DefaultAgentLatencyModel(agent_latency_model)
        elif isinstance(agent_latency_model, AgentLatencyModelBase):
            if (
                    isinstance(agent_latency_model, AgentLatencyModel)
                    and
                    any(
                        len(row) != num_agents or not all(map(is_integer, row))
                        for row in agent_latency_model.latency_matrix
                    )
            ):
                raise ValueError(
                    "Attribute 'latency_matrix' of 'agent_latency_model' "
                    "should be a square matrix of integers of size equals to the number of agents"
                )
            self.agent_latency_model = agent_latency_model
        else:
            raise TypeError("Parameter 'agent_latency_model' should be an instance of 'AgentLatencyModelBase' or None")

        # The kernel maintains an accumulating additional delay parameter
        # for the current agent. This is applied to each message sent
        # and upon return from wakeup/receiveMessage, in addition to the
        # agent's standard computation delay. However, it never carries
        # over to future wakeup/receiveMessage calls. It is useful for
        # staggering of sent messages.
        self.current_agent_additional_delay = 0

        # If a log directory was not specified, use the initial wallclock.
        self.log_dir: Union[str, PathLike, Path] = log_dir or str(int(self.kernel_wallclock_start.timestamp()))

        # Simulation custom state in a freeform dictionary. Allows config files
        # that drive multiple simulations, or require the ability to generate
        # special logs after simulation, to obtain needed output without special
        # case code in the Kernel. Per-agent state should be handled using the
        # provided updateAgentState() method.
        self.custom_state: KernelCustomState = {}

        # Should the Kernel skip writing agent logs?
        if not isinstance(skip_log, bool):
            raise TypeError("'skip_log' must be of type bool")
        self.skip_log = skip_log

        # The data oracle for the simulation, if needed.
        self.oracle = oracle

        log_print(f"Kernel initialized: {self.name}")

    # This is called to actually start the simulation, once all agent
    # configuration is done.
    def runner(self, num_simulations: int = 1) -> KernelCustomState:

        agents = self.agents
        message_queue = self.message_queue
        custom_state = self.custom_state
        agent_current_times = self.agent_current_times
        agent_computation_delays = self.agent_computation_delays
        start_time = self.start_time
        stop_time = self.stop_time

        custom_state.clear()

        self.current_agent_additional_delay = 0

        log_print(f"Kernel started: {self.name}\nSimulation started!")

        # Note that num_simulations has not yet been really used or tested
        # for anything. Instead we have been running multiple simulations
        # with coarse parallelization from a shell script.
        for sim in range(num_simulations):
            log_print(f"Starting sim {sim}")

            # Event notification for kernel init (agents should not try to
            # communicate with other agents, as order is unknown). Agents
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
            # are guaranteed to exist now). Agents should obtain references
            # to other agents they require for proper operation (exchanges,
            # brokers, subscription services...). Note that we generally
            # don't (and shouldn't) permit agents to get direct references
            # to other agents (like the exchange) as they could then bypass
            # the Kernel, and therefore simulation "physics" to send messages
            # directly and instantly or to perform disallowed direct inspection
            # of the other agent's state. Agents should instead obtain the
            # agent ID of other agents, and communicate with them only via
            # the Kernel. Direct references to utility objects that are not
            # agents are acceptable (e.g. oracles).
            log_print("\n--- Agent.kernelStarting() ---")
            for agent in agents:
                agent.kernelStarting(start_time)

            # Set the kernel to its startTime.
            self.current_time = start_time
            log_print(
                "\n--- Kernel Clock started ---\n"
                f"Kernel.currentTime is now {start_time}\n"
                "\n--- Kernel Event Queue begins ---\n"
                f"Kernel will start processing messages. Queue length: {len(message_queue)}"
            )

            # Track starting wallclock time and total message count for stats at the end.
            eventQueueWallClockStart = pd.Timestamp('now')
            total_messages = 0

            # Process messages until there aren't any (at which point there never can
            # be again, because agents only "wake" in response to messages), or until
            # the kernel stop time is reached.
            while message_queue and self.current_time <= stop_time:
                # Get the next message in timestamp order (delivery time) and extract it.
                self.current_time, (agent_id, msg) = message_queue.get()

                # Periodically print the simulation time and total messages, even if muted.
                if not total_messages % 100_000:
                    print(
                        f"\n--- Simulation time: {self.fmtTime(self.current_time)}, "
                        f"messages processed: {total_messages}, "
                        f"wallclock elapsed: {pd.Timestamp('now') - eventQueueWallClockStart} ---\n"
                    )

                msg_class_name = msg.__class__.__name__
                log_print(
                    "\n--- Kernel Event Queue pop ---\n"
                    f"Kernel handling {msg_class_name} message for agent {agent_id} "
                    f"at time {self.fmtTime(self.current_time)}"
                )

                total_messages += 1

                # In between messages, always reset the currentAgentAdditionalDelay.
                self.current_agent_additional_delay = 0

                # Test to see if the agent is already in the future. If so,
                # delay the wakeup until the agent can act again.
                agent_current_time = agent_current_times[agent_id]
                if agent_current_time > self.current_time:
                    # Push the wakeup call back into the PQ with a new time.
                    message_queue.put(
                        (agent_current_time, (agent_id, msg))
                    )
                    log_print(f"Agent in future: {msg_class_name} requeued for {self.fmtTime(agent_current_time)}")
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                agent_current_times[agent_id] = self.current_time

                if isinstance(msg, Message):
                    agents[agent_id].receiveMessage(self.current_time, msg)
                    called_method_name = "receiveMessage"
                elif isinstance(msg, WakeUp):
                    agents[agent_id].wakeup(self.current_time)
                    called_method_name = "wakeup"
                else:
                    raise ValueError(
                        "Unknown message type found in queue",
                        "currentTime:",
                        self.current_time,
                        "messageType:",
                        msg_class_name
                    )

                # Delay the agent by its computation delay plus any transient additional delay requested.
                agent_current_times[agent_id] += pd.Timedelta(
                    agent_computation_delays[agent_id] + self.current_agent_additional_delay
                )

                log_print(
                    f"After {called_method_name} return, agent {agent_id} "
                    f"delayed from {self.fmtTime(self.current_time)} to {self.fmtTime(agent_current_times[agent_id])}"
                )

            if not message_queue:
                log_print("\n--- Kernel Event Queue empty ---")
            elif self.current_time > self.stop_time:
                log_print("\n--- Kernel Stop Time surpassed ---")

            # Record wall clock stop time and elapsed time for stats at the end.
            eventQueueWallClockStop = pd.Timestamp('now')
            event_queue_wallclock_elapsed = eventQueueWallClockStop - eventQueueWallClockStart

            # Event notification for kernel end (agents may communicate with
            # other agents, as all agents are still guaranteed to exist).
            # Agents should not destroy resources they may need to respond
            # to final communications from other agents.
            log_print("\n--- Agent.kernelStopping() ---")
            for agent in agents:
                agent.kernelStopping()

            # Event notification for kernel termination (agents should not
            # attempt communication with other agents, as order of termination
            # is unknown). Agents should clean up all used resources as the
            # simulation program may not actually terminate if num_simulations > 1.
            log_print("\n--- Agent.kernelTerminating() ---")
            for agent in agents:
                agent.kernelTerminating()

            print(
                f"Event Queue elapsed: {event_queue_wallclock_elapsed}, "
                f"messages: {total_messages}, "
                f"messages per second: {total_messages / (event_queue_wallclock_elapsed / one_s_timedelta):0.1f}"
            )
            log_print(f"Ending sim {sim}")

        # The Kernel adds a handful of custom state results for all simulations,
        # which configurations may use, print, log, or discard.
        custom_state['kernel_event_queue_elapsed_wallclock'] = event_queue_wallclock_elapsed
        custom_state['kernel_slowest_agent_finish_time'] = max(agent_current_times)

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

        return custom_state

    def sendMessage(self,
                    sender_id: int,
                    recipient_id: int,
                    msg: MessageAbstractBase,
                    delay: int = 0) -> None:
        # Called by an agent to send a message to another agent. The kernel
        # supplies its own currentTime (i.e. "now") to prevent possible
        # abuse by agents. The kernel will handle computational delay penalties
        # and/or network latency. The message must derive from the message.Message class.
        # The optional delay parameter represents an agent's request for ADDITIONAL
        # delay (beyond the Kernel's mandatory computation + latency delays) to represent
        # parallel pipeline processing delays (that should delay the transmission of messages
        # but do not make the agent "busy" and unable to respond to new messages).

        # Apply the agent's current computation delay to effectively "send" the message
        # at the END of the agent's current computation period when it is done "thinking".
        # NOTE: sending multiple messages on a single wake will transmit all at the same
        # time, at the end of computation. To avoid this, use Agent.delay() to accumulate
        # a temporary delay (current cycle only) that will also stagger messages.

        # The optional pipeline delay parameter DOES push the send time forward, since it
        # represents "thinking" time before the message would be sent. We don't use this
        # for much yet, but it could be important later.

        # This means message delay (before latency) is the agent's standard computation delay
        # PLUS any accumulated delay for this wake cycle PLUS any one-time requested delay
        # for this specific message only.
        sentTime = self.current_time + pd.Timedelta(
            self.agent_computation_delays[sender_id] + self.current_agent_additional_delay + delay
        )

        # Apply communication delay per the agentLatencyModel, if defined, or the
        # agentLatency matrix [sender][recipient] otherwise.

        latency, noise = self.agent_latency_model.get_latency_and_noise(sender_id, recipient_id)
        deliverAt = sentTime + pd.Timedelta(latency + noise)
        log_print(
            f"Kernel applied latency {latency}, noise {noise}, "
            f"accumulated delay {self.current_agent_additional_delay}, one-time delay {delay} "
            f"on sendMessage from: {self.agents[sender_id].name} to {self.agents[recipient_id].name}, "
            f"scheduled for {self.fmtTime(deliverAt)}"
        )

        # Finally drop the message in the queue with priority == delivery time.
        self.message_queue.put(
            (deliverAt, (recipient_id, msg))
        )

        log_print(
            f"Sent time: {sentTime}, current time {self.current_time}, "
            f"computation delay {self.agent_computation_delays[sender_id]}\n"
            f"Message queued: {msg}"
        )

    def setWakeup(self, sender_id: int, requested_time: Optional[pd.Timestamp] = None) -> None:
        # Called by an agent to receive a "wakeup call" from the kernel
        # at some requested future time. Defaults to the next possible
        # timestamp. Wakeup time cannot be the current time or a past time.
        # Sender is required and should be the ID of the agent making the call.
        # The agent is responsible for maintaining any required state; the
        # kernel will not supply any parameters to the wakeup() call.

        if requested_time is None:
            requested_time = self.current_time + one_ns_timedelta
        elif requested_time < self.current_time:
            raise ValueError(
                "setWakeup() called with requested time not in future",
                "currentTime:",
                self.current_time,
                "requestedTime:",
                requested_time
            )

        log_print(f"Kernel adding wakeup for agent {sender_id} at time {self.fmtTime(requested_time)}")

        self.message_queue.put(
            (requested_time, (sender_id, WakeUp()))
        )

    def getAgentComputeDelay(self, sender_id: int) -> int:
        # Allows an agent to query its current computation delay.
        return self.agent_computation_delays[sender_id]

    def setAgentComputeDelay(self, sender_id: int, requested_delay: int) -> None:
        # Called by an agent to update its computation delay.  This does
        # not initiate a global delay, nor an immediate delay for the
        # agent. Rather it sets the new default delay for the calling
        # agent. The delay will be applied upon every return from wakeup
        # or recvMsg. Note that this delay IS applied to any messages
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

        self.agent_computation_delays[sender_id] = requested_delay

    def delayAgent(self, additional_delay: int) -> None:
        # Called by an agent to accumulate temporary delay for the current wake cycle.
        # This will apply the total delay (at time of sendMessage) to each message,
        # and will modify the agent's next available time slot. These happen on top
        # of the agent's compute delay BUT DO NOT ALTER IT. (i.e. effects are transient)
        # Mostly useful for staggering outbound messages.

        # additionalDelay should be in whole nanoseconds.
        if not isinstance(additional_delay, int):
            raise TypeError(
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

    def findAgentByType(self, agent_type: Type['Agent']) -> Optional[int]:
        # Called to request an arbitrary agent ID that matches the class or base class
        # passed as "type". For example, any ExchangeAgent, or any NasdaqExchangeAgent.
        # This method is rather expensive, so the results should be cached by the caller!

        for agent in self.agents:
            if isinstance(agent, agent_type):
                return agent.id
        return None

    def writeLog(self,
                 sender_id: int,
                 log_df: pd.DataFrame,
                 filename: Optional[Union[str, PathLike, Path]] = None) -> None:
        # Called by any agent, usually at the very end of the simulation just before
        # kernel shutdown, to write to disk any log dataframe it has been accumulating
        # during simulation. The format can be decided by the agent, although changes
        # will require a special tool to read and parse the logs. The Kernel places
        # the log in a unique directory per run, with one filename per agent, also
        # decided by the Kernel using agent type, id, etc.

        # If there are too many agents, placing all these files in a directory might
        # be unfortunate. Also if there are too many agents, or if the logs are too
        # large, memory could become an issue. In this case, we might have to take
        # a speed hit to write logs incrementally.

        # If filename is not None, it will be used as the filename. Otherwise,
        # the Kernel will construct a filename based on the name of the Agent
        # requesting log archival.

        if self.skip_log:
            return

        path = joinpath("..", "log", self.log_dir)
        file = f"{filename or self.agents[sender_id].name.replace(' ', '')}.bz2"

        makedirs(path, exist_ok=True)
        log_df.to_pickle(joinpath(path, file), compression='bz2')

    def appendSummaryLog(self, sender_id: int, event_type: str, event: Event) -> None:
        # We don't even include a timestamp, because this log is for one-time-only
        # summary reporting, like starting cash, or ending cash.
        self.summary_log.append(
            {
                'AgentID': sender_id,
                'AgentStrategy': self.agents[sender_id].type,
                'EventType': event_type,
                'Event': event
            }
        )

    def writeSummaryLog(self) -> None:
        path = joinpath("..", "log", self.log_dir)
        file = "summary_log.bz2"

        makedirs(path, exist_ok=True)
        dfLog = pd.DataFrame(self.summary_log)
        dfLog.to_pickle(joinpath(path, file), compression='bz2')

    def updateAgentState(self, agent_id: int, state: Any) -> None:
        """Called by an agent that wishes to replace its custom state in the dictionary
           the Kernel will return at the end of simulation. Shared state must be set directly,
           and agents should coordinate that non-destructively.

           Note that it is never necessary to use this kernel state dictionary for an agent
           to remember information about itself, only to report it back to the config file.
        """
        custom_state = self.custom_state
        if 'agent_state' in custom_state:
            custom_state['agent_state'][agent_id] = state
        else:
            custom_state['agent_state'] = {agent_id: state}

    @staticmethod
    def fmtTime(simulation_time: pd.Timestamp) -> pd.Timestamp:
        # The Kernel class knows how to pretty-print time. It is assumed simulationTime
        # is in nanoseconds since midnight. Note this is a static method which can be
        # called either on the class or an instance.

        # Try just returning the pd.Timestamp now.
        return simulation_time

        # ns = simulationTime
        # hr = int(ns / (1000000000 * 60 * 60))
        # ns -= (hr * 1000000000 * 60 * 60)
        # m = int(ns / (1000000000 * 60))
        # ns -= (m * 1000000000 * 60)
        # s = int(ns / 1000000000)
        # ns = int(ns - (s * 1000000000))
        #
        # return "{:02d}:{:02d}:{:02d}.{:09d}".format(hr, m, s, ns)


class Agent:
    __slots__ = (
        "id",
        "name",
        "kernel",
        "log",
        "current_time",
        "log_to_file",
        "random_state"
    )

    def __init__(self,
                 agent_id: int,
                 name: str,
                 random_state: np.random.RandomState,
                 log_to_file: bool = True) -> None:

        # ID must be a unique number (usually auto-incremented).
        # Name is for human consumption, should be unique (often type + number).
        # Type is for machine aggregation of results, should be same for all
        # agents following the same strategy (incl. parameter settings).
        # Every agent is given a random state to use for any stochastic needs.
        # This is an np.random.RandomState object, already seeded.
        self.id = agent_id
        self.name = name
        self.log_to_file = log_to_file
        self.random_state = random_state

        if not isinstance(random_state, np.random.RandomState):
            raise ValueError(
                "A valid, seeded np.random.RandomState object is required for every agent.Agent",
                self.name
            )

        # Kernel is supplied via kernelInitializing method of kernel lifecycle.
        self.kernel: Kernel = None  # type: ignore

        # What time does the agent think it is? Should be updated each time
        # the agent wakes via wakeup or receiveMessage. (For convenience
        # of reference throughout the Agent class hierarchy, NOT THE
        # CANONICAL TIME.)
        self.current_time: pd.Timestamp = None  # type: ignore

        # Agents may choose to maintain a log. During simulation,
        # it should be stored as a list of dictionaries. The expected
        # keys by default are: EventTime, EventType, Event. Other
        # Columns may be added, but will then require specializing
        # parsing and will increase output dataframe size. If there
        # is a non-empty log, it will be written to disk as a Dataframe
        # at kernel termination.

        # It might, or might not, make sense to formalize these log Events
        # as a class, with enumerated EventTypes and so forth.
        self.log: List[AgentEventLogEntry] = []
        self.logEvent("AGENT_TYPE", self.type)

    @classmethod
    @property
    def type(cls) -> str:
        return cls.__name__

    # Flow of required kernel listening methods:
    # init -> start -> (entire simulation) -> end -> terminate

    def kernelInitializing(self, kernel: Kernel) -> None:
        # Called by kernel one time when simulation first begins.
        # No other agents are guaranteed to exist at this time.

        # Kernel reference must be retained, as this is the only time the
        # agent can "see" it.

        self.kernel = kernel
        log_print(f"{self.name} exists!")

    def kernelStarting(self, start_time: pd.Timestamp) -> None:
        # Called by kernel one time _after_ simulationInitializing.
        # All other agents are guaranteed to exist at this time.
        # startTime is the earliest time for which the agent can
        # schedule a wakeup call (or could receive a message).

        # Base Agent schedules a wakeup call for the first available timestamp.
        # Subclass agents may override this behavior as needed.

        log_print(
            f"Agent {self.id} ({self.name}) requesting kernel wakeup at time {self.kernel.fmtTime(start_time)}"
        )
        self.setWakeup(start_time)

    def kernelStopping(self) -> None:
        # Called by kernel one time _before_ simulationTerminating.
        # All other agents are guaranteed to exist at this time.
        pass

    def kernelTerminating(self) -> None:
        # Called by kernel one time when simulation terminates.
        # No other agents are guaranteed to exist at this time.

        # If this agent has been maintaining a log, convert it to a Dataframe
        # and request that the Kernel write it to disk before terminating.
        if self.log and self.log_to_file:
            dfLog = pd.DataFrame(self.log)
            dfLog.set_index('EventTime', inplace=True)
            self.writeLog(dfLog)

    # >>> Methods for internal use by agents (e.g. bookkeeping) >>>
    def logEvent(self,
                 event_type: str,
                 event: Event = '',
                 append_summary_log: bool = False) -> None:
        # Adds an event to this agent's log. The deepcopy of the Event field,
        # often an object, ensures later state changes to the object will not
        # retroactively update the logged event.

        # We can make a single copy of the object (in case it is an arbitrary
        # class instance) for both potential log targets, because we don't
        # alter logs once recorded.
        e = deepcopy(event)
        self.log.append(
            {
                'EventTime': self.current_time,
                'EventType': event_type,
                'Event': e
            }
        )

        if append_summary_log:
            self.kernel.appendSummaryLog(self.id, event_type, e)

    # >>> Methods required for communication from other agents >>>
    # The kernel will _not_ call these methods on its own behalf,
    # only to pass traffic from other agents..

    def receiveMessage(self, current_time: pd.Timestamp, msg: Message) -> None:
        # Called each time a message destined for this agent reaches
        # the front of the kernel's priority queue. currentTime is
        # the simulation time at which the kernel is delivering this
        # message -- the agent should treat this as "now". msg is
        # an object guaranteed to inherit from the message.Message class.

        log_print(
            f"At {self.kernel.fmtTime(self.current_time)}, agent {self.id} ({self.name}) received: {msg}"
        )

    def wakeup(self, current_time: pd.Timestamp) -> None:
        # Agents can request a wakeup call at a future simulation time using
        # Agent.setWakeup(). This is the method called when the wakeup time
        # arrives.

        self.current_time = current_time
        log_print(
            f"At {self.kernel.fmtTime(current_time)}, agent {self.id} ({self.name}) received wakeup."
        )

    # Methods used to request services from the Kernel. These should be used
    # by all agents. Kernel methods should _not_ be called directly!

    # Presently the kernel expects agent IDs only, not agent references.
    # It is possible this could change in the future. Normal agents will
    # not typically wish to request additional delay.
    def sendMessage(self, recipient_id: int, msg: MessageAbstractBase, delay: int = 0) -> None:
        self.kernel.sendMessage(self.id, recipient_id, msg, delay)

    def setWakeup(self, requested_time: pd.Timestamp) -> None:
        self.kernel.setWakeup(self.id, requested_time)

    def getComputationDelay(self) -> int:
        return self.kernel.getAgentComputeDelay(self.id)

    def setComputationDelay(self, requested_delay: int) -> None:
        self.kernel.setAgentComputeDelay(self.id, requested_delay)

    def delay(self, additional_delay: int) -> None:
        self.kernel.delayAgent(additional_delay)

    def writeLog(self, df_log: pd.DataFrame, filename: Optional[Union[str, PathLike, Path]] = None) -> None:
        self.kernel.writeLog(self.id, df_log, filename)

    def updateAgentState(self, state: Any) -> None:
        """
        Agents should use this method to replace their custom state in the dictionary
        the Kernel will return to the experimental config file at the end of the
        simulation. This is intended to be write-only, and agents should not use
        it to store information for their own later use.
        """
        self.kernel.updateAgentState(self.id, state)

    # Internal methods that should not be modified without a very good reason.
    # def __lt__(self, other: 'Agent') -> bool:
    #     # Required by Python3 for this object to be placed in a priority queue.
    #     return f"{self.id}" < f"{other.id}"
