from typing import Tuple, Sequence

import numpy as np

from backtesting.latency.base import AgentLatencyModelBase

__all__ = (
    "DefaultAgentLatencyModel",
    "AgentLatencyModel"
)


class DefaultAgentLatencyModel(AgentLatencyModelBase):
    """Naive latency model without any noise simulation"""
    __slots__ = ("default_latency",)

    def __init__(self, default_latency: int) -> None:
        """
        Naive latency model without any noise simulation.

        Args:
            default_latency:  default latency in nanoseconds
        """
        self.default_latency = default_latency

    def get_latency_and_noise(self, sender_id: int, recipient_id: int) -> Tuple[int, int]:
        return self.default_latency, 0


class AgentLatencyModel(AgentLatencyModelBase):
    """Latency model with pre-defined latency matrix and noise probability mass function"""
    __slots__ = ("latency_matrix", "noise_probs", "random_state")

    def __init__(self,
                 latency_matrix: Sequence[Sequence[int]],
                 noise_probs: Sequence[float],
                 random_state: np.random.RandomState) -> None:
        """
        Latency model with pre-defined latency matrix and noise probability mass function.

        Args:
            latency_matrix:  Defines the communication delay between every pair of agents.
                             The first dimension refers to the sender ID, the second — to the recipient one
            noise_probs:     list with list index = ns extra delay, value = probability of this delay.
            random_state:    np.random.RandomState used for to sample noise
        """
        noise_probs = np.asarray(noise_probs)  # type: ignore
        if (noise_probs < 0).any() or not np.isclose(noise_probs.sum(), 1):  # type: ignore
            raise ValueError("Parameter 'noise_probs' should define the array of probabilities that add up to 1")
        self.latency_matrix = latency_matrix
        # There is a noise model for latency, intended to be a one-sided
        # distribution with the peak at zero. By default there is no noise
        # (100% chance to add zero ns extra delay).
        self.noise_probs = noise_probs
        self.random_state = random_state

    def get_latency_and_noise(self, sender_id: int, recipient_id: int) -> Tuple[int, int]:
        latency = self.latency_matrix[sender_id][recipient_id]
        noise = self.random_state.choice(len(self.noise_probs), p=self.noise_probs)
        return latency, noise
