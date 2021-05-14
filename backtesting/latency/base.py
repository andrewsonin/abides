from abc import ABCMeta, abstractmethod
from typing import Tuple

__all__ = (
    "AgentLatencyModelBase",
)


class AgentLatencyModelBase(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def get_latency_and_noise(self, sender_id: int, recipient_id: int) -> Tuple[int, int]:
        """
        Get base latency and generate additional noise for connection between sender and recipient.

        Args:
            sender_id:     sender ID
            recipient_id:  recipient ID

        Returns:
            (latency, noise)
        """
