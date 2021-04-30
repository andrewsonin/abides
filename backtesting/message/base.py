from abc import ABCMeta, abstractmethod

from backtesting.utils.util import get_defined_slots

__all__ = (
    "MessageAbstractBase",
    "WakeUp",
    "Message"
)


class MessageAbstractBase(metaclass=ABCMeta):
    __slots__ = ("id",)
    _counter = 0

    def __init__(self) -> None:
        # The auto-incrementing variable here will ensure that, when Messages are
        # due for delivery at the same time step, the Message that was created
        # first is delivered first. (Which is not important, but Python 3
        # requires a fully resolved chain of priority in all cases, so we need
        # something consistent.)  We might want to generate these with stochasticity,
        # but guarantee uniqueness somehow, to make delivery of orders at the same
        # exact timestamp "random" instead of "arbitrary" (FIFO among tied times)
        # as it currently is.
        self.id = MessageAbstractBase._counter
        MessageAbstractBase._counter += 1
        # The base Message class no longer holds envelope/header information,
        # however any desired information can be placed in the arbitrary
        # body. Delivery metadata is now handled outside the message itself.
        # The body may be overridden by specific message type subclasses.
        # It is acceptable for WAKEUP type messages to have no body.

        # The base Message class can no longer do any real error checking.
        # Subclasses are strongly encouraged to do so based on their body.

    def __lt__(self, other: 'MessageAbstractBase') -> bool:
        self_mtp = self.msg_type_priority
        other_mtp = other.msg_type_priority
        return self_mtp < other_mtp or self_mtp == other_mtp and self.id < other.id

    get_defined_slots = classmethod(get_defined_slots)

    @property
    @abstractmethod
    def msg_type_priority(self) -> int:
        """Needed for setting different priorities to the messages in PriorityQueue"""


class WakeUp(MessageAbstractBase):
    __slots__ = ()
    msg_type_priority = 1


class Message(MessageAbstractBase):
    __slots__ = ("sender_id",)
    msg_type_priority = 0

    def __init__(self, sender_id: int) -> None:
        super().__init__()
        self.sender_id = sender_id

    def __str__(self) -> str:
        """Make a printable representation of the message"""
        return f"{{{', '.join(f'{slot}: {getattr(self, slot)}' for slot in self.get_defined_slots())}}}"

    @property
    @abstractmethod
    def type(self) -> str:
        pass
