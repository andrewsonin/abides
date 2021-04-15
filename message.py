from abc import ABC, abstractmethod
from typing import Any

__all__ = (
    "MessageAbstractBase",
    "Message",
    "WakeUp"
)


class MessageAbstractBase(ABC):
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

    def __lt__(self, other: 'MessageAbstractBase') -> bool:
        return (self.msg_type_priority, self.id) < (other.msg_type_priority, other.id)

    @property
    @abstractmethod
    def msg_type_priority(self) -> int:
        pass


class Message(MessageAbstractBase):
    __slots__ = ("body",)
    msg_type_priority = 0

    def __init__(self, body: Any = None) -> None:
        # The base Message class no longer holds envelope/header information,
        # however any desired information can be placed in the arbitrary
        # body. Delivery metadata is now handled outside the message itself.
        # The body may be overridden by specific message type subclasses.
        # It is acceptable for WAKEUP type messages to have no body.
        super().__init__()
        self.body = body
        # The base Message class can no longer do any real error checking.
        # Subclasses are strongly encouraged to do so based on their body.

    def __str__(self) -> str:
        # Make a printable representation of this message.
        return str(self.body)


class WakeUp(MessageAbstractBase):
    __slots__ = ()
    msg_type_priority = 1
