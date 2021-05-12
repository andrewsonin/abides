from enum import Enum

__all__ = (
    "State",
    "DefaultState",
    "AwaitingWakeUp"
)


class State(Enum):
    DEFAULT_STATE = "DEFAULT_STATE"
    AWAITING_WAKEUP = "AWAITING_WAKEUP"


DefaultState = State.DEFAULT_STATE
AwaitingWakeUp = State.AWAITING_WAKEUP
