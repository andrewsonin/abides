from typing import Dict, TypedDict, Any

import pandas as pd

from backtesting.typing import Event

__all__ = (
    "KernelSummaryLogEntry",
    "KernelCustomState",
    "AgentEventLogEntry"
)


class KernelSummaryLogEntry(TypedDict):
    AgentID: int
    AgentStrategy: str
    EventType: str
    Event: Event


KernelCustomState = Dict[str, Any]


class AgentEventLogEntry(TypedDict):
    EventTime: pd.Timestamp
    EventType: str
    Event: Event
