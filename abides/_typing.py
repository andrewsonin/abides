from typing import Dict, TypedDict, Literal, Any

import pandas as pd

MessageType = Literal[
    'LIMIT_ORDER',
    'MARKET_ORDER',
    'CANCEL_ORDER',
    'MODIFY_ORDER',

    'MKT_CLOSED',
    'MARKET_DATA',
]


class MessageBody(TypedDict):
    msg: MessageType


Event = Any


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
