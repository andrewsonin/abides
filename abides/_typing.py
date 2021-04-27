from typing import Tuple, List, Dict, TypedDict, Literal, Any

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


class OrderBookTransactedVolume(TypedDict):
    unrolled_transactions: Any
    history_previous_length: int


LimitOrderChangeTimeAndVolume = Tuple[pd.Timestamp, int]


class OrderBookHistoryEntry(TypedDict):
    order_id: int
    entry_time: pd.Timestamp
    quantity: int
    is_buy_order: bool
    limit_price: int
    transactions: List[LimitOrderChangeTimeAndVolume]
    modifications: List[LimitOrderChangeTimeAndVolume]
    cancellations: List[LimitOrderChangeTimeAndVolume]
