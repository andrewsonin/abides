from typing import Tuple, List, Dict, TypedDict, Any

import pandas as pd

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


OrderBookHistoryStep = Dict[int, 'OrderBookHistoryEntry']
LimitOrderChangeTimeAndVolume = Tuple[pd.Timestamp, int]


class OrderBookHistoryEntry(TypedDict):
    entry_time: pd.Timestamp
    quantity: int
    is_buy_order: bool
    limit_price: int
    transactions: List[LimitOrderChangeTimeAndVolume]
    modifications: List[LimitOrderChangeTimeAndVolume]
    cancellations: List[LimitOrderChangeTimeAndVolume]
