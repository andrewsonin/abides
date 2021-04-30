from typing import TypedDict, Dict, Tuple, List

import pandas as pd

__all__ = (
    "OrderBookHistoryStep",
    "OrderBookHistoryEntry",
    "LimitOrderChangeTimeAndVolume"
)

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
