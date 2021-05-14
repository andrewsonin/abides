from typing import List, Tuple

import pandas as pd

from backtesting.message.base import Message

__all__ = (
    "Notification",

    "MarketData"
)


class Notification(Message):
    __slots__ = ()


# ___________________ MARKET DATA NOTIFICATION ___________________

class MarketData(Notification):
    type = "MARKET_DATA"
    __slots__ = ("symbol", "last_transaction", "exchange_ts", "bids", "asks")

    def __init__(self,
                 agent_id: int,
                 symbol: str,
                 last_transaction: int,
                 exchange_ts: pd.Timestamp,
                 *,
                 bids: List[Tuple[int, int]],
                 asks: List[Tuple[int, int]]) -> None:
        super().__init__(agent_id)
        self.symbol = symbol
        self.last_transaction = last_transaction
        self.exchange_ts = exchange_ts
        self.bids = bids
        self.asks = asks
