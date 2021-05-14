from typing import List, Tuple

import pandas as pd

from backtesting.message.base import Message
from backtesting.order.base import Order
from backtesting.typing.exchange import OrderBookHistoryStep

__all__ = (
    "Reply",

    "MarketClosedReply",

    "OrderReply",
    "OrderAccepted",
    "OrderCancelled",
    "OrderExecuted",
    "OrderModified",

    "MarketOpeningHourReply",
    "WhenMktOpenReply",
    "WhenMktCloseReply",

    "QueryReplyMessage",
    "QueryLastTradeReply",
    "QueryLastSpreadReply",
    "QueryOrderStreamReply",
    "QueryTransactedVolumeReply"
)


class Reply(Message):
    __slots__ = ()


# ___________________ MARKET CLOSED REPLY ___________________

class MarketClosedReply(Reply):
    type = "MKT_CLOSED"
    __slots__ = ()


# ___________________ ORDER REPLIES ___________________

class OrderReply(Reply):
    __slots__ = ("sender_id", "order")

    def __init__(self, sender_id: int, order: Order) -> None:
        super().__init__(sender_id)
        self.order = order


class OrderAccepted(OrderReply):
    type = "ORDER_ACCEPTED"
    __slots__ = ()


class OrderCancelled(OrderReply):
    type = "ORDER_CANCELLED"
    __slots__ = ()


class OrderExecuted(OrderReply):
    type = "ORDER_EXECUTED"
    __slots__ = ()


class OrderModified(OrderReply):
    type = "ORDER_MODIFIED"
    __slots__ = ()


# ___________________ EXCHANGE OPENING HOUR REPLIES ___________________

class MarketOpeningHourReply(Reply):
    __slots__ = ("timestamp",)

    def __init__(self, sender_id: int, time: pd.Timestamp) -> None:
        super().__init__(sender_id)
        self.timestamp = time


class WhenMktOpenReply(MarketOpeningHourReply):
    type = "WHEN_MKT_OPEN_REPLY"
    __slots__ = ()


class WhenMktCloseReply(MarketOpeningHourReply):
    type = "WHEN_MKT_CLOSE_REPLY"
    __slots__ = ()


# ___________________ QUERY REPLIES ___________________

class QueryReplyMessage(Reply):
    __slots__ = ("symbol", "mkt_closed")

    def __init__(self, sender_id: int, symbol: str, mkt_closed: bool) -> None:
        super().__init__(sender_id)
        self.symbol = symbol
        self.mkt_closed = mkt_closed


class QueryLastTradeReply(QueryReplyMessage):
    type = "QUERY_LAST_TRADE_REPLY"
    __slots__ = ("price",)

    def __init__(self, sender_id: int, symbol: str, mkt_closed: bool, price: int) -> None:
        super().__init__(sender_id, symbol, mkt_closed)
        self.price = price


class QueryLastSpreadReply(QueryReplyMessage):
    type = "QUERY_LAST_SPREAD_REPLY"
    __slots__ = ("depth", "bids", "asks", "last_spread")

    def __init__(self,
                 sender_id: int,
                 symbol: str,
                 mkt_closed: bool,
                 depth: int,
                 *,
                 bids: List[Tuple[int, int]],
                 asks: List[Tuple[int, int]],
                 spread: int) -> None:
        super().__init__(sender_id, symbol, mkt_closed)
        self.depth = depth
        self.bids = bids
        self.asks = asks
        self.last_spread = spread


class QueryOrderStreamReply(QueryReplyMessage):
    type = "QUERY_ORDER_STREAM_REPLY"
    __slots__ = ("length", "orders")

    def __init__(self,
                 sender_id: int,
                 symbol: str,
                 mkt_closed: bool,
                 length: int,
                 orders: Tuple[OrderBookHistoryStep, ...]) -> None:
        super().__init__(sender_id, symbol, mkt_closed)
        self.length = length
        self.orders = orders


class QueryTransactedVolumeReply(QueryReplyMessage):
    type = "QUERY_TRANSACTED_VOLUME_REPLY"
    __slots__ = ("transacted_volume",)

    def __init__(self,
                 sender_id: int,
                 symbol: str,
                 mkt_closed: bool,
                 transacted_volume: int) -> None:
        super().__init__(sender_id, symbol, mkt_closed)
        self.transacted_volume = transacted_volume
