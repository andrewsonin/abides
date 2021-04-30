from typing import Tuple, List, Optional

import pandas as pd

from backtesting.message.base import Message
from backtesting.order.base import Order, MarketOrder, LimitOrder

__all__ = (
    "MarketClosedReply",
    "MarketData",

    "OrderRequest",
    "LimitOrderRequest",
    "MarketOrderRequest",
    "CancelOrderRequest",
    "ModifyOrderRequest",

    "OrderReply",
    "OrderAccepted",
    "OrderCancelled",
    "OrderExecuted",
    "OrderModified",

    "MarketDataSubscription",
    "MarketDataSubscriptionRequest",
    "MarketDataSubscriptionCancellation",

    "MarketOpeningHourRequest",
    "WhenMktOpen",
    "WhenMktClose",

    "MarketOpeningHourReply",
    "WhenMktOpenReply",
    "WhenMktCloseReply",

    "Query",
    "QueryLastTrade",
    "QuerySpread",
    "QueryOrderStream",
    "QueryTransactedVolume",

    "QueryLastTradeReply",
    "QueryLastSpreadReply",
    "QueryOrderStreamReply",
    "QueryTransactedVolumeReply"
)


class MarketClosedReply(Message):
    type = "MKT_CLOSED"
    __slots__ = ()


class MarketData(Message):
    type = "MARKET_DATA"
    __slots__ = ("symbol", "last_transaction", "exchange_ts", "bids", "asks")

    def __init__(self,
                 agent_id: int,
                 symbol: str,
                 last_transaction: Optional[int],
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


# >>> ORDER REQUEST CLASSES >>>
class OrderRequest(Message):
    __slots__ = ("sender_id", "order")

    def __init__(self, sender_id: int, order: Order) -> None:
        super().__init__(sender_id)
        self.order = order


class LimitOrderRequest(OrderRequest):
    type = "LIMIT_ORDER"
    __slots__ = ()
    order: LimitOrder


class MarketOrderRequest(OrderRequest):
    type = "MARKET_ORDER"
    __slots__ = ()
    order: MarketOrder


class CancelOrderRequest(OrderRequest):
    type = "CANCEL_ORDER"
    __slots__ = ()
    order: LimitOrder


class ModifyOrderRequest(OrderRequest):
    type = "MODIFY_ORDER"
    __slots__ = ("new_order",)
    order: LimitOrder

    def __init__(self, sender_id: int, order: LimitOrder, new_order: LimitOrder) -> None:
        super().__init__(sender_id, order)
        self.new_order = new_order


# <<< ORDER REQUEST CLASSES <<<


# >>> ORDER REPLY CLASSES >>>
class OrderReply(Message):
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


# <<< ORDER REPLY CLASSES <<<


# >>> MARKET DATA SUBSCRIPTION MESSAGES >>>

class MarketDataSubscription(Message):
    __slots__ = ("symbol",)

    def __init__(self, sender_id: int, symbol: str) -> None:
        super().__init__(sender_id)
        self.symbol = symbol


class MarketDataSubscriptionRequest(MarketDataSubscription):
    type = "MARKET_DATA_SUBSCRIPTION_REQUEST"
    __slots__ = ("levels", "freq")

    def __init__(self, sender_id: int, symbol: str, *, levels: int, freq: int) -> None:
        super().__init__(sender_id, symbol)
        self.levels = levels
        self.freq = freq


class MarketDataSubscriptionCancellation(MarketDataSubscription):
    type = "MARKET_DATA_SUBSCRIPTION_CANCELLATION"
    __slots__ = ()


# <<< MARKET DATA SUBSCRIPTION MESSAGES <<<

# >>> EXCHANGE OPENING HOUR MESSAGES >>>

class MarketOpeningHourRequest(Message):
    __slots__ = ()


class WhenMktOpen(MarketOpeningHourRequest):
    type = "WHEN_MKT_OPEN"
    __slots__ = ()


class WhenMktClose(MarketOpeningHourRequest):
    type = "WHEN_MKT_CLOSE"
    __slots__ = ()


class MarketOpeningHourReply(Message):
    __slots__ = ("data",)

    def __init__(self, sender_id: int, data: pd.Timestamp) -> None:
        super().__init__(sender_id)
        self.data = data


class WhenMktOpenReply(MarketOpeningHourReply):
    type = "WHEN_MKT_OPEN_REPLY"
    __slots__ = ()


class WhenMktCloseReply(MarketOpeningHourReply):
    type = "WHEN_MKT_CLOSE_REPLY"
    __slots__ = ()


# <<< EXCHANGE OPENING HOUR MESSAGES <<<


class Query(Message):
    __slots__ = ("symbol",)

    def __init__(self, sender_id: int, symbol: str) -> None:
        super().__init__(sender_id)
        self.symbol = symbol


class QueryLastTrade(Query):
    type = "QUERY_LAST_TRADE"
    __slots__ = ()


class QuerySpread(Query):
    type = "QUERY_SPREAD"
    __slots__ = ("depth",)

    def __init__(self, sender_id: int, symbol: str, depth: int) -> None:
        super().__init__(sender_id, symbol)
        self.depth = depth


class QueryOrderStream(Query):
    type = "QUERY_ORDER_STREAM"
    __slots__ = ("length",)

    def __init__(self, sender_id: int, symbol: str, length: int) -> None:
        super().__init__(sender_id, symbol)
        self.length = length


class QueryTransactedVolume(Query):
    type = "QUERY_TRANSACTED_VOLUME"
    __slots__ = ("lookback_period",)

    def __init__(self, sender_id: int, symbol: str, lookback_period: int) -> None:
        super().__init__(sender_id, symbol)
        self.lookback_period = lookback_period


class QueryReplyMessage(Message):
    __slots__ = ("symbol", "mkt_closed")

    def __init__(self, sender_id: int, symbol: str, mkt_closed: bool) -> None:
        super().__init__(sender_id)
        self.symbol = symbol
        self.mkt_closed = mkt_closed


class QueryLastTradeReply(QueryReplyMessage):
    type = "QUERY_LAST_TRADE_REPLY"
    __slots__ = ("data",)

    def __init__(self, sender_id: int, symbol: str, mkt_closed: bool, data: Optional[int]) -> None:
        super().__init__(sender_id, symbol, mkt_closed)
        self.data = data


class QueryLastSpreadReply(QueryReplyMessage):
    type = "QUERY_LAST_SPREAD_REPLY"
    __slots__ = ("depth", "bids", "asks", "data", "book")

    def __init__(self,
                 sender_id: int,
                 symbol: str,
                 mkt_closed: bool,
                 depth: int,
                 *,
                 bids: List[Tuple[int, int]],
                 asks: List[Tuple[int, int]],
                 data: Optional[int],
                 book: str) -> None:
        super().__init__(sender_id, symbol, mkt_closed)
        self.depth = depth
        self.bids = bids
        self.asks = asks
        self.data = data
        self.book = book


class QueryOrderStreamReply(QueryReplyMessage):
    type = "QUERY_ORDER_STREAM_REPLY"
    __slots__ = ("length", "orders")

    def __init__(self,
                 sender_id: int,
                 symbol: str,
                 mkt_closed: bool,
                 length: int,
                 orders) -> None:
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
