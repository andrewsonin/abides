from typing import Union

import pandas as pd

from backtesting.message.base import Message
from backtesting.order.base import Order, MarketOrder, LimitOrder

__all__ = (
    "Request",

    "OrderRequest",
    "LimitOrderRequest",
    "MarketOrderRequest",
    "CancelOrderRequest",
    "ModifyOrderRequest",

    "MarketDataSubscription",
    "MarketDataSubscriptionRequest",
    "MarketDataSubscriptionCancellation",

    "MarketOpeningHourRequest",
    "WhenMktOpen",
    "WhenMktClose",

    "Query",
    "QueryLastTrade",
    "QuerySpread",
    "QueryOrderStream",
    "QueryTransactedVolume"
)


class Request(Message):
    __slots__ = ()


# ___________________ ORDER REQUESTS ___________________

class OrderRequest(Request):
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


# ___________________ MARKET DATA SUBSCRIPTION REQUESTS ___________________

class MarketDataSubscription(Request):
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


# ___________________ EXCHANGE OPENING HOUR REQUESTS ___________________

class MarketOpeningHourRequest(Request):
    __slots__ = ()


class WhenMktOpen(MarketOpeningHourRequest):
    type = "WHEN_MKT_OPEN"
    __slots__ = ()


class WhenMktClose(MarketOpeningHourRequest):
    type = "WHEN_MKT_CLOSE"
    __slots__ = ()


# ___________________ QUERIES ___________________

class Query(Request):
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

    def __init__(self, sender_id: int, symbol: str, lookback_period: Union[str, pd.Timedelta]) -> None:
        super().__init__(sender_id, symbol)
        self.lookback_period = lookback_period
