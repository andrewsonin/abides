from abides.order.base import MarketOrder, LimitOrder

__all__ = (
    "BuyMarket",
    "SellMarket",
    "Bid",
    "Ask"
)


class BuyMarket(MarketOrder):
    __slots__ = ()
    is_buy_order = True


class SellMarket(MarketOrder):
    __slots__ = ()
    is_buy_order = False


class Bid(LimitOrder):
    __slots__ = ()
    is_buy_order = True


class Ask(LimitOrder):
    __slots__ = ()
    is_buy_order = False
