import sys
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union, Sequence, Final, Any, Literal, overload

import numpy as np
import pandas as pd

from backtesting.agent.FinancialAgent import FinancialAgent
from backtesting.exchange import ExchangeAgent
from backtesting.message.base import Message
from backtesting.message.types import (

    MarketClosedReply,
    MarketData,

    LimitOrderRequest,
    MarketOrderRequest,
    CancelOrderRequest,
    ModifyOrderRequest,
    OrderReply,
    OrderAccepted,
    OrderCancelled,
    OrderExecuted,
    MarketDataSubscriptionRequest,
    MarketDataSubscriptionCancellation,

    WhenMktOpen,
    WhenMktClose,

    MarketOpeningHourReply,
    WhenMktOpenReply,
    WhenMktCloseReply,

    QueryLastTrade,
    QuerySpread,
    QueryOrderStream,
    QueryTransactedVolume,

    QueryReplyMessage,
    QueryLastTradeReply,
    QueryLastSpreadReply,
    QueryOrderStreamReply,
    QueryTransactedVolumeReply
)
from backtesting.order.base import Order, LimitOrder
from backtesting.order.types import Ask, Bid, BuyMarket, SellMarket
from backtesting.typing.exchange import OrderBookHistoryStep
from backtesting.utils.util import log_print


class TradingAgent(FinancialAgent, metaclass=ABCMeta):
    """
    The TradingAgent class (via FinancialAgent, via Agent) is intended as the
    base class for all trading agents (i.e. not things like exchanges) in a
    market simulation. It handles a lot of messaging (inbound and outbound)
    and state maintenance automatically, so subclasses can focus just on
    implementing a strategy without too much bookkeeping.
    """
    __slots__ = (
        "mkt_open",
        "mkt_close",
        "log_orders",
        "starting_cash",
        "MKT_BUY",
        "MKT_SELL",
        "holdings",
        "orders",
        "last_trade",
        "exchange_ts",
        "daily_close_price",
        "nav_diff",
        "basket_size",
        "known_bids",
        "known_asks",
        "stream_history",
        "transacted_volume",
        "executed_orders",
        "first_wake",
        "mkt_closed",
        "book",
        "ready_to_trade",
        "exchange_id"
    )

    def __init__(self,
                 *,
                 agent_id: int,
                 name: str,
                 random_state: Optional[np.random.RandomState] = None,
                 log_to_file: bool = True,
                 starting_cash: int = 100_000,
                 log_orders: bool = False) -> None:

        super().__init__(agent_id, name, random_state, log_to_file)

        # We don't yet know when the exchange opens or closes.
        self.mkt_open: pd.Timestamp = None  # type: ignore
        self.mkt_close: pd.Timestamp = None  # type: ignore

        # Log order activity?
        self.log_orders = log_orders

        # Log all activity to file?
        self.log_to_file = log_to_file

        # Store starting_cash in case we want to refer to it for performance stats.
        # It should NOT be modified. Use the 'CASH' key in self.holdings.
        # 'CASH' is always in cents! Note that agents are limited by their starting
        # cash, currently without leverage. Taking short positions is permitted,
        # but does NOT increase the amount of at-risk capital allowed.
        self.starting_cash: Final = starting_cash

        # TradingAgent has constants to support simulated market orders.
        self.MKT_BUY = sys.maxsize
        self.MKT_SELL = 0

        # The base TradingAgent will track its holdings and outstanding orders.
        # Holdings is a dictionary of symbol -> shares. CASH is a special symbol
        # worth one cent per share. Orders is a dictionary of active, open orders
        # (not cancelled, not fully executed) keyed by order_id.
        self.holdings: Dict[str, int] = {'CASH': starting_cash}
        self.orders: Dict[int, Order] = {}

        # The base TradingAgent also tracks last known prices for every symbol
        # for which it has received as QUERY_LAST_TRADE message. Subclass
        # agents may use or ignore this as they wish. Note that the subclass
        # agent must request pricing when it wants it. This agent does NOT
        # automatically generate such requests, though it has a helper function
        # that can be used to make it happen.
        self.last_trade: Dict[str, int] = {}

        # used in subscription mode to record the timestamp for which the data was current in the ExchangeAgent
        self.exchange_ts: Dict[str, pd.Timestamp] = {}

        # When a last trade price comes in after market close, the trading agent
        # automatically records it as the daily close price for a symbol.
        self.daily_close_price: Dict[str, int] = {}

        self.nav_diff = self.basket_size = 0

        # The agent remembers the last known bids and asks (with variable depth,
        # showing only aggregate volume at each price level) when it receives
        # a response to QUERY_SPREAD.
        self.known_bids: Dict[str, List[Tuple[int, int]]] = {}
        self.known_asks: Dict[str, List[Tuple[int, int]]] = {}

        # The agent remembers the order history communicated by the exchange
        # when such is requested by an agent (for example, a heuristic belief
        # learning agent).
        self.stream_history: Dict = {}

        # The agent records the total transacted volume in the exchange for a given symbol and lookback period
        self.transacted_volume: Dict[str, int] = {}

        # Each agent can choose to log the orders executed
        self.executed_orders: List[LimitOrder] = []

        # For special logging at the first moment the simulator kernel begins
        # running (which is well after agent init), it is useful to keep a simple
        # boolean flag.
        self.first_wake = True

        # Remember whether we have already passed the exchange close time, as far
        # as we know.
        self.mkt_closed = False

        # This is probably a transient feature, but for now we permit the exchange
        # to return the entire order book sometimes, for development and debugging.
        # It is very expensive to pass this around, and breaks "simulation physics",
        # but can really help understand why agents are making certain decisions.
        # Subclasses should NOT rely on this feature as part of their strategy,
        # as it will go away.
        self.book = ''

        self.ready_to_trade = False  # indicating whether the agent is "ready to trade"
        self.exchange_id: int = None  # type: ignore

    # Simulation lifecycle messages.

    def kernelStarting(self, start_time: pd.Timestamp) -> None:
        # self.kernel is set in Agent.kernelInitializing()
        self.logEvent('STARTING_CASH', self.starting_cash, True)

        # Find an exchange with which we can place orders. It is guaranteed
        # to exist by now (if there is one).
        exchange_id = self.kernel.findAgentByType(ExchangeAgent)
        if exchange_id is None:
            raise RuntimeError("Kernel doesn't communicate with the Exchange")
        self.exchange_id = exchange_id

        log_print(f"Agent {self.id} requested agent of type Agent.ExchangeAgent. Given Agent ID: {self.exchange_id}")

        # Request a wake-up call as in the base Agent.
        super().kernelStarting(start_time)

    def kernelStopping(self) -> None:
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day holdings.
        holdings = self.holdings
        self.logEvent('FINAL_HOLDINGS', self.fmtHoldings(holdings))
        self.logEvent('FINAL_CASH_POSITION', holdings['CASH'], True)

        # Mark to market.
        cash = self.markToMarket(holdings)

        self.logEvent('ENDING_CASH', cash, True)
        print(f"Final holdings for {self.name}: {self.fmtHoldings(holdings)}. Marked to market: {cash}")

        # Record final results for presentation/debugging.  This is an ugly way
        # to do this, but it is useful for now.
        my_type = self.type
        gain = cash - self.starting_cash

        kernel = self.kernel
        mean_result_by_agent_type = kernel.mean_result_by_agent_type
        agent_count_by_type = kernel.agent_count_by_type
        if my_type in mean_result_by_agent_type:
            mean_result_by_agent_type[my_type] += gain
            agent_count_by_type[my_type] += 1
        else:
            mean_result_by_agent_type[my_type] = gain
            agent_count_by_type[my_type] = 1

    # Simulation participation messages.

    def wakeup(self, current_time: pd.Timestamp) -> None:
        super().wakeup(current_time)

        if self.first_wake:
            # Log initial holdings.
            self.logEvent('HOLDINGS_UPDATED', self.holdings)
            self.first_wake = False

        mkt_open_not_set = self.mkt_open is None
        if mkt_open_not_set:
            # Ask our exchange when it opens and closes.
            exchange_id = self.exchange_id
            self_id = self.id
            self.sendMessage(exchange_id, WhenMktOpen(self_id))
            self.sendMessage(exchange_id, WhenMktClose(self_id))

        # indicating whether the agent is "ready to trade" -- has it received
        # the market open and closed times, and is the market not already closed.
        self.ready_to_trade = not mkt_open_not_set and self.mkt_close is not None and not self.mkt_closed

    def requestDataSubscription(self, symbol: str, *, levels: int, freq: int) -> None:
        self.sendMessage(
            self.exchange_id,
            MarketDataSubscriptionRequest(
                self.id,
                symbol,
                levels=levels,
                freq=freq
            )
        )

    # Used by any Trading Agent subclass to cancel subscription to market data from the Exchange Agent
    def cancelDataSubscription(self, symbol: str) -> None:
        self.sendMessage(
            self.exchange_id,
            MarketDataSubscriptionCancellation(self.id, symbol)
        )

    def receiveMessage(self, current_time: pd.Timestamp, msg: Message) -> None:
        super().receiveMessage(current_time, msg)

        # Do we know the market hours?
        not_had_mkt_hours = self.mkt_open is None or self.mkt_close is None

        # Record market open or close times.
        if isinstance(msg, MarketOpeningHourReply):
            if isinstance(msg, WhenMktOpenReply):
                self.mkt_open = msg.data
                log_print(f"Recorded market open: {self.kernel.fmtTime(self.mkt_open)}")
            elif isinstance(msg, WhenMktCloseReply):
                self.mkt_close = msg.data
                log_print(f"Recorded market close: {self.kernel.fmtTime(self.mkt_close)}")
            else:
                print(f"WARNING: {self.name} received MarketOpeningHourReply of type {msg.type}, but not handled")

        elif isinstance(msg, OrderReply):
            if isinstance(msg, OrderExecuted):
                # Call the orderExecuted method, which subclasses should extend.  This parent
                # class could implement default "portfolio tracking" or "returns tracking"
                # behavior.
                self.orderExecuted(msg.order)
            elif isinstance(msg, OrderAccepted):
                self.orderAccepted(msg.order)
            elif isinstance(msg, OrderCancelled):
                self.orderCancelled(msg.order)
            else:
                print(f"WARNING: {self.name} received OrderReply of type {msg.type}, but not handled")

        elif isinstance(msg, MarketClosedReply):
            # We've tried to ask the exchange for something after it closed. Remember this
            # so we stop asking for things that can't happen.
            self.marketClosed()

        elif isinstance(msg, QueryReplyMessage):
            self.mkt_closed = msg.mkt_closed
            if isinstance(msg, QueryLastTradeReply):
                # Call the queryLastTrade method, which subclasses may extend.
                # Also note if the market is closed.
                self.queryLastTrade(msg.symbol, msg.data)
            elif isinstance(msg, QueryLastSpreadReply):
                # Call the querySpread method, which subclasses may extend.
                # Also note if the market is closed.
                self.querySpread(msg.symbol, msg.data, msg.bids, msg.asks, msg.book)
            elif isinstance(msg, QueryOrderStreamReply):
                self.queryOrderStream(msg.symbol, msg.orders)
            elif isinstance(msg, QueryTransactedVolumeReply):
                self.query_transacted_volume(msg.symbol, msg.transacted_volume)
            else:
                print(f"WARNING: {self.name} received QueryReplyMessage of type {msg.type}, but not handled")

        elif isinstance(msg, MarketData):
            self.handleMarketData(msg)
        else:
            print(f"WARNING: {self.name} received Message of type {msg.type}, but not handled")

        # Now do we know the market hours?
        have_mkt_hours = self.mkt_open is not None and self.mkt_close is not None

        # Once we know the market open and close times, schedule a wakeup call for market open.
        # Only do this once, when we first have both items.
        if have_mkt_hours and not_had_mkt_hours:
            # Agents are asked to generate a wake offset from the market open time. We structure
            # this as a subclass request so each agent can supply an appropriate offset relative
            # to its trading frequency.
            ns_offset = self.getWakeFrequency()
            self.setWakeup(self.mkt_open + ns_offset)

    @abstractmethod
    def getWakeFrequency(self) -> pd.Timedelta:
        pass

    # Used by any Trading Agent subclass to query the last trade price for a symbol.
    # This activity is not logged.
    def getLastTrade(self, symbol: str) -> None:
        self.sendMessage(self.exchange_id, QueryLastTrade(self.id, symbol))

    # Used by any Trading Agent subclass to query the current spread for a symbol.

    # This activity is not logged.
    def getCurrentSpread(self, symbol: str, depth: int = 1) -> None:
        self.sendMessage(self.exchange_id, QuerySpread(self.id, symbol, depth))

    # Used by any Trading Agent subclass to query the recent order stream for a symbol.
    def getOrderStream(self, symbol: str, length: int = 1) -> None:
        self.sendMessage(self.exchange_id, QueryOrderStream(self.id, symbol, length))

    def get_transacted_volume(self, symbol: str, lookback_period: Union[str, pd.Timedelta] = '10min') -> None:
        """
        Used by any trading agent subclass to query the total transacted volume in a given lookback period.

        Args:
            symbol:           trading symbol
            lookback_period:  lookback period

        Returns:
            None
        """
        self.sendMessage(self.exchange_id, QueryTransactedVolume(self.id, symbol, lookback_period))

    def placeLimitOrder(self,
                        symbol: str,
                        quantity: int,
                        is_buy_order: bool,
                        *,
                        limit_price: int,
                        order_id: Optional[int] = None,
                        ignore_risk: bool = True,
                        tag: Any = None) -> None:
        """
        Used by any Trading Agent subclass to place a limit order.

        Args:
            symbol:        trading symbol
            quantity:      positive share quantity
            is_buy_order:  whether is buy order
            limit_price:   price in cents
            order_id:      order ID
            ignore_risk:   whether cash or risk limits should be enforced or ignored for the order
            tag:           additional meta info

        Returns:
            None
        """

        order = (Bid if is_buy_order else Ask)(
            self.id,
            self.current_time,
            symbol,
            quantity=quantity,
            limit_price=limit_price,
            order_id=order_id,
            tag=tag
        )
        if order_id is None:
            order_id = order.order_id

        # # DEBUG to see event from Momentum agent
        # stack_list = [f for f in traceback.format_stack()]
        # if 'Momentum' in stack_list[len(stack_list) - 2]:
        #     DEBUG = True

        if quantity > 0:
            # Test if this order can be permitted given our at-risk limits.
            holdings = self.holdings
            new_holdings = holdings.copy()

            if not is_buy_order:
                quantity = -quantity

            if symbol in new_holdings:
                new_holdings[symbol] += quantity
            else:
                new_holdings[symbol] = quantity

            # If at_risk is lower, always allow. Otherwise, new_at_risk must be below starting cash.
            if not ignore_risk:
                # Compute before and after at-risk capital.
                at_risk = self.markToMarket(holdings) - holdings['CASH']
                new_at_risk = self.markToMarket(new_holdings) - new_holdings['CASH']

                if at_risk < new_at_risk > self.starting_cash:  # TODO. Possibly wrong logic !!!
                    log_print(
                        "TradingAgent ignored limit order "
                        f"due to at-risk constraints: {order}\n{self.fmtHoldings(holdings)}"
                    )
                    return

            # Copy the intended order for logging, so any changes made to it elsewhere
            # don't retroactively alter our "as placed" log of the order.  Eventually
            # it might be nice to make the whole history of the order into transaction
            # objects inside the order (we're halfway there) so there CAN be just a single
            # object per order, that never alters its original state, and eliminate all these copies.
            self.orders[order_id] = deepcopy(order)
            self.sendMessage(self.exchange_id, LimitOrderRequest(self.id, order))

            # Log this activity.
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', order.to_dict())
        else:
            log_print(f"TradingAgent ignored limit order of quantity zero: {order}")

    def placeMarketOrder(self,
                         symbol: str,
                         quantity: int,
                         is_buy_order: bool,
                         order_id: Optional[int] = None,
                         ignore_risk: bool = True,
                         tag: Any = None) -> None:
        """
        Used by any Trading Agent subclass to place a market order. The market order is created as multiple limit orders
        crossing the spread walking the book until all the quantities are matched.

        Args:
            symbol:        name of the stock traded
            quantity:      order quantity
            is_buy_order:  True if Buy else False
            order_id:      Order ID for market replay
            ignore_risk:   Determines whether cash or risk limits should be enforced or ignored for the order
            tag:           Additional meta information

        Returns:
            None
        """
        order = (BuyMarket if is_buy_order else SellMarket)(
            self.id,
            self.current_time,
            symbol,
            quantity,
            order_id
        )
        if order_id is None:
            order_id = order.order_id

        if quantity > 0:
            # compute new holdings
            holdings = self.holdings
            new_holdings = holdings.copy()

            if not is_buy_order:
                quantity = -quantity

            if symbol in new_holdings:
                new_holdings[symbol] += quantity
            else:
                new_holdings[symbol] = quantity

            if not ignore_risk:
                # Compute before and after at-risk capital.
                at_risk = self.markToMarket(holdings) - holdings['CASH']
                new_at_risk = self.markToMarket(new_holdings) - new_holdings['CASH']

                if at_risk < new_at_risk > self.starting_cash:  # TODO. Possibly wrong logic !!!
                    log_print(
                        "TradingAgent ignored market order "
                        f"due to at-risk constraints: {order}\n{self.fmtHoldings(holdings)}"
                    )
                    return
            self.orders[order_id] = deepcopy(order)

            self.sendMessage(self.exchange_id, MarketOrderRequest(self.id, order))
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', order.to_dict())
        else:
            log_print(f"TradingAgent ignored market order of quantity zero: {order}")

    def cancelOrder(self, order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to cancel any order. The order must currently
        appear in the agent's open orders list.

        Args:
            order:  Limit order of interest

        Returns:
            None
        """
        if isinstance(order, LimitOrder):
            self.sendMessage(self.exchange_id, CancelOrderRequest(self.id, order))
            # Log this activity.
            if self.log_orders:
                self.logEvent('CANCEL_SUBMITTED', order.to_dict())
        else:
            log_print(f"Order {order} of type {type(order)} cannot be cancelled")

    def modifyOrder(self, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to modify any existing limit order. The order must currently
        appear in the agent's open orders list. Some additional tests might be useful here
        to ensure the old and new orders are the same in some way.

        Args:
            order:      old Limit order
            new_order:  modified Limit order

        Returns:
            None
        """
        self.sendMessage(self.exchange_id, ModifyOrderRequest(self.id, order, new_order))
        if self.log_orders:
            self.logEvent('MODIFY_ORDER', order.to_dict())

    def orderExecuted(self, order: Order) -> None:
        """
        Handles ORDER_EXECUTED messages from an exchange agent. Subclasses may wish to extend,
        but should still call parent method for basic portfolio/returns tracking.

        Args:
            order:  executed order

        Returns:
            None
        """
        log_print(f"Received notification of execution for: {order}")
        if self.log_orders:
            self.logEvent('ORDER_EXECUTED', order.to_dict())

        # At the very least, we must update CASH and holdings at execution time.
        symbol = order.symbol
        quantity = order.quantity
        if order.is_buy_order:
            quantity = -quantity

        holdings = self.holdings
        if symbol in holdings:
            holdings[symbol] += quantity
        else:
            holdings[symbol] = quantity

        if holdings[symbol] == 0:
            del holdings[symbol]

        # As with everything else, CASH holdings are in CENTS.
        holdings['CASH'] -= quantity * order.fill_price

        # If this original order is now fully executed, remove it from the open orders list.
        # Otherwise, decrement by the quantity filled just now.  It is _possible_ that due
        # to timing issues, it might not be in the order list (i.e. we issued a cancellation
        # but it was executed first, or something).
        orders = self.orders
        order_id = order.order_id
        if order_id in orders:
            order_found = orders[order_id]

            if order.quantity >= order_found.quantity:
                del orders[order_id]
            else:
                order_found.quantity -= order.quantity
        else:
            log_print(f"Execution received for order not in orders list: {order}")

        log_print(f"After execution, agent open orders: {orders}")
        self.logEvent('HOLDINGS_UPDATED', holdings)

    def orderAccepted(self, order: Order) -> None:
        """
        Handles ORDER_ACCEPTED messages from an exchange agent. Subclasses may wish to extend.

        Args:
            order:  accepted order

        Returns:
            None
        """
        log_print(f"Received notification of acceptance for: {order}")

        # Log this activity.
        if self.log_orders:
            self.logEvent('ORDER_ACCEPTED', order.to_dict())

        # We may later wish to add a status to the open orders so an agent can tell whether
        # a given order has been accepted or not (instead of needing to override this method).

    def orderCancelled(self, order: Order) -> None:
        """
        Handles ORDER_CANCELLED messages from an exchange agent. Subclasses may wish to extend.

        Args:
            order:  cancelled order

        Returns:
            None
        """
        log_print(f"Received notification of cancellation for: {order}")
        if self.log_orders:
            self.logEvent('ORDER_CANCELLED', order.to_dict())

        # Remove the cancelled order from the open orders list.  We may of course wish to have
        # additional logic here later, so agents can easily "look for" cancelled orders.  Of
        # course they can just override this method.
        if (order_id := order.order_id) in (orders := self.orders):
            del orders[order_id]
        else:
            log_print(f"Cancellation received for order not in orders list: {order}")

    def marketClosed(self) -> None:
        """
        Handles MKT_CLOSED messages from an exchange agent. Subclasses may wish to extend.

        Returns:
            None
        """
        log_print("Received notification of market closure")
        self.logEvent('MKT_CLOSED')

        self.mkt_closed = True

    def queryLastTrade(self, symbol: str, price: int) -> None:
        """
        Handles QUERY_LAST_TRADE_REPLY messages from an exchange agent.

        Args:
            symbol:  trading symbol
            price:   price of the last trade

        Returns:
            None
        """
        last_trade = self.last_trade
        last_trade[symbol] = price

        log_print(f"Received last trade price of {price} for {symbol}")

        if self.mkt_closed:
            # Note this as the final price of the day.
            self.daily_close_price[symbol] = price
            log_print(f"Received daily close price of {price} for {symbol}")

    def querySpread(self,
                    symbol: str,
                    price: int,
                    bids: List[Tuple[int, int]],
                    asks: List[Tuple[int, int]],
                    book: Any) -> None:
        """
        Handles QUERY_LAST_SPREAD_REPLY messages from an exchange agent.

        Args:
            symbol:  trading symbol
            price:   price of the
            bids:    list of bids
            asks:    list of asks
            book:    additional info

        Returns:
            None
        """
        # The spread message now also includes last price for free.
        self.queryLastTrade(symbol, price)

        self.known_bids[symbol] = bids
        self.known_asks[symbol] = asks

        if bids:
            best_bid, best_bid_qty = bids[0]
        else:
            best_bid = best_bid_qty = 0

        if asks:
            best_ask, best_ask_qty = asks[0]
        else:
            best_ask = best_ask_qty = 0

        log_print(f"Received spread of {best_bid_qty} @ {best_bid} / {best_ask_qty} @ {best_ask} for {symbol}")

        self.logEvent("BID_DEPTH", bids)
        self.logEvent("ASK_DEPTH", asks)
        self.logEvent("IMBALANCE", (sum(x[1] for x in bids), sum(x[1] for x in asks)))

        self.book = book

    def handleMarketData(self, msg: MarketData) -> None:
        """
        Handle MARKET_DATA message for agents using subscription mechanism

        Args:
            msg:  MarketData message
        Returns:
            None
        """
        symbol = msg.symbol
        self.known_asks[symbol] = msg.asks
        self.known_bids[symbol] = msg.bids
        self.last_trade[symbol] = msg.last_transaction
        self.exchange_ts[symbol] = msg.exchange_ts

    def queryOrderStream(self, symbol: str, orders: Tuple[OrderBookHistoryStep, ...]) -> None:
        """
        Handles QUERY_ORDER_STREAM_REPLY messages from an exchange agent.

        Args:
            symbol:  trading symbol
            orders:  OrderBook history log

        Returns:
            None
        """
        # It is up to the requesting agent to do something with the data, which is a list of dictionaries keyed
        # by order id. The list index is 0 for orders since the most recent trade, 1 for orders that led up to
        # the most recent trade, and so on. Agents are not given index 0 (orders more recent than the last
        # trade).
        self.stream_history[symbol] = orders

    def query_transacted_volume(self, symbol: str, transacted_volume: int) -> None:
        """
        Handles the QUERY_TRANSACTED_VOLUME_REPLY messages from the exchange agent.

        Args:
            symbol:             trading symbol
            transacted_volume:  total transaction volume

        Returns:
            None
        """
        self.transacted_volume[symbol] = transacted_volume

    # >>> Utility functions that perform calculations from available knowledge,
    # >>> but implement no particular strategy.

    @overload
    def getKnownBidAsk(self,
                       symbol: str,
                       best: Literal[True] = True) -> Tuple[int, int, int, int]:
        pass

    @overload
    def getKnownBidAsk(self,
                       symbol: str,
                       best: Literal[False]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        pass

    def getKnownBidAsk(self,
                       symbol: str,
                       best: bool = True) -> Union[Tuple[int, int, int, int],
                                                   Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Extract the current known bid and asks. This does NOT request new information.

        Args:
            symbol:  trading symbol
            best:    whether to extract best bids/asks or full known LOB

        Returns:
            (bids, asks) [OR] (best bid price, best bid volume, best ask price, best ask volume)
        """
        bids = self.known_bids[symbol]
        asks = self.known_asks[symbol]
        if best:
            if bids:
                bid, bid_vol = bids[0]
            else:
                bid = bid_vol = 0

            if asks:
                ask, ask_vol = asks[0]
            else:
                ask = ask_vol = 0

            return bid, bid_vol, ask, ask_vol

        return bids, asks

    def getKnownLiquidity(self, symbol: str, within: Union[int, float] = 0) -> Tuple[int, int]:
        """
        Extract the current bid and ask liquidity within a certain proportion of the inside bid and ask.

        Args:
            symbol:  trading symbol
            within:  for example 0.01 means to report total BID share within 1% of the best bid price,
                     and total ASK shares within 1% of the best
                     ask price

        Returns:
            (bids liquidity, asks liquidity)

        Note that this is from the order book perspective, not the agent perspective.
        (The agent would be selling into the bid liquidity, etc.)
        """
        bids = self.known_bids[symbol]
        asks = self.known_asks[symbol]
        bid_liq = self.getBookLiquidity(bids, within)
        ask_liq = self.getBookLiquidity(asks, within)

        log_print(
            f"Bid/ask liq: {bid_liq}, {ask_liq}\n"
            f"Known bids: {bids}\n"
            f"Known asks: {asks}"
        )

        return bid_liq, ask_liq

    # Helper function for the above. Checks one side of the known order book.
    @staticmethod
    def getBookLiquidity(book: Sequence[Tuple[int, int]], within: Union[int, float]) -> int:
        liq = 0
        best = book[0][0]
        for price, shares in book:
            # Is this price within "within" proportion of the best price?
            if abs(best - price) <= round(best * within):
                log_print(f"Within {within} of {best}: {price} with {shares} shares")
                liq += shares
        return liq

    # Marks holdings to market (including cash).
    def markToMarket(self, holdings: Dict[str, int], use_midpoint: bool = False):
        cash = holdings['CASH']

        cash += self.basket_size * self.nav_diff

        for symbol, shares in holdings.items():
            if symbol == 'CASH':
                continue

            if use_midpoint:
                bid, ask, midpoint = self.getKnownBidAskMidpoint(symbol)
                if bid is None or ask is None or midpoint is None:
                    value = self.last_trade[symbol] * shares
                else:
                    value = midpoint * shares
            else:
                value = self.last_trade[symbol] * shares

            cash += value

            self.logEvent('MARK_TO_MARKET', f"{shares} {symbol} @ {self.last_trade[symbol]} == {value}")

        self.logEvent('MARKED_TO_MARKET', cash)

        return cash

    # Gets holdings. Returns zero for any symbol not held.
    def getHoldings(self, symbol):
        return self.holdings.get(symbol, 0)

    # Get the known best bid, ask, and bid/ask midpoint from cached data.  No volume.
    def getKnownBidAskMidpoint(self, symbol):
        bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
        ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None

        midpoint = int(round((bid + ask) / 2)) if bid is not None and ask is not None else None

        return bid, ask, midpoint

    def get_average_transaction_price(self):
        """ Calculates the average price paid (weighted by the order size) """
        return round(
            sum(executed_order.quantity * executed_order.fill_price for executed_order in self.executed_orders) / \
            sum(executed_order.quantity for executed_order in self.executed_orders), 2)

    # Prints holdings.  Standard dictionary->string representation is almost fine, but it is
    # less confusing to see the CASH holdings in dollars and cents, instead of just integer
    # cents.  We could change to a Holdings object that knows to print CASH "special".
    @staticmethod
    def fmtHoldings(holdings):
        h = ''
        for k, v in sorted(holdings.items()):
            if k == 'CASH': continue
            h += "{}: {}, ".format(k, v)

        # There must always be a CASH entry.
        h += "{}: {}".format('CASH', holdings['CASH'])
        h = '{ ' + h + ' }'
        return h
