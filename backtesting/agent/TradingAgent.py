import sys
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union, Sequence, Final, Any, Literal, overload, final

import numpy as np
import pandas as pd

from backtesting.agent.FinancialAgent import FinancialAgent
from backtesting.exchange import ExchangeAgent
from backtesting.message.base import Message
from backtesting.message.request import (

    LimitOrderRequest,
    MarketOrderRequest,
    CancelOrderRequest,
    ModifyOrderRequest,
    MarketDataSubscriptionRequest,
    MarketDataSubscriptionCancellation,

    WhenMktOpen,
    WhenMktClose,

    QueryLastTrade,
    QuerySpread,
    QueryOrderStream,
    QueryTransactedVolume
)
from backtesting.message.reply import OrderReply, OrderAccepted, OrderCancelled, OrderExecuted, MarketClosedReply, \
    MarketOpeningHourReply, WhenMktOpenReply, WhenMktCloseReply, QueryReplyMessage, QueryLastTradeReply, \
    QueryLastSpreadReply, QueryOrderStreamReply, QueryTransactedVolumeReply
from backtesting.message.notification import MarketData
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
        """
        The TradingAgent class (via FinancialAgent, via Agent) is intended as the
        base class for all trading agents (i.e. not things like exchanges) in a
        market simulation. It handles a lot of messaging (inbound and outbound)
        and state maintenance automatically, so subclasses can focus just on
        implementing a strategy without too much bookkeeping.

        Args:
            agent_id:       agent ID
            name:           agent name
            random_state:   random state
            log_to_file:    whether to log to file
            starting_cash:  starting cash
            log_orders:     whether to log orders
        """
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

        # When a last trade price comes in after market close, the TradingAgent
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
        # self.book = ''

        self.ready_to_trade = False  # indicating whether the agent is "ready to trade"
        self.exchange_id: int = None  # type: ignore

    def kernelStarting(self, start_time: pd.Timestamp) -> None:
        agent_id = self.id
        if self.kernel is None:
            raise RuntimeError(f"Kernel is not set for Agent {self.type} with ID {agent_id}")

        # Find an exchange with which we can place orders. It is guaranteed
        # to exist by now (if there is one).
        exchange_id = self.kernel.findAgentByType(ExchangeAgent)
        if exchange_id is None:
            raise RuntimeError("Kernel doesn't communicate with the Exchange")
        self.exchange_id = exchange_id

        self.logEvent('STARTING_CASH', self.starting_cash, True)
        log_print(f"Agent {agent_id} requested agent of type Agent.ExchangeAgent. Given Agent ID: {exchange_id}")

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

    @final
    def requestDataSubscription(self, symbol: str, *, levels: int, freq: int) -> None:
        """
        Used by any TradingAgent subclass to request subscription to market data from the ExchangeAgent.

        Args:
            symbol:  trading symbol
            levels:  number of levels
            freq:    subscription frequency in nanoseconds

        Returns:
            None
        """
        self.sendMessage(
            self.exchange_id,
            MarketDataSubscriptionRequest(
                self.id,
                symbol,
                levels=levels,
                freq=freq
            )
        )

    @final
    def cancelDataSubscription(self, symbol: str) -> None:
        """
        Used by any TradingAgent subclass to cancel subscription to market data from the ExchangeAgent.

        Args:
            symbol:  trading symbol

        Returns:
            None
        """
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
                self.mkt_open = msg.timestamp
                log_print(f"Recorded market open: {self.kernel.fmtTime(self.mkt_open)}")
            elif isinstance(msg, WhenMktCloseReply):
                self.mkt_close = msg.timestamp
                log_print(f"Recorded market close: {self.kernel.fmtTime(self.mkt_close)}")
            else:
                print(f"WARNING: {self.name} received MarketOpeningHourReply of type {msg.type}, but not handled")

        elif isinstance(msg, OrderReply):
            if isinstance(msg, OrderExecuted):
                # Call the processOrderExecuted method, which subclasses should extend.  This parent
                # class could implement default "portfolio tracking" or "returns tracking"
                # behavior.
                self.processOrderExecuted(msg.order)
            elif isinstance(msg, OrderAccepted):
                self.processOrderAccepted(msg.order)
            elif isinstance(msg, OrderCancelled):
                self.processOrderCancelled(msg.order)
            else:
                print(f"WARNING: {self.name} received OrderReply of type {msg.type}, but not handled")

        elif isinstance(msg, MarketClosedReply):
            # We've tried to ask the exchange for something after it closed. Remember this
            # so we stop asking for things that can't happen.
            self.processMarketClosed()

        elif isinstance(msg, QueryReplyMessage):
            self.mkt_closed = msg.mkt_closed
            if isinstance(msg, QueryLastTradeReply):
                # Call the processQueryLastTrade method, which subclasses may extend.
                # Also note if the market is closed.
                self.processQueryLastTrade(msg.symbol, msg.price)
            elif isinstance(msg, QueryLastSpreadReply):
                # Call the processQuerySpreadReply method, which subclasses may extend.
                # Also note if the market is closed.
                self.processQuerySpreadReply(msg.symbol, msg.last_spread, msg.bids, msg.asks)
            elif isinstance(msg, QueryOrderStreamReply):
                self.processQueryOrderStreamReply(msg.symbol, msg.orders)
            elif isinstance(msg, QueryTransactedVolumeReply):
                self.processQueryTransactedVolumeReply(msg.symbol, msg.transacted_volume)
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
    def getWakeFrequency(self) -> Union[pd.Timedelta, pd.DateOffset]:
        """
        Get frequency of agent's wakeups.

        Returns:
            wake up frequency
        """

    @final
    def getLastTrade(self, symbol: str) -> None:
        """
        Used by any TradingAgent subclass to query the last trade price for a symbol.

        Args:
            symbol:  trading symbol

        Returns:
            None
        """
        self.sendMessage(self.exchange_id, QueryLastTrade(self.id, symbol))

    @final
    def getCurrentSpread(self, symbol: str, depth: int = 1) -> None:
        """
        Used by any TradingAgent subclass to query the current spread for a symbol.

        Args:
            symbol:  trading symbol
            depth:   spread depth

        Returns:
            None
        """
        self.sendMessage(self.exchange_id, QuerySpread(self.id, symbol, depth))

    @final
    def getOrderStream(self, symbol: str, length: int = 1) -> None:
        """
        Used by any TradingAgent subclass to query the recent order stream for a symbol.

        Args:
            symbol:  trading symbol
            length:  number of recent orders

        Returns:
            None
        """
        self.sendMessage(self.exchange_id, QueryOrderStream(self.id, symbol, length))

    @final
    def getTransactedVolume(self, symbol: str, lookback_period: Union[str, pd.Timedelta] = '10min') -> None:
        """
        Used by any TradingAgent subclass to query the total transacted volume in a given lookback period.

        Args:
            symbol:           trading symbol
            lookback_period:  lookback period

        Returns:
            None
        """
        self.sendMessage(self.exchange_id, QueryTransactedVolume(self.id, symbol, lookback_period))

    @final
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
        Used by any TradingAgent subclass to place a limit order.

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

    @final
    def placeMarketOrder(self,
                         symbol: str,
                         quantity: int,
                         is_buy_order: bool,
                         order_id: Optional[int] = None,
                         ignore_risk: bool = True,
                         tag: Any = None) -> None:
        """
        Used by any TradingAgent subclass to place a market order. The market order is created as multiple limit orders
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

    @final
    def cancelOrder(self, order: LimitOrder) -> None:
        """
        Used by any TradingAgent subclass to cancel any order. The order must currently
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

    @final
    def modifyOrder(self, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any TradingAgent subclass to modify any existing limit order. The order must currently
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

    def processOrderExecuted(self, order: Order) -> None:
        """
        Handle OrderExecuted messages from the ExchangeAgent.

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

        if not holdings[symbol]:
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

    def processOrderAccepted(self, order: Order) -> None:
        """
        Handle OrderAccepted messages from the ExchangeAgent.

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

    def processOrderCancelled(self, order: Order) -> None:
        """
        Handle OrderCancelled messages from the ExchangeAgent.

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
        order_id = order.order_id
        orders = self.orders
        if order_id in orders:
            del orders[order_id]
        else:
            log_print(f"Cancellation received for order not in orders list: {order}")

    def processMarketClosed(self) -> None:
        """
        Handles MarketClosedReply messages from the ExchangeAgent.

        Returns:
            None
        """
        log_print("Received notification of market closure")
        self.logEvent('MKT_CLOSED')
        self.mkt_closed = True

    @final
    def processQueryLastTrade(self, symbol: str, price: int) -> None:
        """
        Handle QueryLastTradeReply message from the ExchangeAgent.

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

    @final
    def processQuerySpreadReply(self,
                                symbol: str,
                                price: int,
                                bids: List[Tuple[int, int]],
                                asks: List[Tuple[int, int]]) -> None:
        """
        Handle QueryLastSpreadReply messages from the ExchangeAgent.

        Args:
            symbol:  trading symbol
            price:   price of the
            bids:    list of bids
            asks:    list of asks

        Returns:
            None
        """
        # The spread message now also includes last price for free.
        self.processQueryLastTrade(symbol, price)

        self.known_bids[symbol] = bids
        if bids:
            best_bid, best_bid_qty = bids[0]
        else:
            best_bid = best_bid_qty = 0

        self.known_asks[symbol] = asks
        if asks:
            best_ask, best_ask_qty = asks[0]
        else:
            best_ask = best_ask_qty = 0

        log_print(f"Received spread of {best_bid_qty} @ {best_bid} / {best_ask_qty} @ {best_ask} for {symbol}")

        self.logEvent("BID_DEPTH", bids)
        self.logEvent("ASK_DEPTH", asks)
        self.logEvent("IMBALANCE", (sum(x[1] for x in bids), sum(x[1] for x in asks)))

    @final
    def handleMarketData(self, msg: MarketData) -> None:
        """
        Handle MarketData message for agents using subscription mechanism.

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

    @final
    def processQueryOrderStreamReply(self, symbol: str, orders: Tuple[OrderBookHistoryStep, ...]) -> None:
        """
        Handle QueryOrderStreamReply messages from the ExchangeAgent.

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

    @final
    def processQueryTransactedVolumeReply(self, symbol: str, transacted_volume: int) -> None:
        """
        Handle QueryTransactedVolumeReply messages from the ExchangeAgent.

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

    @final
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

    @final
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

    @staticmethod
    @final
    def getBookLiquidity(book: Sequence[Tuple[int, int]], within: Union[int, float]) -> int:
        """
        Helper function for the above. Checks one side of the known order book.

        Args:
            book:    Order Book history
            within:

        Returns:
            Order Book liquidity
        """
        liq = 0
        best = book[0][0]
        threshold = round(best * within)
        for price, shares in book:
            # Is this price within "within" proportion of the best price?
            if abs(best - price) <= threshold:
                log_print(f"Within {within} of {best}: {price} with {shares} shares")
                liq += shares
        return liq

    @final
    def markToMarket(self, holdings: Dict[str, int], use_midpoint: bool = False) -> int:
        """
        Marks holdings to market (including cash).

        Args:
            holdings:      holdings dictionary
            use_midpoint:  whether to use midpoint

        Returns:
            cash
        """
        cash = holdings['CASH']
        cash += self.basket_size * self.nav_diff
        last_trades = self.last_trade

        if use_midpoint:
            for symbol, shares in holdings.items():
                if symbol == 'CASH':
                    continue
                last_trade = last_trades[symbol]
                bid, ask, midpoint = self.getKnownBidAskMidpoint(symbol)
                if bid and ask:
                    value = midpoint * shares
                else:
                    value = last_trade * shares
                cash += value
                self.logEvent('MARK_TO_MARKET', f"{shares} {symbol} @ {last_trade} == {value}")
        else:
            for symbol, shares in holdings.items():
                if symbol == 'CASH':
                    continue
                last_trade = last_trades[symbol]
                value = last_trade * shares
                cash += value
                self.logEvent('MARK_TO_MARKET', f"{shares} {symbol} @ {last_trade} == {value}")

        self.logEvent('MARKED_TO_MARKET', cash)
        return cash

    @final
    def getHoldings(self, symbol: str) -> int:
        """
        Get holdings for a given symbol. Return zero for any symbol not held.

        Args:
            symbol:  trading symbol

        Returns:
            number of holdings
        """
        return self.holdings.get(symbol, 0)

    @final
    def getKnownBidAskMidpoint(self, symbol: str) -> Tuple[int, int, int]:
        """
        Get the known best bid, ask, and bid/ask midpoint from cached data. No volume.

        Args:
            symbol:  trading symbol

        Returns:
            best bid, best ask, midpoint
        """
        bids = self.known_bids[symbol]
        asks = self.known_asks[symbol]

        bid = bids[0][0] if bids else 0
        ask = asks[0][0] if asks else 0
        midpoint = round((bid + ask) / 2)
        return bid, ask, midpoint

    @final
    def get_average_transaction_price(self) -> float:
        """
        Calculates the average price paid (weighted by the order size).

        Returns:
            average transaction price
        """
        exec_orders = self.executed_orders
        return round(sum(o.quantity * o.fill_price for o in exec_orders) / sum(o.quantity for o in exec_orders), 2)

    @staticmethod
    @final
    def fmtHoldings(holdings: Dict[str, int]) -> str:
        """
        Prints holdings. Standard dictionary->string representation is almost fine, but it is
        less confusing to see the CASH holdings in dollars and cents, instead of just integer
        cents. We could change to a Holdings object that knows to print CASH "special".

        Args:
            holdings:  holdings dictionary

        Returns:
            string representation of holdings

        Examples:
            >>> TradingAgent.fmtHoldings({'USD': 3, 'RUR': 323, 'CASH': 200, 'AAPL': 212})
            '{ AAPL: 212, RUR: 323, USD: 3, CASH: 200 }'
        """
        holdings_items = (
            *sorted(item for item in holdings.items() if item[0] != 'CASH'),
            ('CASH', holdings['CASH'])  # There must always be a CASH entry.
        )
        return f"{{ {', '.join(f'{k}: {v}' for k, v in holdings_items)} }}"
