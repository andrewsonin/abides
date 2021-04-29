"""
The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp to open and close trading,
a list of equity symbols for which it should create order books, a frequency at which to archive snapshots
of its order books, a pipeline delay (in ns) for order activity, the exchange computation delay (in ns),
the levels of order stream history to maintain per symbol (maintains all orders that led to the last N trades),
whether to log all order activity to the agent log, and a random state object (already seeded) to use
for stochasticity.


This file also contain a basic class for an order book for one symbol, in the style of the major US Stock Exchanges.
List of bid prices (index zero is best bid), each with a list of LimitOrders.
List of ask prices (index zero is best ask), each with a list of LimitOrders.
"""
import datetime as dt
import sys
import warnings
from collections import deque
from copy import deepcopy
from itertools import islice, chain
from typing import Iterable, Dict, Tuple, Union, Optional, TypeVar, Generic, Deque, List, MutableSet, Literal, overload

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from tqdm import tqdm

from abides.agent.FinancialAgent import FinancialAgent
from abides.core import Kernel
from abides.message.base import MessageAbstractBase, Message
from abides.message.types import (
    MarketClosedReply,

    OrderRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    CancelOrderRequest,
    ModifyOrderRequest,

    OrderReply,
    MarketDataSubscription,
    MarketDataSubscriptionRequest,

    MarketOpeningHourRequest,
    WhenMktOpen,

    MarketOpeningHourReply,
    WhenMktOpenReply,
    WhenMktCloseReply,

    Query,
    QueryLastTrade,
    QuerySpread,
    QueryOrderStream,
    QueryTransactedVolume,

    QueryLastTradeReply,
    QueryLastSpreadReply,
    QueryOrderStreamReply,
    QueryTransactedVolumeReply,

    MarketData, OrderExecuted, OrderAccepted, OrderCancelled, OrderModified
)
from abides.oracle.types import DataOracle, ExternalFileOracle, SparseMeanRevertingOracle
from abides.order.types import Bid, Ask, LimitOrder, MarketOrder
from abides.typing import OrderBookHistoryStep
from abides.util import log_print, be_silent

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.set_option('display.max_rows', 500)

__all__ = (
    "ExchangeAgent",
    "OrderBook"
)

_OracleType = TypeVar('_OracleType')


class ExchangeAgent(FinancialAgent, Generic[_OracleType]):
    __slots__ = (
        "reschedule",
        "mkt_open",
        "mkt_close",
        "pipeline_delay",
        "computation_delay",
        "stream_history",
        "log_orders",
        "order_books",
        "book_freq",
        "wide_book",
        "subscription_dict",
        "oracle"
    )

    def __init__(self,
                 *,
                 agent_id: int,
                 name: str,
                 mkt_open: pd.Timestamp,
                 mkt_close: pd.Timestamp,
                 symbols: Iterable[str],
                 book_freq: Union[pd.DateOffset, str] = 'S',
                 wide_book: bool = False,
                 pipeline_delay: int = 40_000,
                 computation_delay: int = 1,
                 stream_history: int = 0,
                 log_orders: bool = False,
                 random_state: Optional[np.random.RandomState] = None,
                 oracle: _OracleType) -> None:

        super().__init__(agent_id, name, random_state)

        # Do not request repeated wakeup calls.
        self.reschedule = False

        # Store this exchange's open and close times.
        self.mkt_open = mkt_open
        self.mkt_close = mkt_close

        # Right now, only the exchange agent has a parallel processing pipeline delay.  This is an additional
        # delay added only to order activity (placing orders, etc) and not simple inquiries (market operating
        # hours, etc).
        self.pipeline_delay = pipeline_delay

        # Computation delay is applied on every wakeup call or message received.
        self.computation_delay = computation_delay

        # The exchange maintains an order stream of all orders leading to the last L trades
        # to support certain agents from the auction literature (GD, HBL, etc).
        if not isinstance(stream_history, int) or stream_history < 0:
            raise TypeError("'stream_history' must be positive integer")
        self.stream_history = stream_history

        # Log all order activity?
        self.log_orders = log_orders

        # Create an order book for each symbol.
        self.order_books = {
            symbol: OrderBook(self, symbol)
            for symbol in symbols
        }

        # At what frequency will we archive the order books for visualization and analysis?
        self.book_freq = book_freq

        # Store orderbook in wide format? ONLY WORKS with book_freq == 0
        self.wide_book = wide_book

        # The subscription dict is a dictionary with the key = agent ID,
        # value = dict (key = symbol, value = list [levels (no of levels to receive updates for),
        # frequency (min number of ns between messages), last agent update timestamp]
        # e.g. {101 : {'AAPL' : [1, 10, pd.Timestamp(10:00:00)}}
        self.subscription_dict: Dict[int, Dict[str, Tuple[int, int, pd.Timestamp]]] = {}

        self.oracle = oracle

        # The exchange agent overrides this to obtain a reference to an oracle.
        # This is needed to establish a "last trade price" at open (i.e. an opening
        # price) in case agents query last trade before any simulated trades are made.
        # This can probably go away once we code the opening cross auction.

    def kernelInitializing(self, kernel: Kernel) -> None:
        super().kernelInitializing(kernel)

        if self.oracle is not self.kernel.oracle:
            raise ValueError(f"{self.__class__.__name__} oracle and Kernel oracle should be the same object")

        # Obtain opening prices (in integer cents).  These are not noisy right now.
        oracle = self.oracle
        order_books = self.order_books
        if isinstance(oracle, DataOracle):
            for symbol in self.order_books:
                last_trade = oracle.getDailyOpenPrice(symbol, self.mkt_open)
                order_books[symbol].last_trade = last_trade
                log_print(f"Opening price for {symbol} is {last_trade}")

    # The exchange agent overrides this to additionally log the full depth of its
    # order books for the entire day.
    def kernelTerminating(self) -> None:
        super().kernelTerminating()

        # If the oracle supports writing the fundamental value series for its
        # symbols, write them to disk.
        oracle = self.oracle
        if isinstance(oracle, (ExternalFileOracle, SparseMeanRevertingOracle)):
            for symbol in oracle.f_log:
                dfFund = pd.DataFrame(oracle.f_log[symbol])
                if not dfFund.empty:
                    dfFund.set_index('FundamentalTime', inplace=True)
                    self.writeLog(dfFund, filename=f'fundamental_{symbol}')
                    log_print("Fundamental archival complete.")
        if self.book_freq is not None:
            # Iterate over the order books controlled by this exchange.
            for symbol in self.order_books:
                start_time = dt.datetime.now()
                self.logOrderBookSnapshots(symbol)
                end_time = dt.datetime.now()
                print(
                    f"Time taken to log the order book: {end_time - start_time}\n"
                    "Order book archival complete."
                )

    def receiveMessage(self, current_time: pd.Timestamp, msg: Message) -> None:
        super().receiveMessage(current_time, msg)

        # Unless the intent of an experiment is to examine computational issues within an Exchange,
        # it will typically have either 1 ns delay (near instant but cannot process multiple orders
        # in the same atomic time unit) or 0 ns delay (can process any number of orders, always in
        # the atomic time unit in which they are received).  This is separate from, and additional
        # to, any parallel pipeline delay imposed for order book activity.

        # Note that computation delay MUST be updated before any calls to sendMessage.
        self.setComputationDelay(self.computation_delay)

        sender_id = msg.sender_id
        msg_type = msg.type
        mkt_closed = current_time > self.mkt_close
        # Is the exchange closed?  (This block only affects post-close, not pre-open.)
        if isinstance(msg, OrderRequest):
            if mkt_closed:
                log_print(f"{self.name} received {msg_type}: {msg.order}")
                self.sendMessage(sender_id, MarketClosedReply(self.id))
                return
            if self.log_orders:
                self.logEvent(msg_type, msg.order.to_dict())

            if isinstance(msg, LimitOrderRequest):
                self.processLimitOrderRequest(msg)
            elif isinstance(msg, MarketOrderRequest):
                self.processMarketOrderRequest(msg)
            elif isinstance(msg, CancelOrderRequest):
                self.processCancelOrderRequest(msg)
            elif isinstance(msg, ModifyOrderRequest):
                self.processModifyOrderRequest(msg)
            else:
                log_print(f"WARNING: {self.name} received OrderRequest of type {msg_type}, but not handled")

        elif isinstance(msg, Query):
            if mkt_closed:
                log_print(f"{self.name} received {msg_type}, discarded: market is closed.")
                self.sendMessage(sender_id, MarketClosedReply(self.id))
                # Don't do any further processing on these messages!
                return
            self.logEvent(msg_type, sender_id)

            if isinstance(msg, QueryLastTrade):
                self.processQueryLastTrade(msg, mkt_closed)
            elif isinstance(msg, QuerySpread):
                self.processQuerySpread(msg, mkt_closed)
            elif isinstance(msg, QueryOrderStream):
                self.processQueryOrderStream(msg, mkt_closed)
            elif isinstance(msg, QueryTransactedVolume):
                self.processQueryTransactedVolume(msg, mkt_closed)
            else:
                log_print(f"WARNING: {self.name} received Query of type {msg_type}, but not handled")

        else:
            self.logEvent(msg_type, sender_id)
            if isinstance(msg, MarketDataSubscription):
                self.processMarketDataSubscriptionMessage(msg, current_time)
            elif isinstance(msg, MarketOpeningHourRequest):
                self.processMarketOpeningHourRequest(msg)
            else:
                log_print(f"WARNING: {self.name} received {msg_type}, but not handled")

    def updateSubscriptionDict(self, msg: MarketDataSubscription, current_time: pd.Timestamp) -> None:
        # The subscription dict is a dictionary with the key = agent ID,
        # value = dict (key = symbol, value = list [levels (no of levels to receive updates for),
        # frequency (min number of ns between messages), last agent update timestamp]
        # e.g. {101 : {'AAPL' : [1, 10, pd.Timestamp(10:00:00)}}
        agent_id = msg.sender_id
        symbol = msg.symbol
        if isinstance(msg, MarketDataSubscriptionRequest):
            self.subscription_dict[agent_id] = {symbol: (msg.levels, msg.freq, current_time)}
        else:  # MarketDataSubscriptionCancellation
            del self.subscription_dict[agent_id][symbol]

    def publishOrderBookData(self) -> None:
        """
        The exchange agents sends an order book update to the agents using the subscription API if one of the following
        conditions are met:
        1) agent requests ALL order book updates (freq == 0)
        2) order book update timestamp > last time agent was updated AND the orderbook update time stamp is greater than
        the last agent update time stamp by a period more than that specified in the freq parameter.
        """
        current_time = self.current_time
        exchange_id = self.id
        subscription_dict = self.subscription_dict
        order_books = self.order_books
        for agent_id, params in subscription_dict.items():
            for symbol, (levels, freq, last_agent_update) in params.items():
                order_book = order_books[symbol]
                ob_last_update = order_book.last_update_ts
                if freq == 0 or (ob_last_update is not None
                                 and ob_last_update > last_agent_update
                                 and (ob_last_update - last_agent_update).delta >= freq):
                    self.sendMessage(
                        agent_id,
                        MarketData(
                            exchange_id,
                            symbol,
                            order_book.last_trade,
                            current_time,
                            bids=order_book.getInsideBids(levels),
                            asks=order_book.getInsideAsks(levels)
                        )
                    )
                    agent_subscriptions = subscription_dict[agent_id]
                    levels, freq, _ = agent_subscriptions[symbol]
                    agent_subscriptions[symbol] = (levels, freq, ob_last_update)

    def logOrderBookSnapshots(self, symbol: str) -> None:
        """
        Log full depth quotes (price, volume) from this order book at some pre-determined frequency.
        Here we are looking at the actual log for this order book (i.e. are there snapshots to export,
        independent of the requested frequency).
        """

        def get_quote_range_iterator(s):
            """ Helper method for order book logging. Takes pandas Series and returns python range() from first to last
                element.
            """
            forbidden_values = [0, 19999900]  # TODO: Put constant value in more sensible place!
            quotes = sorted(s)
            for val in forbidden_values:
                try:
                    quotes.remove(val)
                except ValueError:
                    pass
            return quotes

        book = self.order_books[symbol]

        if book.book_log:

            print("Logging order book to file...")
            dfLog = book.book_log_to_df()
            dfLog.set_index('QuoteTime', inplace=True)
            dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
            dfLog.sort_index(inplace=True)

            if str(self.book_freq).isdigit() and int(self.book_freq) == 0:  # Save all possible information
                # Get the full range of quotes at the finest possible resolution.
                quotes = get_quote_range_iterator(dfLog.columns.unique())

                # Restructure the log to have multi-level rows of all possible pairs of time and quote
                # with volume as the only column.
                if not self.wide_book:
                    filledIndex = pd.MultiIndex.from_product([dfLog.index, quotes], names=['time', 'quote'])
                    dfLog = dfLog.stack()
                    dfLog = dfLog.reindex(filledIndex)

                filename = f'ORDERBOOK_{symbol}_FULL'

            else:
                # Sample at frequency self.book_freq
                # With multiple quotes in a nanosecond, use the last one, then resample to the requested freq.
                dfLog = dfLog.resample(self.book_freq)
                dfLog.ffill(inplace=True)
                dfLog.sort_index(inplace=True)

                # Create a fully populated index at the desired frequency from market open to close.
                # Then project the logged data into this complete index.
                time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
                dfLog = dfLog.reindex(time_idx, method='ffill')
                dfLog.sort_index(inplace=True)

                if not self.wide_book:
                    dfLog = dfLog.stack()
                    dfLog.sort_index(inplace=True)

                    # Get the full range of quotes at the finest possible resolution.
                    quotes = get_quote_range_iterator(dfLog.index.get_level_values(1).unique())

                    # Restructure the log to have multi-level rows of all possible pairs of time and quote
                    # with volume as the only column.
                    filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time', 'quote'])
                    dfLog = dfLog.reindex(filledIndex)

                filename = f'ORDERBOOK_{symbol}_FREQ_{self.book_freq}'

            # Final cleanup
            if not self.wide_book:
                dfLog.rename('Volume')
                df = pd.SparseDataFrame(index=dfLog.index)
                df['Volume'] = dfLog
            else:
                df = dfLog
                df = df.reindex(sorted(df.columns), axis=1)

            # Archive the order book snapshots directly to a file named with the symbol, rather than
            # to the exchange agent log.
            self.writeLog(df, filename=filename)
            print("Order book logging complete!")

    def sendMessage(self, recipient_id: int, msg: MessageAbstractBase, delay: int = 0) -> None:
        # The ExchangeAgent automatically applies appropriate parallel processing pipeline delay
        # to those message types which require it.
        # TODO: probably organize the order types into categories once there are more, so we can
        # take action by category (e.g. ORDER-related messages) instead of enumerating all message
        # types to be affected.
        if isinstance(msg, OrderReply):
            # Messages that require order book modification (not simple queries) incur the additional
            # parallel processing delay as configured.
            super().sendMessage(recipient_id, msg, self.pipeline_delay)
            if self.log_orders:
                self.logEvent(msg.type, msg.order.to_dict())
        else:
            # Other message types incur only the currently-configured computation delay for this agent.
            super().sendMessage(recipient_id, msg, delay)

    # Simple accessor methods for the market open and close times.
    def getMarketOpen(self) -> pd.Timestamp:
        return self.mkt_open

    def getMarketClose(self) -> pd.Timestamp:
        return self.mkt_close

    def processLimitOrderRequest(self, msg: LimitOrderRequest) -> None:
        order = msg.order
        symbol = order.symbol
        log_print(f"{self.name} received LIMIT_ORDER: {order}")
        if symbol not in self.order_books:
            log_print(f"Limit Order discarded. Unknown symbol: {symbol}")
        else:
            # Hand the order to the order book for processing.
            self.order_books[symbol].handleLimitOrder(deepcopy(order))
            self.publishOrderBookData()

    def processMarketOrderRequest(self, msg: MarketOrderRequest) -> None:
        order = msg.order
        symbol = order.symbol
        log_print(f"{self.name} received MARKET_ORDER: {order}")
        if symbol not in self.order_books:
            log_print(f"Market Order discarded. Unknown symbol: {symbol}")
        else:
            # Hand the market order to the order book for processing.
            self.order_books[symbol].handleMarketOrder(deepcopy(order))
            self.publishOrderBookData()

    def processCancelOrderRequest(self, msg: CancelOrderRequest) -> None:
        # Note: this is somewhat open to abuse, as in theory agents could cancel other agents' orders.
        # An agent could also become confused if they receive a (partial) execution on an order they
        # then successfully cancel, but receive the cancel confirmation first. Things to think about
        # for later...
        order = msg.order
        symbol = order.symbol
        log_print(f"{self.name} received CANCEL_ORDER: {order}")
        if order.symbol not in self.order_books:
            log_print(f"Cancellation request discarded. Unknown symbol: {symbol}")
        else:
            # Hand the order to the order book for processing.
            self.order_books[symbol].cancelLimitOrder(deepcopy(order))
            self.publishOrderBookData()

    def processModifyOrderRequest(self, msg: ModifyOrderRequest) -> None:
        # Replace an existing order with a modified order. There could be some timing issues
        # here. What if an order is partially executed, but the submitting agent has not
        # yet received the notification, and submits a modification to the quantity of the
        # (already partially executed) order? I guess it is okay if we just think of this
        # as "delete and then add new" and make it the agent's problem if anything weird
        # happens.
        order = msg.order
        symbol = order.symbol
        new_order = msg.new_order
        log_print(f"{self.name} received MODIFY_ORDER: {order}, new order: {new_order}")
        if order.symbol not in self.order_books:
            log_print(f"Modification request discarded. Unknown symbol: {symbol}")
        else:
            self.order_books[symbol].modifyLimitOrder(deepcopy(order), deepcopy(new_order))
            self.publishOrderBookData()

    def processQueryLastTrade(self, msg: QueryLastTrade, mkt_closed: bool) -> None:
        symbol = msg.symbol
        sender_id = msg.sender_id
        if symbol not in self.order_books:
            log_print(f"Last trade request discarded. Unknown symbol: {symbol}")
        else:
            log_print(f"{self.name} received QUERY_LAST_TRADE ({symbol}) request from agent {sender_id}")

            # Return the single last executed trade price (currently not volume) for the requested symbol.
            # This will return the average share price if multiple executions resulted from a single order.
            self.sendMessage(
                sender_id,
                QueryLastTradeReply(
                    self.id,
                    symbol,
                    mkt_closed,
                    self.order_books[symbol].last_trade
                )
            )

    def processQuerySpread(self, msg: QuerySpread, mkt_closed: bool) -> None:
        symbol = msg.symbol
        depth = msg.depth
        sender_id = msg.sender_id
        if symbol not in self.order_books:
            log_print(f"Bid-ask spread request discarded. Unknown symbol: {symbol}")
        else:
            log_print(f"{self.name} received {self.type} ({symbol}:{depth}) request from agent {sender_id}")

            # Return the requested depth on both sides of the order book for the requested symbol.
            # Returns price levels and aggregated volume at each level (not individual orders).
            order_book = self.order_books[symbol]
            self.sendMessage(
                sender_id,
                QueryLastSpreadReply(
                    self.id,
                    symbol,
                    mkt_closed,
                    depth,
                    bids=order_book.getInsideBids(depth),
                    asks=order_book.getInsideAsks(depth),
                    data=order_book.last_trade,
                    book=''
                )
            )

    def processQueryOrderStream(self, msg: QueryOrderStream, mkt_closed: bool) -> None:
        symbol = msg.symbol
        length = msg.length
        sender_id = msg.sender_id
        if symbol not in self.order_books:
            log_print(f"Order stream request discarded. Unknown symbol: {symbol}")
        else:
            log_print(f"{self.name} received {self.type} ({symbol}:{length}) request from agent {sender_id}")

        # We return indices [1:length] inclusive because the agent will want "orders leading up to the last
        # L trades", and the items under index 0 are more recent than the last trade.
        self.sendMessage(
            sender_id,
            QueryOrderStreamReply(
                self.id,
                symbol,
                mkt_closed,
                length,
                self.order_books[symbol].history[1:(length + 1)]
            )
        )

    def processQueryTransactedVolume(self, msg: QueryTransactedVolume, mkt_closed: bool) -> None:
        symbol = msg.symbol
        lookback_period = msg.lookback_period
        sender_id = msg.sender_id
        if symbol not in self.order_books:
            log_print(f"Order stream request discarded. Unknown symbol: {symbol}")
        else:
            log_print(
                f"{self.name} received {self.type} ({symbol}:{lookback_period}) request from agent {sender_id}"
            )
        self.sendMessage(
            sender_id,
            QueryTransactedVolumeReply(
                self.id,
                symbol,
                mkt_closed,
                self.order_books[symbol].get_transacted_volume(lookback_period)
            )
        )

    def processMarketDataSubscriptionMessage(self,
                                             msg: MarketDataSubscription,
                                             current_time: pd.Timestamp) -> None:
        log_print(f"{self.name} received {msg.type} request from agent {msg.sender_id}")
        self.updateSubscriptionDict(msg, current_time)

    def processMarketOpeningHourRequest(self, msg: MarketOpeningHourRequest) -> None:
        sender_id = msg.sender_id
        log_print(f"{self.name} received {msg.type} request from agent {sender_id}")

        # The exchange is permitted to respond to requests for simple immutable data (like "what are your
        # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
        # quotes or trades.
        self.setComputationDelay(0)
        if isinstance(msg, WhenMktOpen):
            reply: MarketOpeningHourReply = WhenMktOpenReply(self.id, self.mkt_open)
        else:
            reply = WhenMktCloseReply(self.id, self.mkt_close)
        self.sendMessage(sender_id, reply)


class OrderBook:
    __slots__ = (
        "exchange",
        "symbol",
        "bids",
        "asks",
        "last_trade",
        "book_log",
        "quotes_seen",
        "history",
        "limit_orders_seen",
        "last_update_ts",
        "_history_unseen",
        "_unrolled_transactions"
    )

    # An OrderBook requires an owning agent object, which it will use to send messages
    # outbound via the simulator Kernel (notifications of order creation, rejection,
    # cancellation, execution, etc).
    def __init__(self, exchange: ExchangeAgent, symbol: str) -> None:
        if not isinstance(exchange, ExchangeAgent):
            raise TypeError('Argument exchange must be of type ExchangeAgent')
        if not isinstance(symbol, str):
            raise TypeError('Argument symbol must be of type str')
        self.exchange = exchange
        self.symbol = symbol
        self.bids: Deque[Deque[Bid]] = deque(())
        self.asks: Deque[Deque[Ask]] = deque(())
        self.last_trade: Optional[int] = None

        # Create an empty list of dictionaries to log the full order book depth (price and volume) each time it changes.
        self.book_log: List[Dict] = []
        self.quotes_seen: MutableSet[int] = set()

        # Last timestamp the orderbook for that symbol was updated
        self.last_update_ts: Optional[pd.Timestamp] = None

        # Create an order history for the exchange to report to certain agent types
        self.history: Deque[OrderBookHistoryStep] = deque([{}], exchange.stream_history + 1)
        self.limit_orders_seen: MutableSet[int] = set()

        # Internal variables used for computing transacted volumes
        self._history_unseen = 1
        self._unrolled_transactions = pd.DataFrame(columns=['execution_time', 'quantity'])

    def handleLimitOrder(self, order: LimitOrder) -> None:
        """
        Matches a limit order or adds it to the order book. Handles partial matches piecewise,
        consuming all possible shares at the best price before moving on, without regard to
        order size "fit" or minimizing number of transactions. Sends one notification per
        match.
        """
        if order.order_id in self.limit_orders_seen:
            print(
                f"WARNING: {order.symbol} order with ID {order.order_id} discarded. "
                "Order with such ID is already present in this Order Book"
            )
            return

        book_symbol = self.symbol
        if order.symbol != book_symbol:
            print(
                f"WARNING: {order.symbol} order with ID {order.order_id} discarded. "
                f"Does not match OrderBook symbol: {book_symbol}"
            )
            return

        if not isinstance(order.quantity, int) or order.quantity <= 0:
            print(
                f"WARNING: {book_symbol} order with ID {order.order_id} discarded. "
                f"Quantity ({order.quantity}) must be a positive integer"
            )
            return

        # Add the order under index 0 of history: orders since the most recent trade
        history = self.history
        history[0][order.order_id] = {
            'entry_time': self.exchange.current_time,
            'quantity': order.quantity,
            'is_buy_order': order.is_buy_order,
            'limit_price': order.limit_price,
            'transactions': [],
            'modifications': [],
            'cancellations': []
        }
        self.limit_orders_seen.add(order.order_id)

        asks = self.asks
        bids = self.bids
        book: Deque[Deque[LimitOrder]] = asks if order.is_buy_order else bids  # type: ignore

        executed = []
        exchange = self.exchange
        exchange_id = exchange.id
        current_time = exchange.current_time
        while book and order.isMatch(best_order := (best_orders := book[0])[0]):
            if order.quantity >= best_order.quantity:
                # Consumed entire matched order
                match = best_order
                del best_orders[0]
                if not best_orders:  # If the matched price now has no orders, remove it completely
                    del book[0]
            else:
                match = deepcopy(best_order)
                match.quantity = order.quantity
                best_order.quantity -= match.quantity

            match.fill_price = match.limit_price

            history[0][order.order_id]['transactions'].append((current_time, order.quantity))
            self._add_event_to_history(match.order_id, current_time, match.quantity, 'transactions')

            filled = deepcopy(order)
            filled.quantity = match.quantity
            filled.fill_price = match.fill_price
            order.quantity -= filled.quantity

            log_print(
                f"MATCHED: new order {filled} vs old order {match}\n"
                f"SENT: notifications of order execution to agents {order.agent_id} "
                f"and {match.agent_id} for orders {order.order_id} and {match.order_id} respectively"
            )

            exchange.sendMessage(order.agent_id, OrderExecuted(exchange_id, filled))
            exchange.sendMessage(match.agent_id, OrderExecuted(exchange_id, match))
            # Accumulate the volume and average share price of the currently executing inbound trade
            executed.append((match.quantity, match.fill_price))

            if order.quantity <= 0:
                break
        else:
            # No matching order was found, so the new order enters the order book. Notify the agent
            self._enterLimitOrder(deepcopy(order))

            log_print(
                f"ACCEPTED: new order {order}\n"
                "SENT: notifications of order acceptance "
                f"to agent {order.agent_id} for order {order.order_id}"
            )

            exchange.sendMessage(order.agent_id, OrderAccepted(exchange_id, order))

        # Now that we are done executing or accepting this order, log the new best bid and ask
        if bids:
            best_bids = bids[0]
            exchange.logEvent(
                'BEST_BID',
                f"{book_symbol},{best_bids[0].limit_price},{sum(o.quantity for o in best_bids)}"
            )

        if asks:
            best_asks = asks[0]
            exchange.logEvent(
                'BEST_ASK',
                f"{book_symbol},{best_asks[0].limit_price},{sum(o.quantity for o in best_asks)}"
            )

        # Also log the last trade (total share quantity, average share price)
        if executed:
            trade_price = trade_qty = 0
            for q, p in executed:
                log_print(f"Executed: {q} @ {p}")
                trade_qty += q
                trade_price += (p * q)  # type: ignore

            avg_price = round(trade_price / trade_qty)
            log_print(f"Avg: {trade_qty} @ ${avg_price:0.4f}")
            exchange.logEvent('LAST_TRADE', f"{trade_qty},${avg_price:0.4f}")

            self.last_trade = avg_price

            # Transaction occurred, so do append left new dict
            self.history.appendleft({})
            self._history_unseen += 1

        # Finally, log the full depth of the order book, ONLY if we have been requested to store the order book
        # for later visualization. (This is slow.)
        book_log = self.book_log
        if exchange.book_freq is not None:
            row: Dict = {'QuoteTime': exchange.current_time}
            quotes_seen = self.quotes_seen
            for quote, volume in self.getInsideBids():
                row[quote] = -volume
                quotes_seen.add(quote)
            for quote, volume in self.getInsideAsks():
                if quote in row:
                    print(
                        "WARNING: THIS IS A REAL PROBLEM: an order book contains bids and asks at the same quote price!"
                    )
                row[quote] = volume
                quotes_seen.add(quote)
            book_log.append(row)
        self.last_update_ts = exchange.current_time
        self.prettyPrint()

    def handleMarketOrder(self, order: MarketOrder) -> None:

        if order.symbol != self.symbol:
            log_print(f"WARNING: {order.symbol} order discarded. Does not match OrderBook symbol: {self.symbol}")
            return

        if not isinstance(order.quantity, int) or order.quantity <= 0:
            log_print(
                f"WARNING: {order.symbol} order discarded. Quantity ({order.quantity}) must be a positive integer."
            )
            return

        is_buy_order = order.is_buy_order
        orderbook_side = self.getInsideAsks() if is_buy_order else self.getInsideBids()

        limit_orders: Dict[int, int] = {}  # limit orders to be placed (key=price, value=quantity)
        order_quantity = order.quantity
        for price, size in orderbook_side:
            if order_quantity <= size:
                limit_orders[price] = order_quantity  # i.e. the top of the book has enough volume for the full order
                break
            limit_orders[price] = size  # i.e. not enough liquidity at the top of the book for the full order
            # therefore walk through the book until all the quantities are matched
            order_quantity -= size
        log_print(f"{order.symbol} placing market order as multiple limit orders of total size {order.quantity}")

        order_type = Bid if is_buy_order else Ask
        for p, q in limit_orders.items():
            limit_order = order_type(order.agent_id, order.time_placed, order.symbol, quantity=q, limit_price=p)
            self.handleLimitOrder(limit_order)

    def _enterLimitOrder(self, order: LimitOrder) -> None:
        # Enters a limit order into the OrderBook in the appropriate location.
        # This does not test for matching/executing orders -- this function
        # should only be called after a failed match/execution attempt.

        book: Deque[Deque[LimitOrder]] = self.asks if order.is_buy_order else self.bids  # type: ignore

        if not book:
            # There were no orders on this side of the book.
            book.append(deque([order]))
        else:
            worst_order = book[-1][0]
            if worst_order.hasBetterPrice(order):
                # There were orders on this side, but this order is worse than all of them.
                # (New lowest bid or highest ask.)
                book.append(deque([order]))
            else:
                # There are orders on this side. Insert this order in the correct position in the list.
                # Note that o is a DEQUE of all orders (oldest at index 0) at this same price.
                populate_new_level = False
                for i, same_price_orders in enumerate(book):
                    first_order = same_price_orders[0]
                    if order.hasEqPrice(first_order):
                        same_price_orders.append(order)
                        break
                    if order.hasBetterPrice(first_order):
                        populate_new_level = True
                        break
                else:
                    print(f"WARNING: enterLimitOrder() called with order {order.order_id}, but it did not enter")
                    return
                if populate_new_level:
                    book.insert(i, deque([order]))
        self.last_update_ts = self.exchange.current_time

    def cancelLimitOrder(self, order: LimitOrder) -> None:
        # Attempts to cancel (the remaining, unexecuted portion of) a trade in the order book.
        # By definition, this pretty much has to be a limit order.  If the order cannot be found
        # in the order book (probably because it was already fully executed), presently there is
        # no message back to the agent. This should possibly change to some kind of failed
        # cancellation message.  (?)  Otherwise, the agent receives ORDER_CANCELLED with the
        # order as the message body, with the cancelled quantity correctly represented as the
        # number of shares that had not already been executed.

        book: Deque[Deque[LimitOrder]] = self.asks if order.is_buy_order else self.bids  # type: ignore
        # If there are no orders on this side of the book, there is nothing to do.
        if not book:
            print(f"WARNING: cancelLimitOrder() called with order {order.order_id}, but OrderBook is empty")
            return

        # There are orders on this side.  Find the price level of the order to cancel,
        # then find the exact order and cancel it.
        # Note that o is a LIST of all orders (oldest at index 0) at this same price.
        order_id = order.order_id
        for i, same_price_orders in enumerate(book):
            if order.hasEqPrice(same_price_orders[0]):
                # This is the correct price level.
                break
        else:
            print(
                f"WARNING: cancelLimitOrder() called with order {order.order_id}, "
                "but OrderBook doesn't have orders with corresponding limit price"
            )
            return

        for ci, order_to_cancel in enumerate(same_price_orders):
            if order_id == order_to_cancel.order_id:
                break
        else:
            print(
                f"WARNING: cancelLimitOrder() called with order {order.order_id}, "
                "but OrderBook doesn't contain order with such ID"
            )
            return

        del same_price_orders[ci]
        # If the cancelled price now has no orders, remove it completely.
        if not same_price_orders:
            del book[i]

        exchange = self.exchange
        current_time = exchange.current_time
        self._add_event_to_history(order_id, current_time, order_to_cancel.quantity, 'cancellations')

        agent_id = order_to_cancel.agent_id
        log_print(
            f"CANCELLED: order {order}\n"
            f"SENT: notifications of order cancellation to agent {agent_id} for order {order_id}"
        )

        exchange.sendMessage(
            agent_id,
            OrderCancelled(exchange.id, order_to_cancel)
        )
        # We found the order and cancelled it, so stop looking.
        self.last_update_ts = current_time

    def modifyLimitOrder(self, order: LimitOrder, new_order: LimitOrder) -> None:
        # Modifies the quantity of an existing limit order in the order book
        if not order.hasSameID(new_order):
            print(
                f"WARNING: modifyLimitOrder() called with order {order.order_id} and new_order {new_order.order_id}, "
                "but their IDs must be equal"
            )
            return

        book: Deque[Deque[LimitOrder]] = self.bids if order.is_buy_order else self.asks  # type: ignore
        if not book:
            print(f"WARNING: modifyLimitOrder() called with order {order.order_id}, but OrderBook is empty")
            return
        for i, same_price_orders in enumerate(book):
            if order.hasEqPrice(same_price_orders[0]):
                break
        else:
            print(
                f"WARNING: modifyLimitOrder() called with order {order.order_id}, "
                "but OrderBook doesn't have orders with corresponding limit price"
            )
            return

        order_id = order.order_id
        for mi, order_to_modify in enumerate(same_price_orders):
            if order_id == order_to_modify.order_id:
                break
        else:
            print(
                f"WARNING: modifyLimitOrder() called with order {order.order_id}, "
                "but OrderBook doesn't contain order with such ID"
            )
            return
        same_price_orders[mi] = new_order

        exchange = self.exchange
        current_time = exchange.current_time
        self._add_event_to_history(order_id, current_time, new_order.quantity, 'modifications')

        agent_id = new_order.agent_id
        log_print(
            f"MODIFIED: order {order}\n"
            f"SENT: notifications of order modification to agent {agent_id} for order {order_id}"
        )

        self.exchange.sendMessage(
            agent_id,
            OrderModified(
                exchange.id,
                new_order
            )
        )
        self.last_update_ts = current_time

    # Get the inside bid price(s) and share volume available at each price, to a limit
    # of "depth".  (i.e. inside price, inside 2 prices)  Returns a list of tuples:
    # list index is best bids (0 is best); each tuple is (price, total shares).
    def getInsideBids(self, depth: int = sys.maxsize) -> List[Tuple[int, int]]:
        book = [
            (same_price_orders[0].limit_price, sum(order.quantity for order in same_price_orders))
            for same_price_orders
            in islice(self.bids, depth)
        ]
        return book

    # As above, except for ask price(s).
    def getInsideAsks(self, depth: int = sys.maxsize) -> List[Tuple[int, int]]:
        book = [
            (same_price_orders[0].limit_price, sum(order.quantity for order in same_price_orders))
            for same_price_orders
            in islice(self.asks, depth)
        ]
        return book

    def _add_event_to_history(self,
                              order_id: int,
                              current_time: pd.Timestamp,
                              quantity: int,
                              update_field: Literal['transactions', 'modifications', 'cancellations']) -> None:
        for orders in self.history:
            history_entry = orders.get(order_id, None)
            if history_entry is not None:
                history_entry[update_field].append((current_time, quantity))
                break

    def _get_recent_history(self) -> Tuple[OrderBookHistoryStep, ...]:
        """ Gets portion of self.history that has arrived since last call of self.get_transacted_volume.
        :return:
        """
        recent_history = tuple(islice(self.history, self._history_unseen))
        self._history_unseen = 0
        return recent_history

    def _update_unrolled_transactions(self, recent_history: Iterable[OrderBookHistoryStep]) -> None:
        """
        Update ``self._unrolled_transactions`` with data from ``recent_history``.
        """
        new_unrolled_txn = self._unrolled_transactions_from_order_history(recent_history)
        self._unrolled_transactions = self._unrolled_transactions.append(new_unrolled_txn, ignore_index=True)

    @staticmethod
    def _unrolled_transactions_from_order_history(history: Iterable[OrderBookHistoryStep]) -> pd.DataFrame:
        """
        Return a DataFrame with columns ``['execution_time', 'quantity']`` from a dictionary with same format as
        ``self.history``, describing executed transactions.
        """
        # Load history into DataFrame
        unrolled_history = [
            value
            for history_entry in history
            for value in history_entry.values()
        ]

        unrolled_history_df = pd.DataFrame(
            unrolled_history,
            columns=[
                'entry_time', 'quantity', 'is_buy_order', 'limit_price', 'transactions', 'modifications',
                'cancellations'
            ]
        )

        if unrolled_history_df.empty:
            return pd.DataFrame(columns=['execution_time', 'quantity'])

        executed_transactions = unrolled_history_df[unrolled_history_df['transactions'].map(bool)]
        # remove cells that are an empty list

        #  Reshape into DataFrame with columns ['execution_time', 'quantity']
        transaction_seq = chain.from_iterable(executed_transactions['transactions'].values)
        unrolled_transactions = pd.DataFrame(transaction_seq, columns=['execution_time', 'quantity'])
        unrolled_transactions.sort_values(by=['execution_time'], inplace=True)
        unrolled_transactions.drop_duplicates(keep='last', inplace=True)

        return unrolled_transactions

    def get_transacted_volume(self, lookback_period: Union[str, pd.Timedelta] = '10min') -> int:
        """
        Retrieve the total transacted volume for a symbol over a lookback period finishing at the current
        simulation time.
        """
        # Update unrolled transactions DataFrame
        recent_history = self._get_recent_history()
        self._update_unrolled_transactions(recent_history)
        unrolled_transactions = self._unrolled_transactions

        #  Get transacted volume in time window
        lookback_pd = pd.to_timedelta(lookback_period)
        window_start = self.exchange.current_time - lookback_pd
        executed_within_lookback_period = unrolled_transactions[unrolled_transactions['execution_time'] >= window_start]
        transacted_volume = executed_within_lookback_period['quantity'].sum()

        return transacted_volume

    def book_log_to_df(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame constructed from the order book log, to be consumed by
        agent.ExchangeAgent.logOrderbookSnapshots.

        The first column of the DataFrame is `QuoteTime`. The succeeding columns are prices quoted during the
        simulation (as taken from self.quotes_seen).

        Each row is a snapshot at a specific time instance. If there is volume at a certain price level (negative
        for bids, positive for asks) this volume is written in the column corresponding to the price level. If there
        is no volume at a given price level, the corresponding column has a `0`.

        The data is stored in a sparse format, such that a value of `0` takes up no space.
        """
        quotes = sorted(self.quotes_seen)
        log_len = len(self.book_log)
        quote_idx_dict = {quote: idx for idx, quote in enumerate(quotes)}
        quote_times = []

        # Construct sparse matrix, where rows are timestamps, columns are quotes and elements are volume.
        S = dok_matrix((log_len, len(quotes)), dtype=int)  # Dictionary Of Keys based sparse matrix.

        for i, row in enumerate(tqdm(self.book_log, desc="Processing orderbook log")):
            quote_times.append(row['QuoteTime'])
            for quote, vol in row.items():
                if quote == "QuoteTime":
                    continue
                S[i, quote_idx_dict[quote]] = vol

        S = S.tocsc()  # Convert this matrix to Compressed Sparse Column format for pandas to consume.
        df = pd.DataFrame.sparse.from_spmatrix(S, columns=quotes)
        df.insert(0, 'QuoteTime', quote_times, allow_duplicates=True)
        return df

    @overload
    def prettyPrint(self, silent: Literal[False] = False) -> None:
        pass

    @overload
    def prettyPrint(self, silent: Literal[True]) -> str:
        pass

    # Print a nicely-formatted view of the current order book.
    def prettyPrint(self, silent: bool = False) -> Optional[str]:
        # Start at the highest ask price and move down.  Then switch to the highest bid price and move down.
        # Show the total volume at each price.  If silent is True, return the accumulated string and print nothing.

        # If the global silent flag is set, skip prettyPrinting entirely, as it takes a LOT of time.
        if be_silent:
            if silent:
                return ''
            return None

        symbol = self.symbol
        exchange = self.exchange
        current_time = exchange.current_time
        historical_price = exchange.oracle.observePrice(
            symbol,
            current_time,
            sigma_n=0,
            random_state=exchange.random_state
        )

        ten_spaces = ' ' * 10
        asks = '\n'.join(
            f"{ten_spaces}{str(quote):10s}{str(volume):10s}"
            for quote, volume in self.getInsideAsks()[-1::-1]
        )
        bids = '\n'.join(
            f"{str(volume):10s}{str(quote):10s}{ten_spaces}"
            for quote, volume in self.getInsideBids()
        )
        book = (
            f"{symbol} order book as of {current_time}\n"
            f"Last trades: simulated {self.last_trade}, historical {historical_price}\n"
            f"{'BID':10s}{'PRICE':10s}{'ASK':10s}\n"
            f"{'---':10s}{'-----':10s}{'---':10s}\n"
            f"{asks}\n{bids}"
        )

        if silent:
            return book
        log_print(book)
