# The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp to open and close trading,
# a list of equity symbols for which it should create order books, a frequency at which to archive snapshots
# of its order books, a pipeline delay (in ns) for order activity, the exchange computation delay (in ns),
# the levels of order stream history to maintain per symbol (maintains all orders that led to the last N trades),
# whether to log all order activity to the agent log, and a random state object (already seeded) to use
# for stochasticity.
import datetime as dt
import warnings
from copy import deepcopy
from typing import Iterable, Dict, Tuple, Union, Optional, TypeVar, Generic

import numpy as np
import pandas as pd

from OrderBook import OrderBook
from agent.FinancialAgent import FinancialAgent
from core import Kernel
from message import (
    MessageAbstractBase,
    Message,

    MarketClosedReply,

    OrderRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    CancelOrderRequest,
    ModifyOrderRequest,

    OrderReply,
    MarketDataSubscriptionMessage,
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

    MarketData
)
from oracle.DataOracle import DataOracle
from oracle.ExternalFileOracle import ExternalFileOracle
from oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from util import log_print

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.set_option('display.max_rows', 500)

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
        mkt_closed = current_time > self.mkt_close
        # Is the exchange closed?  (This block only affects post-close, not pre-open.)
        if isinstance(msg, OrderRequest):
            if mkt_closed:
                log_print(f"{self.name} received {msg.type}: {msg.order}")
                self.sendMessage(msg.sender_id, MarketClosedReply(self.id))
                return
            if self.log_orders:
                self.logEvent(msg.type, msg.order.to_dict())

            if isinstance(msg, LimitOrderRequest):
                self.__LimitOrderRequest(msg)
            elif isinstance(msg, MarketOrderRequest):
                self.__MarketOrderRequest(msg)
            elif isinstance(msg, CancelOrderRequest):
                self.__CancelOrderRequest(msg)
            elif isinstance(msg, ModifyOrderRequest):
                self.__ModifyOrderRequest(msg)

        elif isinstance(msg, Query):
            if mkt_closed:
                log_print(f"{self.name} received {msg.type}, discarded: market is closed.")
                self.sendMessage(msg.sender_id, MarketClosedReply(self.id))
                # Don't do any further processing on these messages!
                return
            self.logEvent(msg.type, sender_id)

            if isinstance(msg, QueryLastTrade):
                self.__QueryLastTrade(msg, sender_id, mkt_closed)
            elif isinstance(msg, QuerySpread):
                self.__QuerySpread(msg, sender_id, mkt_closed)
            elif isinstance(msg, QueryOrderStream):
                self.__QueryOrderStream(msg, sender_id, mkt_closed)
            elif isinstance(msg, QueryTransactedVolume):
                self.__QueryTransactedVolume(msg, sender_id, mkt_closed)
        else:
            self.logEvent(msg.type, sender_id)
            if isinstance(msg, MarketDataSubscriptionMessage):
                self.__MarketDataSubscriptionMessage(msg, sender_id, current_time)
            elif isinstance(msg, MarketOpeningHourRequest):
                self.__MarketOpeningHourRequest(msg, sender_id)
            else:
                log_print(f"{self.name} received {msg.type}, but not handled")

    def updateSubscriptionDict(self, msg: MarketDataSubscriptionMessage, current_time: pd.Timestamp) -> None:
        # The subscription dict is a dictionary with the key = agent ID,
        # value = dict (key = symbol, value = list [levels (no of levels to receive updates for),
        # frequency (min number of ns between messages), last agent update timestamp]
        # e.g. {101 : {'AAPL' : [1, 10, pd.Timestamp(10:00:00)}}
        agent_id = msg.sender_id
        symbol = msg.symbol
        if isinstance(msg, MarketDataSubscriptionRequest):
            self.subscription_dict[agent_id] = {symbol: (msg.levels, msg.freq, current_time)}
        else:
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
                orderbook_last_update = order_book.last_update_ts
                if freq == 0 or \
                        orderbook_last_update > last_agent_update \
                        and (orderbook_last_update - last_agent_update).delta >= freq:
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
                    agent_subscriptions[symbol] = (levels, freq, orderbook_last_update)

    def logOrderBookSnapshots(self, symbol: str) -> None:
        """
        Log full depth quotes (price, volume) from this order book at some pre-determined frequency. Here we are looking at
        the actual log for this order book (i.e. are there snapshots to export, independent of the requested frequency).
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
            super().sendMessage(recipient_id, msg)

    # Simple accessor methods for the market open and close times.
    def getMarketOpen(self) -> pd.Timestamp:
        return self.mkt_open

    def getMarketClose(self) -> pd.Timestamp:
        return self.mkt_close

    def __LimitOrderRequest(self, msg: LimitOrderRequest) -> None:
        order = msg.order
        symbol = order.symbol
        log_print(f"{self.name} received LIMIT_ORDER: {order}")
        if symbol not in self.order_books:
            log_print(f"Limit Order discarded. Unknown symbol: {symbol}")
        else:
            # Hand the order to the order book for processing.
            self.order_books[symbol].handleLimitOrder(deepcopy(order))
            self.publishOrderBookData()

    def __MarketOrderRequest(self, msg: MarketOrderRequest) -> None:
        order = msg.order
        symbol = order.symbol
        log_print(f"{self.name} received MARKET_ORDER: {order}")
        if symbol not in self.order_books:
            log_print(f"Market Order discarded. Unknown symbol: {symbol}")
        else:
            # Hand the market order to the order book for processing.
            self.order_books[symbol].handleMarketOrder(deepcopy(order))
            self.publishOrderBookData()

    def __CancelOrderRequest(self, msg: CancelOrderRequest) -> None:
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
            self.order_books[symbol].cancelOrder(deepcopy(order))
            self.publishOrderBookData()

    def __ModifyOrderRequest(self, msg: ModifyOrderRequest) -> None:
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
            self.order_books[symbol].modifyOrder(deepcopy(order), deepcopy(new_order))
            self.publishOrderBookData()

    def __QueryLastTrade(self, msg: QueryLastTrade, sender_id: int, mkt_closed: bool) -> None:
        symbol = msg.symbol
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

    def __QuerySpread(self, msg: QuerySpread, sender_id: int, mkt_closed: bool) -> None:
        symbol = msg.symbol
        depth = msg.depth
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

    def __QueryOrderStream(self, msg: QueryOrderStream, sender_id: int, mkt_closed: bool) -> None:
        symbol = msg.symbol
        length = msg.length
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
                self.order_books[symbol].history[1:length + 1]
            )
        )

    def __QueryTransactedVolume(self, msg: QueryTransactedVolume, sender_id: int, mkt_closed: bool) -> None:
        symbol = msg.symbol
        lookback_period = msg.lookback_period
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

    def __MarketDataSubscriptionMessage(self,
                                        msg: MarketDataSubscriptionMessage,
                                        sender_id: int,
                                        current_time: pd.Timestamp) -> None:
        log_print(f"{self.name} received {msg.type} request from agent {sender_id}")
        self.updateSubscriptionDict(msg, current_time)

    def __MarketOpeningHourRequest(self, msg: MarketOpeningHourRequest, sender_id: int) -> None:
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
