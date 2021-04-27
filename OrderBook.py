# Basic class for an order book for one symbol, in the style of the major US Stock Exchanges.
# List of bid prices (index zero is best bid), each with a list of LimitOrders.
# List of ask prices (index zero is best ask), each with a list of LimitOrders.
import sys
from collections import deque
from copy import deepcopy
from itertools import islice, chain
from typing import List, Deque, Tuple, Dict, MutableSet, Iterable, Optional, Literal

import pandas as pd
from scipy.sparse import dok_matrix
from tqdm import tqdm

from abides._typing import OrderBookHistoryEntry
from abides.message.types import OrderExecuted, OrderAccepted, OrderCancelled, OrderModified
from agent.ExchangeAgent import ExchangeAgent
from order import LimitOrder, Bid, Ask, MarketOrder
from util import log_print, be_silent


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
        "last_update_ts",
        "limit_ids_seen",
        "_history_unseen",
        "_unrolled_transactions"
    )

    # An OrderBook requires an owning agent object, which it will use to send messages
    # outbound via the simulator Kernel (notifications of order creation, rejection,
    # cancellation, execution, etc).
    def __init__(self, exchange: ExchangeAgent, symbol: str) -> None:
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
        self.history: Deque[OrderBookHistoryEntry] = deque((), exchange.stream_history)
        self.limit_ids_seen: MutableSet[int] = set()

        # Internal variables used for computing transacted volumes
        self._history_unseen = 0
        self._unrolled_transactions = None

    def handleLimitOrder(self, order: LimitOrder) -> None:
        # Matches a limit order or adds it to the order book. Handles partial matches piecewise,
        # consuming all possible shares at the best price before moving on, without regard to
        # order size "fit" or minimizing number of transactions. Sends one notification per
        # match.
        book_symbol = self.symbol
        order_symbol = order.symbol
        if order_symbol != book_symbol:
            log_print(f"{order_symbol} order discarded. Does not match OrderBook symbol: {book_symbol}")
            return

        order_quantity = order.quantity
        if not isinstance(order_quantity, int) or order_quantity <= 0:
            log_print(f"{order_symbol} order discarded. Quantity ({order_quantity}) must be a positive integer.")
            return

        # Add the order under index 0 of history: orders since the most recent trade
        exchange = self.exchange
        history = self.history
        submitted_order_id = order.order_id
        submitted_agent_id = order.agent_id
        history.appendleft(
            {
                'order_id': submitted_order_id,
                'entry_time': exchange.current_time,
                'quantity': order.quantity,
                'is_buy_order': order.is_buy_order,
                'limit_price': order.limit_price,
                'transactions': [],
                'modifications': [],
                'cancellations': []
            }
        )
        self.limit_ids_seen.add(submitted_order_id)
        self._history_unseen += 1

        exchange_id = exchange.id
        executed = []
        while True:
            matched_order = self.executeLimitOrder(order)

            if matched_order is not None:
                matched_order = deepcopy(matched_order)
                matched_agent_id = matched_order.agent_id

                # Decrement quantity on new order and notify traders of execution
                filled_order = deepcopy(order)

                fill_price = matched_order.fill_price
                match_quantity = matched_order.quantity

                filled_order.quantity = match_quantity
                filled_order.fill_price = fill_price

                order.quantity -= match_quantity

                log_print(
                    f"MATCHED: new order {filled_order} vs old order {matched_order}\n"
                    f"SENT: notifications of order execution to agents {submitted_agent_id} "
                    f"and {matched_agent_id} for orders {submitted_order_id} and {matched_order.order_id}"
                )

                exchange.sendMessage(
                    submitted_agent_id,
                    OrderExecuted(exchange_id, filled_order)
                )
                exchange.sendMessage(
                    matched_agent_id,
                    OrderExecuted(exchange_id, matched_order)
                )
                # Accumulate the volume and average share price of the currently executing inbound trade
                executed.append((match_quantity, fill_price))

                if order.quantity <= 0:
                    break
            else:
                # No matching order was found, so the new order enters the order book. Notify the agent
                self.enterLimitOrder(deepcopy(order))

                log_print(
                    f"ACCEPTED: new order {order}\n"
                    "SENT: notifications of order acceptance "
                    f"to agent {submitted_agent_id} for order {submitted_order_id}"
                )

                exchange.sendMessage(submitted_agent_id, OrderAccepted(exchange_id, order))
                break

        # Now that we are done executing or accepting this order, log the new best bid and ask.
        bids = self.bids
        if bids:
            best_bids = bids[0]
            exchange.logEvent(
                'BEST_BID',
                f"{book_symbol},{best_bids[0].limit_price},{sum(o.quantity for o in best_bids)}"
            )

        asks = self.asks
        if asks:
            best_asks = asks[0]
            exchange.logEvent(
                'BEST_ASK',
                f"{book_symbol},{best_asks[0].limit_price},{sum(o.quantity for o in best_asks)}"
            )

        # Also log the last trade (total share quantity, average share price).
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
            log_print(f"{order.symbol} order discarded. Does not match OrderBook symbol: {self.symbol}")
            return

        if not isinstance(order.quantity, int) or order.quantity <= 0:
            log_print(f"{order.symbol} order discarded. Quantity ({order.quantity}) must be a positive integer.")
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

    def executeLimitOrder(self, order: LimitOrder) -> Optional[LimitOrder]:
        # Finds a single best match for this order, without regard for quantity.
        # Returns the matched order or None if no match found. DOES remove,
        # or decrement quantity from, the matched order from the order book
        # (i.e. executes at least a partial trade, if possible).

        # Track which (if any) existing order was matched with the current order.
        book: Deque[Deque[LimitOrder]] = self.asks if order.is_buy_order else self.bids  # type: ignore

        # TODO: Simplify?  It is ever possible to actually select an execution match
        # other than the best bid or best ask? We may not need these execute loops.

        # First, examine the correct side of the order book for a match.
        if not book:
            # No orders on this side.
            return None

        best_limit_orders = book[0]
        best_order = best_limit_orders[0]
        if not order.isMatch(best_order):
            # There were orders on the right side, but the prices do not overlap.
            # Or: bid could not match with best ask, or vice versa.
            # Or: bid offer is below the lowest asking price, or vice versa.
            return None
        # There are orders on the right side, and the new order's price does fall
        # somewhere within them.  We can/will only match against the oldest order
        # among those with the best price.  (i.e. best price, then FIFO)

        # Note that book[i] is a LIST of all orders (oldest at index book[i][0]) at the same price.

        # The matched order might be only partially filled. (i.e. new order is smaller)
        if order.quantity >= best_order.quantity:
            # Consumed entire matched order.
            matched_order = best_limit_orders.popleft()

            # If the matched price now has no orders, remove it completely.
            if not best_limit_orders:
                book.popleft()

        else:
            # Consumed only part of matched order.
            matched_order = deepcopy(best_order)
            matched_order.quantity = order.quantity

            best_order.quantity -= matched_order.quantity

        # When two limit orders are matched, they execute at the price that
        # was being "advertised" in the order book.
        matched_order.fill_price = matched_order.limit_price

        # Record the transaction in the order history and push the indices
        # out one, possibly truncating to the maximum history length.

        # The incoming order is guaranteed to exist under index 0.
        current_time = self.exchange.current_time
        self.history[0]['transactions'].append((current_time, order.quantity))

        # The pre-existing order may or may not still be in the recent history.
        self._add_event_to_history(matched_order.order_id, current_time, matched_order.quantity, 'transactions')

        # Return (only the executed portion of) the matched order.
        return matched_order

    def enterLimitOrder(self, order: LimitOrder) -> None:
        # Enters a limit order into the OrderBook in the appropriate location.
        # This does not test for matching/executing orders -- this function
        # should only be called after a failed match/execution attempt.

        book: Deque[Deque[LimitOrder]] = self.asks if order.is_buy_order else self.bids  # type: ignore

        if not book:
            # There were no orders on this side of the book.
            book.append(deque([order]))
        else:
            worst_order = book[-1][0]
            if not self.isBetterPrice(order, worst_order) and not order.hasEqPrice(worst_order):
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
                    if self.isBetterPrice(order, first_order):
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
        if not order.hasEqID(new_order):
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
        for history_entry in self.history:
            if order_id == history_entry['order_id']:
                history_entry[update_field].append((current_time, quantity))
                break

    def _get_recent_history(self) -> Tuple[OrderBookHistoryEntry, ...]:
        """ Gets portion of self.history that has arrived since last call of self.get_transacted_volume.
        :return:
        """
        recent_history = tuple(islice(self.history, self._history_unseen))
        self._history_unseen = 0
        return recent_history

    def _update_unrolled_transactions(self, recent_history: Iterable[OrderBookHistoryEntry]) -> None:
        """ Updates self._transacted_volume["unrolled_transactions"] with data from recent_history
        """
        new_unrolled_txn = self._unrolled_transactions_from_order_history(recent_history)
        old_unrolled_txn = self._unrolled_transactions
        total_unrolled_txn = pd.concat((old_unrolled_txn, new_unrolled_txn), ignore_index=True)
        self._unrolled_transactions = total_unrolled_txn

    @staticmethod
    def _unrolled_transactions_from_order_history(history):
        """ Returns a DataFrame with columns ['execution_time', 'quantity'] from a dictionary with same format as
            self.history, describing executed transactions.
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

    def get_transacted_volume(self, lookback_period='10min'):
        """ Method retrieves the total transacted volume for a symbol over a lookback period finishing at the current
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

    # These could be moved to the LimitOrder class.  We could even operator overload them
    # into >, <, ==, etc.
    @staticmethod
    def isBetterPrice(order, o):
        # Returns True if order has a 'better' price than o.  (That is, a higher bid
        # or a lower ask.)  Must be same order type.
        if order.is_buy_order != o.is_buy_order:
            print(f"WARNING: isBetterPrice() called on orders of different type: {order} vs {o}")
            return False

        if order.is_buy_order and (order.limit_price > o.limit_price):
            return True

        if not order.is_buy_order and (order.limit_price < o.limit_price):
            return True

        return False

    @staticmethod
    def isSameOrder(order, new_order):
        return order.order_id == new_order.order_id

    def book_log_to_df(self):
        """ Returns a pandas DataFrame constructed from the order book log, to be consumed by
            agent.ExchangeAgent.logOrderbookSnapshots.

            The first column of the DataFrame is `QuoteTime`. The succeeding columns are prices quoted during the
            simulation (as taken from self.quotes_seen).

            Each row is a snapshot at a specific time instance. If there is volume at a certain price level (negative
            for bids, positive for asks) this volume is written in the column corresponding to the price level. If there
            is no volume at a given price level, the corresponding column has a `0`.

            The data is stored in a sparse format, such that a value of `0` takes up no space.

        :return:
        """
        quotes = sorted(self.quotes_seen)
        log_len = len(self.book_log)
        quote_idx_dict = {quote: idx for idx, quote in enumerate(quotes)}
        quotes_times = []

        # Construct sparse matrix, where rows are timesteps, columns are quotes and elements are volume.
        S = dok_matrix((log_len, len(quotes)), dtype=int)  # Dictionary Of Keys based sparse matrix.

        for i, row in enumerate(tqdm(self.book_log, desc="Processing orderbook log")):
            quotes_times.append(row['QuoteTime'])
            for quote, vol in row.items():
                if quote == "QuoteTime":
                    continue
                S[i, quote_idx_dict[quote]] = vol

        S = S.tocsc()  # Convert this matrix to Compressed Sparse Column format for pandas to consume.
        df = pd.DataFrame.sparse.from_spmatrix(S, columns=quotes)
        df.insert(0, 'QuoteTime', quotes_times, allow_duplicates=True)
        return df

    # Print a nicely-formatted view of the current order book.
    def prettyPrint(self, silent=False):
        # Start at the highest ask price and move down.  Then switch to the highest bid price and move down.
        # Show the total volume at each price.  If silent is True, return the accumulated string and print nothing.

        # If the global silent flag is set, skip prettyPrinting entirely, as it takes a LOT of time.
        if be_silent: return ''

        book = "{} order book as of {}\n".format(self.symbol, self.exchange.current_time)
        book += "Last trades: simulated {:d}, historical {:d}\n".format(self.last_trade,
                                                                        self.exchange.oracle.observePrice(self.symbol,
                                                                                                          self.exchange.current_time,
                                                                                                          sigma_n=0,
                                                                                                          random_state=self.exchange.random_state))

        book += "{:10s}{:10s}{:10s}\n".format('BID', 'PRICE', 'ASK')
        book += "{:10s}{:10s}{:10s}\n".format('---', '-----', '---')

        for quote, volume in self.getInsideAsks()[-1::-1]:
            book += "{:10s}{:10s}{:10s}\n".format("", "{:d}".format(quote), "{:d}".format(volume))

        for quote, volume in self.getInsideBids():
            book += "{:10s}{:10s}{:10s}\n".format("{:d}".format(volume), "{:d}".format(quote), "")

        if silent: return book

        log_print(book)
