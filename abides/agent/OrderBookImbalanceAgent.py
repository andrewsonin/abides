from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

from abides.agent.TradingAgent import TradingAgent
from abides.util import log_print

matplotlib.use('TkAgg')
_1s_TimeDelta = pd.Timedelta('1s')


class OrderBookImbalanceAgent(TradingAgent):
    # The OrderBookImbalanceAgent is a simple example of an agent that predicts the short term
    # price movement of a security by the preponderance of directionality in the limit order
    # book.  Specifically, it predicts the price will fall if the ratio of bid volume to total volume
    # in the first N levels of the book is greater than a configurable threshold, and will rise if
    # that ratio is below a symmetric value.  There is a trailing stop on the bid_pct to exit.
    #
    # Note that this means the current iteration of the OBI agent is treating order book imbalance
    # as an oscillating indicator rather than a momentum indicator.  (Imbalance to the buy side
    # indicates "overbought" rather than an upward trend.)
    #
    # Parameters unique to this agent:
    # Levels: how much order book depth should be considered when evaluating imbalance?
    # Entry Threshold: how much imbalance is required before the agent takes a non-flat position?
    #                  For example, entry_threshold=0.1 causes long entry at 0.6 or short entry at 0.4.
    # Trail Dist: how far behind the peak bid_pct should the trailing stop follow?

    __slots__ = (
        "symbol",
        "levels",
        "entry_threshold",
        "trail_dist",
        "freq",
        "last_market_data_update",
        "is_long",
        "is_short",
        "trailing_stop",
        "plot_me"
    )

    def __init__(self,
                 *,
                 agent_id: int,
                 name: str,
                 symbol: Optional[str] = None,
                 levels: int = 10,
                 entry_threshold: float = 0.17,
                 trail_dist: float = 0.085,
                 freq: int = 3_600_000_000_000,
                 starting_cash: int = 1_000_000,
                 log_orders: bool = True,
                 random_state: Optional[np.random.RandomState] = None) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            random_state=random_state,
            starting_cash=starting_cash,
            log_orders=log_orders
        )
        self.symbol = symbol
        self.levels = levels
        self.entry_threshold = entry_threshold
        self.trail_dist = trail_dist
        self.freq = freq
        self.last_market_data_update = None
        self.is_long = False
        self.is_short = False

        self.trailing_stop = None
        self.plot_me = []

    def kernelStarting(self, start_time) -> None:
        super().kernelStarting(start_time)

    def wakeup(self, current_time) -> None:
        super().wakeup(current_time)
        super().requestDataSubscription(self.symbol, levels=self.levels, freq=self.freq)
        self.setComputationDelay(1)

    def receiveMessage(self, current_time, msg) -> None:
        super().receiveMessage(current_time, msg)
        if msg.body['msg'] == 'MARKET_DATA':
            self.cancelOrders()

            self.last_market_data_update = current_time
            bids = msg.body['bids']
            asks = msg.body['asks']

            bid_liq = sum(x[1] for x in bids)
            ask_liq = sum(x[1] for x in asks)

            log_print("bid, ask levels: {}", len(bids), len(asks))
            log_print("bids: {}, asks: {}", bids, asks)

            # OBI strategy.
            target = 0

            if bid_liq == 0 or ask_liq == 0:
                log_print("OBI agent inactive: zero bid or ask liquidity")
                return
            else:
                # bid_pct encapsulates both sides of the question, as a normalized expression
                # representing what fraction of total visible volume is on the buy side.
                bid_pct = bid_liq / (bid_liq + ask_liq)

                # If we are short, we need to decide if we should hold or exit.
                if self.is_short:
                    # Update trailing stop.
                    if bid_pct - self.trail_dist > self.trailing_stop:
                        log_print("Trailing stop updated: new > old ({:2f} > {:2f})", bid_pct - self.trail_dist,
                                  self.trailing_stop)
                        self.trailing_stop = bid_pct - self.trail_dist
                    else:
                        log_print("Trailing stop remains: potential < old ({:2f} < {:2f})", bid_pct - self.trail_dist,
                                  self.trailing_stop)

                    # Check the trailing stop.
                    if bid_pct < self.trailing_stop:
                        log_print("OBI agent exiting short position: bid_pct < trailing_stop ({:2f} < {:2f})", bid_pct,
                                  self.trailing_stop)
                        target = 0
                        self.is_short = False
                        self.trailing_stop = None
                    else:
                        log_print("OBI agent holding short position: bid_pct > trailing_stop ({:2f} > {:2f})", bid_pct,
                                  self.trailing_stop)
                        target = -100
                # If we are long, we need to decide if we should hold or exit.
                elif self.is_long:
                    if bid_pct + self.trail_dist < self.trailing_stop:
                        log_print("Trailing stop updated: new < old ({:2f} < {:2f})", bid_pct + self.trail_dist,
                                  self.trailing_stop)
                        self.trailing_stop = bid_pct + self.trail_dist
                    else:
                        log_print("Trailing stop remains: potential > old ({:2f} > {:2f})", bid_pct + self.trail_dist,
                                  self.trailing_stop)

                    # Check the trailing stop.
                    if bid_pct > self.trailing_stop:
                        log_print("OBI agent exiting long position: bid_pct > trailing_stop ({:2f} > {:2f})", bid_pct,
                                  self.trailing_stop)
                        target = 0
                        self.is_long = False
                        self.trailing_stop = None
                    else:
                        log_print("OBI agent holding long position: bid_pct < trailing_stop ({:2f} < {:2f})", bid_pct,
                                  self.trailing_stop)
                        target = 100
                # If we are flat, we need to decide if we should enter (long or short).
                else:
                    if bid_pct > (0.5 + self.entry_threshold):
                        log_print("OBI agent entering long position: bid_pct < entry_threshold ({:2f} < {:2f})",
                                  bid_pct, 0.5 - self.entry_threshold)
                        target = 10000  # TODO: Amount?
                        self.is_long = True
                        self.trailing_stop = bid_pct + self.trail_dist
                        log_print("Initial trailing stop: {:2f}", self.trailing_stop)
                    elif bid_pct < (0.5 - self.entry_threshold):
                        log_print("OBI agent entering short position: bid_pct > entry_threshold ({:2f} > {:2f})",
                                  bid_pct, 0.5 + self.entry_threshold)
                        target = -10000
                        self.is_short = True
                        self.trailing_stop = bid_pct - self.trail_dist
                        log_print("Initial trailing stop: {:2f}", self.trailing_stop)
                    else:
                        log_print("OBI agent staying flat: long_entry < bid_pct < short_entry ({:2f} < {:2f} < {:2f})",
                                  0.5 - self.entry_threshold, bid_pct, 0.5 + self.entry_threshold)
                        target = 0

                self.plot_me.append(
                    {'currentTime': self.current_time, 'midpoint': (asks[0][0] + bids[0][0]) / 2, 'bid_pct': bid_pct})

            # Adjust holdings to target.
            holdings = self.holdings[self.symbol] if self.symbol in self.holdings else 0
            delta = target - holdings
            direction = True if delta > 0 else False
            price = self.computeRequiredPrice(direction, abs(delta), bids, asks)

            log_print("Current holdings: {}", self.holdings)

            if delta == 0:
                log_print("No adjustments to holdings needed.")
            else:
                log_print("Adjusting holdings by {}", delta)
                self.placeLimitOrder(self.symbol, abs(delta), direction, price)

    @staticmethod
    def getWakeFrequency() -> pd.Timedelta:
        return _1s_TimeDelta

    # Computes required limit price to immediately execute a trade for the specified quantity
    # of shares.
    @staticmethod
    def computeRequiredPrice(direction, shares, known_bids, known_asks):
        book = known_asks if direction else known_bids

        # Start at the inside and add up the shares.
        t = 0

        for i in range(len(book)):
            p, v = book[i]
            t += v

            # If we have accumulated enough shares, return this price.
            if t >= shares:
                return p

        # Not enough shares.  Just return worst price (highest ask, lowest bid).
        return book[-1][0]

    # Cancel all open orders.
    # Return value: did we issue any cancellation requests?
    def cancelOrders(self):
        if not self.orders: return False

        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True

    # Lifecycle.
    def kernelTerminating(self):
        # Plotting code is probably not needed here long term, but helps during development.

        # df = pd.DataFrame(self.plotme)
        # df.set_index('currentTime', inplace=True)
        # df.rolling(30).mean().plot(secondary_y=['bid_pct'], figsize=(12,9))
        # plt.show()
        super().kernelTerminating()