from typing import List, TypedDict

import numpy as np
import pandas as pd

from abides.agent.TradingAgent import TradingAgent
from abides.oracle.base import Oracle


# >>> TYPING >>>
class FundamentalInfo(TypedDict):
    FundamentalTime: int
    FundamentalValue: int


# <<< TYPING <<<

class FundamentalTrackingAgent(TradingAgent):
    """ Agent who collects and saves to disk noise-free observations of the fundamental. """
    __slots__ = (
        "log_frequency",
        "oracle",
        "fundamental_series",
        "symbol"
    )

    def __init__(self,
                 *,
                 agent_id: int,
                 name: str,
                 log_frequency: bool,
                 symbol: str,
                 oracle: Oracle,
                 log_orders: bool = False) -> None:
        """ Constructor for FundamentalTrackingAgent

            :param log_frequency: Frequency to update log (in nanoseconds)
            :param symbol: symbol for which fundamental is being logged
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=(2 ** 32), dtype=np.uint64)),
            starting_cash=0,
            log_orders=log_orders
        )

        self.log_frequency = log_frequency
        self.fundamental_series: List[FundamentalInfo] = []
        self.symbol = symbol
        self.oracle = oracle

    def kernelStarting(self, start_time: pd.Timestamp) -> None:
        # self.kernel is set in Agent.kernelInitializing()
        # self.exchangeID is set in TradingAgent.kernelStarting()
        super().kernelStarting(start_time)
        if self.oracle is not self.kernel.oracle:
            raise ValueError(f"{self.__class__.__name__} oracle and Kernel oracle should be the same object")

    def kernelStopping(self) -> None:
        """ Stops kernel and saves fundamental series to disk. """
        # Always call parent method to be safe.
        super().kernelStopping()
        self.writeFundamental()

    def measureFundamental(self) -> None:
        """ Saves the fundamental value at self.currentTime to self.fundamental_series. """
        obs_t = self.oracle.observePrice(self.symbol, self.current_time, sigma_n=0)
        self.fundamental_series.append(
            {
                'FundamentalTime': self.current_time,
                'FundamentalValue': obs_t
            }
        )

    def wakeup(self, current_time: pd.Timestamp) -> None:
        """

        :param current_time:
        :return:
        """
        """ Advances agent in time and takes measurement of fundamental. """
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)

        if self.mkt_open and self.mkt_close:
            # No logging if market is closed
            self.measureFundamental()
            self.setWakeup(current_time + self.getWakeFrequency())

    def writeFundamental(self) -> None:
        """ Log fundamental series to file. """
        dfFund = pd.DataFrame(self.fundamental_series)
        dfFund.set_index('FundamentalTime', inplace=True)
        self.writeLog(dfFund, filename=f'fundamental_{self.symbol}_freq_{self.log_frequency}_ns')

        print("Noise-free fundamental archival complete.")

    def getWakeFrequency(self) -> pd.Timedelta:
        return pd.Timedelta(self.log_frequency, unit='ns')
