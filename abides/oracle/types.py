# The DataOracle class reads real historical trade data (not price or quote)
# from a given date in history to be resimulated.  It stores these trades
# in a time-sorted array at maximum resolution.  It can be called by
# certain "background" agents to obtain noisy observations about the "real"
# price of a stock at a current time.  It is intended to provide some realistic
# behavior and "price gravity" to the simulated market -- i.e. to make the
# market behave something like historical reality in the absence of whatever
# experiment we are running with more active agent types.

import datetime as dt
import os
from bisect import bisect_left
from math import sqrt, exp

import numpy as np
import pandas as pd
from joblib import Memory

from abides.oracle.base import Oracle
from abides.util import log_print

mem = Memory(cachedir='./cache', verbose=0)

__all__ = (
    "DataOracle",
    "ExternalFileOracle",
    "MeanRevertingOracle",
    "SparseMeanRevertingOracle"
)


# @mem.cache
def read_trades(trade_file, symbols):
    log_print("Data not cached. This will take a minute...")

    df = pd.read_pickle(trade_file, compression='bz2')

    # Filter to requested symbols.
    df = df.loc[symbols]

    # Filter duplicate indices (trades on two exchanges at the PRECISE same time).  Rare.
    df = df[~df.index.duplicated(keep='first')]

    # Ensure resulting index is sorted for best performance later on.
    df = df.sort_index()

    return df


class DataOracle(Oracle):
    def __init__(self, historical_date=None, symbols=None, data_dir=None):
        self.historical_date = historical_date
        self.symbols = symbols

        self.mkt_open = None

        # Read historical trades here...
        h = historical_date
        pre = 'ct' if h.year < 2015 else 'ctm'
        trade_file = os.path.join(data_dir, 'trades', f'trades_{h.year}', f'{pre}_{h.year}{h.month:02d}{h.day:02d}.bgz')

        bars_1m_file = os.path.join(data_dir, '1m_ohlc', f'1m_ohlc_{h.year}',
                                    f'{h.year}{h.month:02d}{h.day:02d}_ohlc_1m.bgz')

        log_print("DataOracle initializing trades from file {}", trade_file)
        log_print("DataOracle initializing 1m bars from file {}", bars_1m_file)

        then = dt.datetime.now()
        self.df_trades = read_trades(trade_file, symbols)
        self.df_bars_1m = read_trades(bars_1m_file, symbols)
        now = dt.datetime.now()

        log_print("DataOracle initialized for {} with symbols {}", historical_date, symbols)
        log_print("DataOracle initialization took {}", now - then)

    # Return the daily open price for the symbol given.  The processing to create the 1m OHLC
    # files does propagate the earliest trade backwards, which helps.  The exchange should
    # pass its opening time.
    def getDailyOpenPrice(self, symbol, mkt_open, cents=True):
        # Remember market open time.
        self.mkt_open = mkt_open

        log_print("Oracle: client requested {} at market open: {}", symbol, mkt_open)

        # Find the opening historical price in the 1m OHLC bars for this symbol.
        open = self.df_bars_1m.loc[(symbol, mkt_open.time()), 'open']
        log_print("Oracle: market open price was was {}", open)

        return int(round(open * 100)) if cents else open

    # Return the latest trade price for the symbol at or prior to the given currentTime,
    # which must be of type pd.Timestamp.
    def getLatestTrade(self, symbol, currentTime):

        log_print("Oracle: client requested {} as of {}", symbol, currentTime)

        # See when the last historical trade was, prior to simulated currentTime.
        dt_last_trade = self.df_trades.loc[symbol].index.asof(currentTime)
        if pd.notnull(dt_last_trade):
            last_trade = self.df_trades.loc[(symbol, dt_last_trade)]

            price = last_trade['PRICE']
            time = dt_last_trade

        # If we know the market open time, and the last historical trade was before it, use
        # the market open price instead.  If there were no trades before the requested time,
        # also use the market open price.
        if pd.isnull(dt_last_trade) or (self.mkt_open and time < self.mkt_open):
            price = self.getDailyOpenPrice(symbol, self.mkt_open, cents=False)
            time = self.mkt_open

        log_print("Oracle: latest historical trade was {} at {}", price, time)

        return price

    # Return a noisy observed historical price for agents which have that ability.
    # currentTime must be of type pd.Timestamp.  Only the Exchange or other privileged
    # agents should use noisy=False.
    #
    # NOTE: sigma_n is the observation variance, NOT STANDARD DEVIATION.
    #
    # Each agent must pass its own np.random.RandomState object to the oracle.
    # This helps to preserve the consistency of multiple simulations with experimental
    # changes (if the oracle used a global Random object, simply adding one new agent
    # would change everyone's "noise" on all subsequent observations).
    def observePrice(self, symbol, current_time, sigma_n=0.0001, random_state=None) -> int:
        last_trade_price = self.getLatestTrade(symbol, current_time)

        # Noisy belief is a normal distribution around 1% the last trade price with variance
        # as requested by the agent.
        if sigma_n == 0:
            belief = float(last_trade_price)
        else:
            belief = random_state.normal(loc=last_trade_price, scale=last_trade_price * sqrt(sigma_n))

        log_print("Oracle: giving client value observation {:0.2f}", belief)

        # All simulator prices are specified in integer cents.
        return int(round(belief * 100))


class ExternalFileOracle(Oracle):
    """
    Oracle using an external price series as the fundamental. The external series are specified files in the ABIDES
    config. If an agent requests the fundamental value in between two timestamps the returned fundamental value is
    linearly interpolated.
    """
    __slots__ = (
        "mkt_open",
        "symbols",
        "fundamentals",
        "f_log"
    )

    def __init__(self, symbols):
        self.mkt_open = None
        self.symbols = symbols
        self.fundamentals = self.load_fundamentals()
        self.f_log = {symbol: [] for symbol in symbols}

    def load_fundamentals(self):
        """
        Method extracts fundamentals for each symbol into DataFrames. Note that input files must be of the form
        generated by util/formatting/mid_price_from_orderbook.py.
        """
        fundamentals = {}
        log_print("Oracle: loading fundamental price series...")
        for symbol, params_dict in self.symbols.items():
            fundamental_file_path = params_dict['fundamental_file_path']
            log_print("Oracle: loading {}", fundamental_file_path)
            fundamental_df = pd.read_pickle(fundamental_file_path)
            fundamentals.update({symbol: fundamental_df})

        log_print("Oracle: loading fundamental price series complete!")
        return fundamentals

    def getDailyOpenPrice(self, symbol, mkt_open):

        # Remember market open time.
        self.mkt_open = mkt_open

        log_print("Oracle: client requested {} at market open: {}", symbol, mkt_open)

        # Find the opening historical price or this symbol.
        open_price = self.getPriceAtTime(symbol, mkt_open)
        log_print("Oracle: market open price was was {}", open_price)

        return int(round(open_price))

    def getPriceAtTime(self, symbol, query_time):
        """ Get the true price of a symbol at the requested time.
            :param symbol: which symbol to query
            :type symbol: str
            :param time: at this time
            :type time: pd.Timestamp
        """

        log_print("Oracle: client requested {} as of {}", symbol, query_time)

        fundamental_series = self.fundamentals[symbol]
        time_of_query = pd.Timestamp(query_time)

        series_open_time = fundamental_series.index[0]
        series_close_time = fundamental_series.index[-1]

        if time_of_query < series_open_time:  # time queried before open
            return fundamental_series[0]
        elif time_of_query > series_close_time:  # time queried after close
            return fundamental_series[-1]
        else:  # time queried during trading

            # find indices either side of requested time
            lower_idx = bisect_left(fundamental_series.index, time_of_query) - 1
            upper_idx = lower_idx + 1 if lower_idx < len(fundamental_series.index) - 1 else lower_idx

            # interpolate between values
            lower_val = fundamental_series[lower_idx]
            upper_val = fundamental_series[upper_idx]

            log_print(
                f"DEBUG: lower_idx: {lower_idx}, lower_val: {lower_val}, upper_idx: {upper_idx}, upper_val: {upper_val}")

            interpolated_price = self.getInterpolatedPrice(query_time, fundamental_series.index[lower_idx],
                                                           fundamental_series.index[upper_idx], lower_val, upper_val)
            log_print("Oracle: latest historical trade was {} at {}. Next historical trade is {}. "
                      "Interpolated price is {}", lower_val, query_time, upper_val, interpolated_price)

            self.f_log[symbol].append({'FundamentalTime': query_time, 'FundamentalValue': interpolated_price})

            return interpolated_price

    def observePrice(self, symbol, current_time, sigma_n=0.0001, random_state=None):
        """ Make observation of price at a given time.
        :param symbol: symbol for which to observe price
        :type symbol: str
        :param current_time: time of observation
        :type current_time: pd.Timestamp
        :param sigma_n: Observation noise parameter
        :type sigma_n: float
        :param random_state: random state for Agent making observation
        :type random_state: np.RandomState
        :return: int, price in cents
        """
        true_price = self.getPriceAtTime(symbol, current_time)
        if sigma_n == 0:
            observed = true_price
        else:
            observed = random_state.normal(loc=true_price, scale=sqrt(sigma_n))

        return int(round(observed))

    def getInterpolatedPrice(self, current_time, time_low, time_high, price_low, price_high):
        """ Get the price at current_time, linearly interpolated between price_low and price_high measured at times
            time_low and time_high
            :param current_time: time for which price is to be interpolated
            :type current_time: pd.Timestamp
            :param time_low: time of first fundamental value
            :type time_low: pd.Timestamp
            :param time_high: time of first fundamental value
            :type time_high: pd.Timestamp
            :param price_low: first fundamental value
            :type price_low: float
            :param price_high: first fundamental value
            :type price_high: float
            :return float of interpolated price:
        """
        log_print(
            f'DEBUG: current_time: {current_time} time_low {time_low} time_high: {time_high} price_low:  {price_low} price_high: {price_high}')
        delta_y = price_high - price_low
        delta_x = (time_high - time_low).total_seconds()

        slope = delta_y / delta_x if price_low != price_high else 0
        x_fwd = (current_time - time_low).total_seconds()

        return price_low + (x_fwd * slope)


class MeanRevertingOracle(Oracle):
    """
    The MeanRevertingOracle requires three parameters: a mean fundamental value,
    a mean reversion coefficient, and a shock variance.  It constructs and retains
    a fundamental value time series for each requested symbol, and provides noisy
    observations of those values upon agent request.  The expectation is that
    agents using such an oracle will know the mean-reverting equation and all
    relevant parameters, but will not know the random shocks applied to the
    sequence at each time step.

    Historical dates are effectively meaningless to this oracle.  It is driven by
    the numpy random number seed contained within the experimental config file.
    This oracle uses the nanoseconds portion of the current simulation time as
    discrete "time steps".  A suggestion: to keep wallclock runtime reasonable,
    have the agents operate for only ~1000 nanoseconds, but interpret nanoseconds
    as seconds or minutes.
    """
    __slots__ = (
        "mkt_open",
        "mkt_close",
        "symbols",
        "r"
    )

    def __init__(self, mkt_open, mkt_close, symbols):
        # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
        # inner keys: r_bar, kappa, sigma_s.
        self.mkt_open = mkt_open
        self.mkt_close = mkt_close
        self.symbols = symbols

        # The dictionary r holds the fundamenal value series for each symbol.
        self.r = {}

        then = dt.datetime.now()

        for symbol in symbols:
            s = symbols[symbol]
            log_print("MeanRevertingOracle computing fundamental value series for {}", symbol)
            self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol, **s)

        now = dt.datetime.now()

        log_print("MeanRevertingOracle initialized for symbols {}", symbols)
        log_print("MeanRevertingOracle initialization took {}", now - then)

    def generate_fundamental_value_series(self, symbol, r_bar, kappa, sigma_s):
        # Generates the fundamental value series for a single stock symbol.  r_bar is the
        # mean fundamental value, kappa is the mean reversion coefficient, and sigma_s
        # is the shock variance.  (Note: NOT STANDARD DEVIATION.)

        # Because the oracle uses the global np.random PRNG to create the fundamental value
        # series, it is important to create the oracle BEFORE the agents.  In this way the
        # addition of a new agent will not affect the sequence created.  (Observations using
        # the oracle will use an agent's PRNG and thus not cause a problem.)

        # Turn variance into std.
        sigma_s = sqrt(sigma_s)

        # Create the time series into which values will be projected and initialize the first value.
        date_range = pd.date_range(self.mkt_open, self.mkt_close, closed='left', freq='N')

        s = pd.Series(index=date_range)
        r = np.zeros(len(s.index))
        r[0] = r_bar

        # Predetermine the random shocks for all time steps (at once, for computation speed).
        shock = np.random.normal(scale=sigma_s, size=(r.shape[0]))

        # Compute the mean reverting fundamental value series.
        for t in range(1, r.shape[0]):
            r[t] = max(0, (kappa * r_bar) + ((1 - kappa) * r[t - 1]) + shock[t])

        # Replace the series values with the fundamental value series.  Round and convert to
        # integer cents.
        s[:] = np.round(r)
        s = s.astype(int)

        return s

    # Return the daily open price for the symbol given.  In the case of the MeanRevertingOracle,
    # this will simply be the first fundamental value, which is also the fundamental mean.
    # We will use the mkt_open time as given, however, even if it disagrees with this.
    def getDailyOpenPrice(self, symbol, mkt_open=None):

        # If we did not already know mkt_open, we should remember it.
        if (mkt_open is not None) and (self.mkt_open is None):
            self.mkt_open = mkt_open

        log_print("Oracle: client requested {} at market open: {}", symbol, self.mkt_open)

        open = self.r[symbol].loc[self.mkt_open]
        log_print("Oracle: market open price was was {}", open)

        return open

    # Return a noisy observation of the current fundamental value.  While the fundamental
    # value for a given equity at a given time step does not change, multiple agents
    # observing that value will receive different observations.
    #
    # Only the Exchange or other privileged agents should use noisy=False.
    #
    # sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.
    #
    # Each agent must pass its RandomState object to observePrice.  This ensures that
    # each agent will receive the same answers across multiple same-seed simulations
    # even if a new agent has been added to the experiment.
    def observePrice(self, symbol, current_time, sigma_n=1000, random_state=None):
        # If the request is made after market close, return the close price.
        if current_time >= self.mkt_close:
            r_t = self.r[symbol].loc[self.mkt_close - pd.Timedelta('1ns')]
        else:
            r_t = self.r[symbol].loc[current_time]

        # Generate a noisy observation of fundamental value at the current time.
        if sigma_n == 0:
            obs = r_t
        else:
            obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))

        log_print("Oracle: current fundamental value is {} at {}", r_t, current_time)
        log_print("Oracle: giving client value observation {}", obs)

        # Reminder: all simulator prices are specified in integer cents.
        return obs


class SparseMeanRevertingOracle(MeanRevertingOracle):
    """
    The SparseMeanRevertingOracle produces a fundamental value time series for
    each requested symbol, and provides noisy observations of the fundamental
    value upon agent request.  This "sparse discrete" fundamental uses a
    combination of two processes to produce relatively realistic synthetic
    "values": a continuous mean-reverting Ornstein-Uhlenbeck process plus
    periodic "megashocks" which arrive following a Poisson process and have
    magnitude drawn from a bimodal normal distribution (overall mean zero,
    but with modes well away from zero).  This is necessary because OU itself
    is a single noisy return to the mean (from a perturbed initial state)
    that does not then depart the mean except in terms of minor "noise".

    Historical dates are effectively meaningless to this oracle.  It is driven by
    the numpy random number seed contained within the experimental config file.
    This oracle uses the nanoseconds portion of the current simulation time as
    discrete "time steps".

    This version of the MeanRevertingOracle expects agent activity to be spread
    across a large amount of time, with relatively sparse activity.  That is,
    agents each acting at realistic "retail" intervals, on the order of seconds
    or minutes, spread out across the day.
    """
    __slots__ = (
        "f_log",
        "megashocks"
    )

    def __init__(self, mkt_open, mkt_close, symbols):
        # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
        # inner keys: r_bar, kappa, sigma_s.
        self.mkt_open = mkt_open
        self.mkt_close = mkt_close
        self.symbols = symbols
        self.f_log = {}

        # The dictionary r holds the most recent fundamental values for each symbol.
        self.r = {}

        # The dictionary megashocks holds the time series of megashocks for each symbol.
        # The last one will always be in the future (relative to the current simulation time).
        #
        # Without these, the OU process just makes a noisy return to the mean and then stays there
        # with relatively minor noise.  Here we want them to follow a Poisson process, so we sample
        # from an exponential distribution for the separation intervals.
        self.megashocks = {}

        then = dt.datetime.now()

        # Note that each value in the self.r dictionary is a 2-tuple of the timestamp at
        # which the series was computed and the true fundamental value at that time.
        for symbol in symbols:
            s = symbols[symbol]
            log_print("SparseMeanRevertingOracle computing initial fundamental value for {}", symbol)
            self.r[symbol] = (mkt_open, s['r_bar'])
            self.f_log[symbol] = [{'FundamentalTime': mkt_open, 'FundamentalValue': s['r_bar']}]

            # Compute the time and value of the first megashock.  Note that while the values are
            # mean-zero, they are intentionally bimodal (i.e. we always want to push the stock
            # some, but we will tend to cancel out via pushes in opposite directions).
            ms_time_delta = np.random.exponential(scale=1.0 / s['megashock_lambda_a'])
            mst = self.mkt_open + pd.Timedelta(ms_time_delta, unit='ns')
            msv = s['random_state'].normal(loc=s['megashock_mean'], scale=sqrt(s['megashock_var']))
            msv = msv if s['random_state'].randint(2) == 0 else -msv

            self.megashocks[symbol] = [{'MegashockTime': mst, 'MegashockValue': msv}]

        now = dt.datetime.now()

        log_print("SparseMeanRevertingOracle initialized for symbols {}", symbols)
        log_print("SparseMeanRevertingOracle initialization took {}", now - then)

    # This method takes a requested timestamp to which we should advance the fundamental,
    # a value adjustment to apply after advancing time (must pass zero if none),
    # a symbol for which to advance time, a previous timestamp, and a previous fundamental
    # value.  The last two parameters should relate to the most recent time this method
    # was invoked.  It returns the new value.  As a side effect, it updates the log of
    # computed fundamental values.

    def compute_fundamental_at_timestamp(self, ts, v_adj, symbol, pt, pv):
        s = self.symbols[symbol]

        # This oracle uses the Ornstein-Uhlenbeck Process.  It is quite close to being a
        # continuous version of the discrete mean reverting process used in the regular
        # (dense) MeanRevertingOracle.

        # Compute the time delta from the previous time to the requested time.
        d = int((ts - pt) / np.timedelta64(1, 'ns'))

        # Extract the parameters for the OU process update.
        mu = s['r_bar']
        gamma = s['kappa']
        theta = s['fund_vol']

        # The OU process is able to skip any amount of time and sample the next desired value
        # from the appropriate distribution of possible values.
        v = s['random_state'].normal(loc=mu + (pv - mu) * (exp(-gamma * d)),
                                     scale=((theta) / (2 * gamma)) * (1 - exp(-2 * gamma * d)))

        # Apply the value adjustment that was passed in.
        v += v_adj

        # The process is not permitted to become negative.
        v = max(0, v)

        # For our purposes, the value must be rounded and converted to integer cents.
        v = int(round(v))

        # Cache the new time and value as the "previous" fundamental values.
        self.r[symbol] = (ts, v)

        # Append the change to the permanent log of fundamental values for this symbol.
        self.f_log[symbol].append({'FundamentalTime': ts, 'FundamentalValue': v})

        # Return the new value for the requested timestamp.
        return v

    # This method advances the fundamental value series for a single stock symbol,
    # using the OU process.  It may proceed in several steps due to our periodic
    # application of "megashocks" to push the stock price around, simulating
    # exogenous forces.
    def advance_fundamental_value_series(self, currentTime, symbol):

        # Generation of the fundamental value series uses a separate random state object
        # per symbol, which is part of the dictionary we maintain for each symbol.
        # Agent observations using the oracle will use an agent's random state object.
        s = self.symbols[symbol]

        # This is the previous fundamental time and value.
        pt, pv = self.r[symbol]

        # If time hasn't changed since the last advance, just use the current value.
        if currentTime <= pt: return pv

        # Otherwise, we have some work to do, advancing time and computing the fundamental.

        # We may not jump straight to the requested time, because we periodically apply
        # megashocks to push the series around (not always away from the mean) and we need
        # to compute OU at each of those times, so the aftereffects of the megashocks
        # properly affect the remaining OU interval.

        mst = self.megashocks[symbol][-1]['MegashockTime']
        msv = self.megashocks[symbol][-1]['MegashockValue']

        while mst < currentTime:
            # A megashock is scheduled to occur before the new time to which we are advancing.  Handle it.

            # Advance time from the previous time to the time of the megashock using the OU process and
            # then applying the next megashock value.
            v = self.compute_fundamental_at_timestamp(mst, msv, symbol, pt, pv)

            # Update our "previous" values for the next computation.
            pt, pv = mst, v

            # Since we just surpassed the last megashock time, compute the next one, which we might or
            # might not immediately consume.  This works just like the first time (in __init__()).

            mst = pt + pd.Timedelta('{}ns'.format(np.random.exponential(scale=1.0 / s['megashock_lambda_a'])))
            msv = s['random_state'].normal(loc=s['megashock_mean'], scale=sqrt(s['megashock_var']))
            msv = msv if s['random_state'].randint(2) == 0 else -msv

            self.megashocks[symbol].append({'MegashockTime': mst, 'MegashockValue': msv})

            # The loop will continue until there are no more megashocks before the time requested
            # by the calling method.

        # Once there are no more megashocks to apply (i.e. the next megashock is in the future, after
        # currentTime), then finally advance using the OU process to the requested time.
        v = self.compute_fundamental_at_timestamp(currentTime, 0, symbol, pt, pv)

        return v

    # Return the daily open price for the symbol given.  In the case of the MeanRevertingOracle,
    # this will simply be the first fundamental value, which is also the fundamental mean.
    # We will use the mkt_open time as given, however, even if it disagrees with this.
    def getDailyOpenPrice(self, symbol, mkt_open=None):

        # The sparse oracle doesn't maintain full fundamental value history, but rather
        # advances on demand keeping only the most recent price, except for the opening
        # price.  Thus we cannot honor a mkt_open that isn't what we already expected.

        log_print("Oracle: client requested {} at market open: {}", symbol, self.mkt_open)

        open = self.symbols[symbol]['r_bar']
        log_print("Oracle: market open price was was {}", open)

        return open

    # Return a noisy observation of the current fundamental value.  While the fundamental
    # value for a given equity at a given time step does not change, multiple agents
    # observing that value will receive different observations.
    #
    # Only the Exchange or other privileged agents should use sigma_n==0.
    #
    # sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.
    #
    # Each agent must pass its RandomState object to observePrice.  This ensures that
    # each agent will receive the same answers across multiple same-seed simulations
    # even if a new agent has been added to the experiment.
    def observePrice(self, symbol, current_time, sigma_n=1000, random_state=None):
        # If the request is made after market close, return the close price.
        if current_time >= self.mkt_close:
            r_t = self.advance_fundamental_value_series(self.mkt_close - pd.Timedelta('1ns'), symbol)
        else:
            r_t = self.advance_fundamental_value_series(current_time, symbol)

        # Generate a noisy observation of fundamental value at the current time.
        if sigma_n == 0:
            obs = r_t
        else:
            obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))

        log_print("Oracle: current fundamental value is {} at {}", r_t, current_time)
        log_print("Oracle: giving client value observation {}", obs)

        # Reminder: all simulator prices are specified in integer cents.
        return obs
