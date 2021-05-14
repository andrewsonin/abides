from backtesting.core import Agent
from backtesting.utils.util import dollarize


class FinancialAgent(Agent):
    """
    The FinancialAgent class contains attributes and methods that should be available
    to all agent types (traders, exchanges, etc) in a financial market simulation.
    To be honest, it mainly exists because the base Agent class should not have any
    finance-specific aspects and it doesn't make sense for ExchangeAgent to inherit
    from TradingAgent. Hopefully we'll find more common ground for traders and
    exchanges to make this more useful later on.
    """
    __slots__ = ()
    # Used by any subclass to dollarize an int-cents price for printing.
    dollarize = staticmethod(dollarize)
