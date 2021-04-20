from traceback import format_stack
from typing import List, Iterable, Union

from typing_extensions import overload

from core import Agent


@overload
def dollarize(cents: Iterable[int]) -> List[str]:
    pass


@overload
def dollarize(cents: int) -> str:
    pass


# Dollarizes int-cents prices for printing. Defined outside the class for
# utility access by non-agent classes.
def dollarize(cents: Union[Iterable[int], int]) -> Union[List[str], str]:
    if isinstance(cents, Iterable):
        return list(map(dollarize, cents))  # type: ignore
    elif isinstance(cents, int):
        return f"${cents / 100:0.2}"
    else:
        # If cents is already a float, there is an error somewhere.
        error_msg = f"ERROR: dollarize(cents) called without int or iterable of ints: {cents}"
        print(error_msg)
        raise TypeError(error_msg, "Current traceback:", ''.join(format_stack()))


# The FinancialAgent class contains attributes and methods that should be available
# to all agent types (traders, exchanges, etc) in a financial market simulation.
# To be honest, it mainly exists because the base Agent class should not have any
# finance-specific aspects and it doesn't make sense for ExchangeAgent to inherit
# from TradingAgent. Hopefully we'll find more common ground for traders and
# exchanges to make this more useful later on.
class FinancialAgent(Agent):
    __slots__ = ()
    # Used by any subclass to dollarize an int-cents price for printing.
    dollarize = staticmethod(dollarize)
