from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import MutableSet, Optional, Any, Dict

import pandas as pd

from abides.util import get_defined_slots

silent_mode = False

__all__ = (
    "Order",
    "silent_mode"
)


class Order(metaclass=ABCMeta):
    """
    A basic Order type used by an Exchange to conduct trades or maintain an order book.
    This should not be confused with order Messages agents send to request an Order.
    Specific order types will inherit from this (like LimitOrder).
    """
    __slots__ = (
        "agent_id",
        "time_placed",
        "symbol",
        "quantity",
        "order_id",
        "fill_price",
        "tag"
    )
    _counter = 0
    _order_ids: MutableSet[int] = set()

    def __init__(self,
                 agent_id: int,
                 time_placed: pd.Timestamp,
                 symbol: str,
                 quantity: int,
                 order_id: Optional[int] = None,
                 tag: Any = None) -> None:

        self.agent_id = agent_id

        # Time at which the order was created by the agent.
        self.time_placed = time_placed

        # Equity symbol for the order.
        self.symbol = symbol

        # Number of equity units affected by the order.
        self.quantity = quantity

        # Order ID: either self generated or assigned
        order_ids = Order._order_ids
        if order_id is None:
            while Order._counter in order_ids:
                Order._counter += 1
            order_id = Order._counter
            Order._counter += 1

        order_ids.add(order_id)
        self.order_id = order_id

        # Create placeholder fields that don't get filled in until certain
        # events happen. (We could instead subclass to a special FilledOrder
        # class that adds these later?)
        self.fill_price: Optional[int] = None

        # Tag: a free-form user-defined field that can contain any information relevant to the
        #      entity placing the order.  Recommend keeping it alphanumeric rather than
        #      shoving in objects, as it will be there taking memory for the lifetime of the
        #      order and in all logging mechanisms.  Intent: for strategy agents to set tags
        #      to help keep track of the intent of particular orders, to simplify their code.
        self.tag = tag

    def to_dict(self) -> Dict[str, Any]:
        """
        Make dictionary representation of the user-defined attributes in the ``Order`` instance.

        >>> from abides.order.types import Ask, Bid
        >>>
        >>> bid = Bid(2, pd.Timestamp('1989'), 'AAPL', quantity=3, limit_price=324).to_dict()
        >>> del bid['order_id']
        >>> dict_repr = {                          \
            'agent_id':     2,                     \
            'time_placed': '1989-01-01T00:00:00',  \
            'symbol':      'AAPL',                 \
            'quantity':     3,                     \
            'fill_price':   None,                  \
            'tag':          None,                  \
            'limit_price':  324                    \
        }
        >>> assert bid == dict_repr, f"\\nExpected\\n{dict_repr}\\nGot\\n{bid}"

        >>> ask = Ask(5_000_021, pd.Timestamp('2013-11-02'), 'USD/RUB', quantity=20, limit_price=2, tag='$$').to_dict()
        >>> del ask['order_id']
        >>> dict_repr = {                          \
            'agent_id':     5000021,               \
            'time_placed': '2013-11-02T00:00:00',  \
            'symbol':      'USD/RUB',              \
            'quantity':     20,                    \
            'fill_price':   None,                  \
            'tag':         '$$',                   \
            'limit_price':  2                      \
        }
        >>> assert ask == dict_repr, f"\\nExpected\\n{dict_repr}\\nGot\\n{ask}"

        Returns:
            dictionary of fields defined
        """
        self_copy = deepcopy(self)
        as_dict = {
            s: getattr(self_copy, s)
            for s in self.get_defined_slots()
            if hasattr(self_copy, s)
        }
        if hasattr(self_copy, '__dict__'):
            as_dict.update(self_copy.__dict__)
        as_dict['time_placed'] = as_dict['time_placed'].isoformat()
        return as_dict

    get_defined_slots = classmethod(get_defined_slots)

    @abstractmethod
    def __copy__(self) -> 'Order':
        pass

    @abstractmethod
    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> 'Order':
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @property
    @abstractmethod
    def is_buy_order(self) -> bool:
        pass

    def hasSameID(self, other: 'Order') -> bool:
        return self.order_id == other.order_id
