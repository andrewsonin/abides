from _heapq import heappop, heappush, heapify  # Guaranteed C-implementation
from typing import Optional, Generic, Iterable

from backtesting.typing.vars import T

__slots__ = (
    "PriorityQueue",
)

_private_property = property(doc="Use of private base class method")


class PriorityQueue(list, Generic[T]):
    """
    As fast as possible object-oriented implementation of priority queue in Python.
    Attention: it is NOT thread-safe
    """
    __slots__ = ()

    def __init__(self, value: Optional[Iterable[T]] = None) -> None:
        if value is None:
            super().__init__()
        else:
            super().__init__(value)
            heapify(self)

    def get(self) -> T:
        return heappop(self)

    def put(self, item: T) -> None:
        heappush(self, item)

    # >>> Delete base 'list' class methods to avoid unintended heap disruption >>>
    append = extend = insert = remove = pop = index = sort = reverse = copy = _private_property  # type: ignore
    __add__ = __iadd__ = __mul__ = __imul__ = __rmul__ = _private_property  # type: ignore
    __setitem__ = __delitem__ = _private_property  # type: ignore
