from heapq import heappop, heappush
from typing import Generic

from backtesting.typing.vars import T

__slots__ = (
    "PriorityQueue",
)


class PriorityQueue(list, Generic[T]):
    """
    As fast as possible object-oriented implementation of priority queue in Python.
    Attention: it is NOT thread-safe
    """
    __slots__ = ()

    def get(self) -> T:
        return heappop(self)

    def put(self, item: T) -> None:
        heappush(self, item)
