from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

__all__ = (
    "Oracle",
)


class Oracle(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def observePrice(self,
                     symbol: str,
                     current_time: pd.Timestamp,
                     sigma_n: int = 1_000,
                     random_state: Optional[np.random.RandomState] = None) -> int:
        pass
