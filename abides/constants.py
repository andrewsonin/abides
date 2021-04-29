import numpy as np
import pandas as pd

__all__ = (
    "one_ns_timedelta",
    "one_s_timedelta"
)

one_ns_timedelta = pd.Timedelta(1)
one_s_timedelta = np.timedelta64(1, 's')
