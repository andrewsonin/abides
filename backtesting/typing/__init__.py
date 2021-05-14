from os import PathLike
from pathlib import Path
from typing import Union, Any

__all__ = (
    "FileName",
    "Event",
    "NoneType"
)

NoneType = type(None)
FileName = Union[str, PathLike, Path]
Event = Any
