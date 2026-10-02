from os import PathLike
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

def core_version() -> str: ...
def read_stl(path: str | PathLike[str]) -> NDArray[np.float64]: ...
def write_stl(path: str | PathLike[str], tris: ArrayLike) -> None: ...
def weld(
    tris: ArrayLike, tolerance: float
) -> tuple[NDArray[np.float64], NDArray[np.uint32], NDArray[np.uint32], dict[str, Any]]: ...
def convert_soup(
    tris: ArrayLike,
    linear_tolerance: float | None,
    angular_snap_deg: float,
    tangent_threshold_deg: float,
    vertex_merge: float,
) -> tuple[str, dict[str, Any]]: ...
def convert_indexed(
    vertices: ArrayLike,
    faces: ArrayLike,
    linear_tolerance: float | None,
    angular_snap_deg: float,
    tangent_threshold_deg: float,
    vertex_merge: float,
) -> tuple[str, dict[str, Any]]: ...
