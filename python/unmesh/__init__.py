from unmesh._core import core_version, read_stl, weld, write_stl
from unmesh.api import (
    ConvertOptions,
    Report,
    Result,
    convert,
    convert_from_labels,
    read_mesh,
)
from unmesh.ir import Ir

__version__ = core_version()

__all__ = [
    "ConvertOptions",
    "Ir",
    "Report",
    "Result",
    "__version__",
    "convert",
    "convert_from_labels",
    "convert_to_step",
    "read_mesh",
    "read_stl",
    "weld",
    "write_stl",
]


def __getattr__(name):
    if name == "step":
        import importlib

        return importlib.import_module("unmesh.step")
    if name == "convert_to_step":
        from unmesh.pipeline import convert_to_step

        return convert_to_step
    raise AttributeError(f"module 'unmesh' has no attribute {name!r}")
