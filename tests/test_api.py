import subprocess
import sys

import numpy as np
import pytest

import unmesh
from unmesh import ConvertOptions, Result


def test_convert_rejects_bad_input():
    with pytest.raises(ValueError):
        unmesh.convert(np.zeros((0, 3, 3)), ConvertOptions())
    with pytest.raises(ValueError):
        unmesh.convert(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        unmesh.convert((np.zeros((3, 3)), np.array([[0, 1, 5]])))
    with pytest.raises(OSError):
        unmesh.convert("does-not-exist.stl")


def test_result_unpacks():
    assert Result._fields == ("ir", "report")


def test_step_namespace_is_lazy_and_ocp_free():
    code = (
        "import sys, unmesh, unmesh.step as step\n"
        "assert unmesh.step is step\n"
        "assert 'OCP' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
