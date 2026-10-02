import subprocess
import sys

import pytest

import unmesh
from unmesh import ConvertOptions, Result


def test_convert_is_a_stub():
    with pytest.raises(NotImplementedError):
        unmesh.convert("part.stl", ConvertOptions())


def test_result_unpacks():
    assert Result._fields == ("ir", "report")


def test_step_namespace_is_lazy_and_ocp_free():
    code = (
        "import sys, unmesh, unmesh.step as step\n"
        "assert unmesh.step is step\n"
        "assert 'OCP' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
