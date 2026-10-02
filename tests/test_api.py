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
    import unmesh.step as step

    assert unmesh.step is step
    assert "OCP" not in sys.modules
    with pytest.raises(NotImplementedError):
        step.write(None, "out.step")
