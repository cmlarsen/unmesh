import sys

import unmesh


def test_version():
    assert unmesh.__version__ == "0.1.0"


def test_core_does_not_import_ocp():
    assert "OCP" not in sys.modules
