import subprocess
import sys

import unmesh


def test_version():
    assert unmesh.__version__ == "0.1.0"


def test_core_does_not_import_ocp():
    code = "import sys, unmesh; sys.exit(7 if 'OCP' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 7, "unmesh imported OCP"
    assert result.returncode == 0, result.stderr
