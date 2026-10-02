import subprocess
import sys

import unmesh


def test_version():
    assert unmesh.__version__ == "0.1.0"


def test_core_does_not_import_ocp():
    code = "import sys, unmesh; sys.exit('OCP' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
