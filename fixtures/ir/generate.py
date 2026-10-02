import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from build import FIXTURES  # noqa: E402

out = pathlib.Path(__file__).parent
for name, make in FIXTURES.items():
    (out / f"{name}.json").write_text(make().dumps() + "\n")
