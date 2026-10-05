from __future__ import annotations

import importlib.util


def _load():
    from unmesh_harness.runner.grid import repo_root

    path = repo_root() / "harness" / "scripts" / "sample_standard.py"
    spec = importlib.util.spec_from_file_location("sample_standard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_skipped_cells_do_not_count_as_errors():
    mod = _load()
    assert mod.bad_count({"ok": 390, "skipped": 10}) == 0
    assert mod.bad_count({"ok": 390, "skipped": 10, "timeout": 3, "operator": 2}) == 5
    assert mod.bad_count({"ok": 400}) == 0
