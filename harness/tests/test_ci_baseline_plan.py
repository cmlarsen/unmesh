from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def load_module(name="ci_baseline_plan"):
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader(name, "scripts/ci-baseline-plan")
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


REPO = "o/r"
BASE_SHA = "f" * 40


def run(run_id, event="push", branch="main", sha=BASE_SHA, repo=REPO, fork=False, module=None):
    return module.BaselineRun(
        id=run_id, event=event, head_branch=branch, head_sha=sha, head_repo=repo, fork=fork
    )


@pytest.fixture
def module():
    return load_module()


def decide(module, **kw):
    args = {
        "repo": REPO,
        "base_sha": BASE_SHA,
        "base": None,
        "latest": None,
        "compare_on_main": True,
    }
    args.update(kw)
    return module.decide_baseline(**args)


def test_base_sha_hit_is_used_without_fallback(module):
    d = decide(module, base=run(11, module=module), latest=run(12, module=module))
    assert (d.action, d.run_id, d.fell_back) == ("use", 11, False)


def test_base_sha_miss_falls_back_to_latest_with_warning(module):
    d = decide(module, latest=run(12, sha="e" * 40, module=module))
    assert d.action == "use" and d.run_id == 12 and d.fell_back
    assert BASE_SHA in d.message and "12" in d.message


def test_base_sha_mismatch_falls_back(module):
    d = decide(
        module,
        base=run(11, sha="e" * 40, module=module),
        latest=run(12, module=module),
    )
    assert (d.action, d.run_id, d.fell_back) == ("use", 12, True)


def test_no_runs_at_all_skips(module):
    d = decide(module)
    assert d.action == "skip" and d.run_id is None


def test_bootstrap_skips_when_compare_is_not_on_main(module):
    d = decide(
        module,
        base=run(11, module=module),
        latest=run(12, module=module),
        compare_on_main=False,
    )
    assert d.action == "skip" and "not on main yet" in d.message


def test_fork_run_is_rejected(module):
    d = decide(module, latest=run(12, repo="mallory/r", module=module))
    assert d.action == "fail"
    d = decide(module, latest=run(12, fork=True, module=module))
    assert d.action == "fail"


def test_non_push_run_is_rejected(module):
    d = decide(module, latest=run(12, event="pull_request", module=module))
    assert d.action == "fail"


def test_wrong_branch_run_is_rejected(module):
    d = decide(module, latest=run(12, branch="feature", module=module))
    assert d.action == "fail"


def test_untrusted_base_falls_back_to_trusted_latest(module):
    d = decide(
        module,
        base=run(11, event="pull_request", module=module),
        latest=run(12, module=module),
    )
    assert (d.action, d.run_id) == ("use", 12)


def test_validate_baseline_rejects_empty_or_missing(tmp_path, module):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="missing|unreadable|empty"):
        module.validate_baseline(missing)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        module.validate_baseline(empty)
    blank = tmp_path / "blank.jsonl"
    blank.write_text("  \n\n")
    with pytest.raises(ValueError, match="empty"):
        module.validate_baseline(blank)
    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text("not json\n")
    with pytest.raises(ValueError, match="no JSON records"):
        module.validate_baseline(garbage)
    good = tmp_path / "good.jsonl"
    good.write_text('{"part": "box", "status": "ok"}\n')
    module.validate_baseline(good)


def test_main_entrypoint_bootstraps_without_touching_gh(tmp_path, monkeypatch, capsys):
    module = load_module("ci_baseline_bootstrap")

    def no_gh(*args, **kwargs):
        raise AssertionError("gh must not be called during bootstrap")

    monkeypatch.setattr(module.subprocess, "run", no_gh)
    out = tmp_path / "baseline.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci-baseline-plan",
            "--repo",
            REPO,
            "--base-sha",
            BASE_SHA,
            "--artifact",
            "smoke-results-x",
            "--out",
            str(out),
            "--main-tree",
            str(tmp_path / "empty-main"),
        ],
    )
    assert module.main() == 0
    assert "not on main yet" in capsys.readouterr().out
    assert not out.exists()


def test_main_entrypoint_fails_when_download_fails(tmp_path, monkeypatch):
    module = load_module("ci_baseline_dlfail")
    main_tree = tmp_path / "main"
    (main_tree / "harness/src/unmesh_harness/runner").mkdir(parents=True)
    (main_tree / "harness/src/unmesh_harness/runner/compare.py").write_text("x = 1\n")
    monkeypatch.setattr(module, "list_run_id", lambda repo, workflow, commit: 11)
    monkeypatch.setattr(module, "run_metadata", lambda repo, run_id: run(11, module=module))

    def boom(repo, run_id, artifact, dest, *args, **kwargs):
        raise ValueError("no such artifact")

    monkeypatch.setattr(module, "download_artifact", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci-baseline-plan",
            "--repo",
            REPO,
            "--base-sha",
            BASE_SHA,
            "--artifact",
            "smoke-results-x",
            "--out",
            str(tmp_path / "baseline.jsonl"),
            "--main-tree",
            str(main_tree),
        ],
    )
    assert module.main() == 1


def test_pick_artifact_file_selects_grid(module):
    names = ["smoke.jsonl", "smoke_curved.jsonl"]
    assert module.pick_artifact_file(names, "smoke") == "smoke.jsonl"
    assert module.pick_artifact_file(names, "smoke_curved") == "smoke_curved.jsonl"
    assert module.pick_artifact_file(["smoke.jsonl"], "smoke_curved") is None
    assert module.pick_artifact_file([], "smoke") is None


def test_pick_artifact_file_without_select_keeps_first_jsonl(module):
    assert module.pick_artifact_file(["smoke.jsonl", "smoke_curved.jsonl"], None) == "smoke.jsonl"
    assert module.pick_artifact_file([], None) is None
    assert module.pick_artifact_file(["notes.txt"], None) is None


def _argv(tmp_path, **kw):
    argv = [
        "ci-baseline-plan",
        "--repo",
        REPO,
        "--base-sha",
        BASE_SHA,
        "--artifact",
        "smoke-results-x",
        "--out",
        str(tmp_path / "baseline.jsonl"),
        "--main-tree",
        str(tmp_path / "main"),
    ]
    for flag, value in kw.items():
        argv.append(f"--{flag.replace('_', '-')}")
        if value is not True:
            argv.append(str(value))
    return argv


def _main_tree_with_compare(tmp_path):
    main_tree = tmp_path / "main"
    (main_tree / "harness/src/unmesh_harness/runner").mkdir(parents=True)
    (main_tree / "harness/src/unmesh_harness/runner/compare.py").write_text("x = 1\n")
    return main_tree


def test_allow_missing_skips_new_grid_only(tmp_path, monkeypatch, capsys):
    module = load_module("ci_baseline_allow_missing")
    _main_tree_with_compare(tmp_path)
    monkeypatch.setattr(module, "list_run_id", lambda repo, workflow, commit: 11)
    monkeypatch.setattr(module, "run_metadata", lambda repo, run_id: run(11, module=module))

    def missing(repo, run_id, artifact, dest, select=None):
        raise module.MissingGridFile(f"artifact holds no {select}.jsonl")

    monkeypatch.setattr(module, "download_artifact", missing)
    calls = []

    def absent(repo, path, sha):
        calls.append((repo, path, sha))
        return False

    monkeypatch.setattr(module, "grid_exists_at_sha", absent)
    out = tmp_path / "baseline.jsonl"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, select="smoke_curved", allow_missing=True))
    assert module.main() == 0
    assert "predates the grid" in capsys.readouterr().out
    assert calls == [(REPO, "harness/grids/smoke_curved.json", BASE_SHA)]
    assert not out.exists()


def test_bootstrap_for_missing_grid_truth_table(module):
    assert (
        module.bootstrap_for_missing_grid(allow_missing=True, grid_present_at_baseline=False)
        is True
    )
    assert (
        module.bootstrap_for_missing_grid(allow_missing=True, grid_present_at_baseline=True)
        is False
    )
    assert (
        module.bootstrap_for_missing_grid(allow_missing=False, grid_present_at_baseline=False)
        is False
    )
    assert (
        module.bootstrap_for_missing_grid(allow_missing=False, grid_present_at_baseline=True)
        is False
    )


def test_grid_path(module):
    assert module.grid_path("smoke_curved") == "harness/grids/smoke_curved.json"


def test_grid_exists_at_sha_maps_404_to_absent(module, monkeypatch):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )
    assert module.grid_exists_at_sha(REPO, "harness/grids/smoke_curved.json", BASE_SHA) is True

    def not_found(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="gh: Not Found (HTTP 404)")

    monkeypatch.setattr(module.subprocess, "run", not_found)
    assert module.grid_exists_at_sha(REPO, "harness/grids/smoke_curved.json", BASE_SHA) is False

    def bad_ref(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, args[0], stderr=f"gh: No commit found for the ref {BASE_SHA} (HTTP 404)"
        )

    monkeypatch.setattr(module.subprocess, "run", bad_ref)
    with pytest.raises(subprocess.CalledProcessError):
        module.grid_exists_at_sha(REPO, "harness/grids/smoke_curved.json", BASE_SHA)

    def rate_limited(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, args[0], stderr="gh: API rate limit exceeded (HTTP 403)"
        )

    monkeypatch.setattr(module.subprocess, "run", rate_limited)
    with pytest.raises(subprocess.CalledProcessError):
        module.grid_exists_at_sha(REPO, "harness/grids/smoke_curved.json", BASE_SHA)


def test_missing_grid_file_fails_when_grid_exists_at_baseline_sha(tmp_path, monkeypatch, capsys):
    module = load_module("ci_baseline_grid_present")
    _main_tree_with_compare(tmp_path)
    monkeypatch.setattr(module, "list_run_id", lambda repo, workflow, commit: 11)
    monkeypatch.setattr(module, "run_metadata", lambda repo, run_id: run(11, module=module))

    def missing(repo, run_id, artifact, dest, select=None):
        raise module.MissingGridFile(f"artifact holds no {select}.jsonl")

    monkeypatch.setattr(module, "download_artifact", missing)
    monkeypatch.setattr(module, "grid_exists_at_sha", lambda repo, path, sha: True)
    out = tmp_path / "baseline.jsonl"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, select="smoke_curved", allow_missing=True))
    assert module.main() == 1
    assert "exists at baseline" in capsys.readouterr().out
    assert not out.exists()


def test_grid_check_failure_fails_closed(tmp_path, monkeypatch, capsys):
    module = load_module("ci_baseline_gridcheck_fail")
    _main_tree_with_compare(tmp_path)
    monkeypatch.setattr(module, "list_run_id", lambda repo, workflow, commit: 11)
    monkeypatch.setattr(module, "run_metadata", lambda repo, run_id: run(11, module=module))

    def missing(repo, run_id, artifact, dest, select=None):
        raise module.MissingGridFile(f"artifact holds no {select}.jsonl")

    monkeypatch.setattr(module, "download_artifact", missing)

    def boom(repo, path, sha):
        raise subprocess.CalledProcessError(
            1, ["gh", "api"], stderr="gh: API rate limit exceeded (HTTP 403)"
        )

    monkeypatch.setattr(module, "grid_exists_at_sha", boom)
    out = tmp_path / "baseline.jsonl"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, select="smoke_curved", allow_missing=True))
    assert module.main() == 1
    assert "cannot verify" in capsys.readouterr().out
    assert not out.exists()


def test_present_grid_file_downloads_normally(tmp_path, monkeypatch, capsys):
    module = load_module("ci_baseline_present_file")
    _main_tree_with_compare(tmp_path)
    monkeypatch.setattr(module, "list_run_id", lambda repo, workflow, commit: 11)
    monkeypatch.setattr(module, "run_metadata", lambda repo, run_id: run(11, module=module))
    out = tmp_path / "baseline.jsonl"

    def present(repo, run_id, artifact, dest, select=None):
        dest.write_text('{"part": "box", "status": "ok"}\n')
        return True

    monkeypatch.setattr(module, "download_artifact", present)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, select="smoke_curved", allow_missing=True))
    assert module.main() == 0
    assert "baseline ready" in capsys.readouterr().out
    assert out.exists()


def test_missing_grid_file_fails_without_allow_missing(tmp_path, monkeypatch):
    module = load_module("ci_baseline_no_allow_missing")
    _main_tree_with_compare(tmp_path)
    monkeypatch.setattr(module, "list_run_id", lambda repo, workflow, commit: 11)
    monkeypatch.setattr(module, "run_metadata", lambda repo, run_id: run(11, module=module))

    def missing(repo, run_id, artifact, dest, select=None):
        raise module.MissingGridFile(f"artifact holds no {select}.jsonl")

    monkeypatch.setattr(module, "download_artifact", missing)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, select="smoke_curved"))
    assert module.main() == 1


def test_script_is_executable():
    assert os.access("scripts/ci-baseline-plan", os.X_OK)
    assert Path("scripts/ci-baseline-plan").read_text().startswith("#!")
