from __future__ import annotations

import importlib.util
import os
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

    def boom(repo, run_id, artifact, dest):
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


def test_script_is_executable():
    assert os.access("scripts/ci-baseline-plan", os.X_OK)
    assert Path("scripts/ci-baseline-plan").read_text().startswith("#!")
