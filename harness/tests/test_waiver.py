from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from unmesh_harness.runner.waiver import waiver_approved

LABEL = "waiver:approved"


def issue(author="cmlarsen", author_type="User", labels=(LABEL,)):
    return {
        "user": {"login": author, "type": author_type},
        "labels": [{"name": name} for name in labels],
    }


def labeled(actor, actor_type="User"):
    return {
        "event": "labeled",
        "label": {"name": LABEL},
        "actor": {"login": actor, "type": actor_type},
    }


def test_human_label_waives():
    ok, reason = waiver_approved(issue(), [labeled("cmlarsen")])
    assert ok and "cmlarsen" in reason


def test_missing_label_is_not_a_waiver():
    ok, _ = waiver_approved(issue(labels=()), [labeled("cmlarsen")])
    assert not ok


def test_label_without_event_is_not_a_waiver():
    ok, _ = waiver_approved(issue(), [{"event": "unlabeled"}])
    assert not ok


def test_bot_actor_is_rejected():
    for actor in ("github-actions[bot]", "dependabot[bot]", "claude[bot]"):
        ok, _ = waiver_approved(issue(), [labeled(actor, "Bot")])
        assert not ok, actor


def test_bot_suffixed_login_is_rejected_even_as_user_type():
    ok, _ = waiver_approved(issue(), [labeled("helper[bot]", "User")])
    assert not ok


def test_bot_author_cannot_waive_its_own_pr():
    bot_pr = issue(author="greptile[bot]", author_type="Bot")
    ok, _ = waiver_approved(bot_pr, [labeled("greptile[bot]", "Bot")])
    assert not ok
    ok, _ = waiver_approved(bot_pr, [labeled("greptile[bot]", "User")])
    assert not ok


def test_human_can_waive_a_bot_pr():
    ok, _ = waiver_approved(issue(author="greptile[bot]", author_type="Bot"), [labeled("cmlarsen")])
    assert ok


def test_human_event_wins_over_bot_event():
    ok, _ = waiver_approved(issue(), [labeled("agent[bot]", "Bot"), labeled("cmlarsen")])
    assert ok


def test_other_labels_are_ignored():
    other = {"event": "labeled", "label": {"name": "in-progress"}, "actor": {"login": "cmlarsen"}}
    ok, _ = waiver_approved(issue(), [other])
    assert not ok


def load_script_module(name):
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader(name, "scripts/check-waiver")
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_check_waiver_script_runs_standalone(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check-waiver", "--help"])
    proc = subprocess.run(
        [sys.executable, "scripts/check-waiver", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_REF": "refs/pull/32/merge"},
    )
    assert proc.returncode == 0 and "--pr" in proc.stdout


def test_check_waiver_script_needs_repo_and_pr(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("PR_NUMBER", raising=False)
    monkeypatch.delenv("GITHUB_PR_NUMBER", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setattr(sys, "argv", ["check-waiver"])
    module = load_script_module("check_waiver")
    assert module.pr_number_from_env() is None
    monkeypatch.setenv("GITHUB_REF", "refs/pull/32/merge")
    assert module.pr_number_from_env() == 32
    monkeypatch.setenv("PR_NUMBER", "77")
    assert module.pr_number_from_env() == 77


@pytest.mark.parametrize("ref", ["refs/pull/32/merge", "refs/heads/main", ""])
def test_pr_ref_parsing_never_raises(ref, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check-waiver"])
    monkeypatch.setenv("GITHUB_REF", ref)
    monkeypatch.delenv("PR_NUMBER", raising=False)
    monkeypatch.delenv("GITHUB_PR_NUMBER", raising=False)
    module = load_script_module("check_waiver_ref")
    assert module.pr_number_from_env() == (32 if ref.startswith("refs/pull/") else None)


def test_check_waiver_is_executable():
    assert os.access("scripts/check-waiver", os.X_OK)
    assert Path("scripts/check-waiver").read_text().startswith("#!")
