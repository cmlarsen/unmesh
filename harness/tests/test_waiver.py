from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from unmesh_harness.runner.waiver import parse_waived_cells, waiver_approved

LABEL = "waiver:approved"
HEAD = "2026-10-02T12:00:00Z"
BEFORE = "2026-10-02T11:00:00Z"
AFTER = "2026-10-02T13:00:00Z"

CELL = ("box", "identity", 0.0)


def issue(author="cmlarsen", author_type="User", labels=(LABEL,)):
    return {
        "user": {"login": author, "type": author_type},
        "labels": [{"name": name} for name in labels],
    }


def labeled(actor, actor_type="User", created_at=AFTER, label=LABEL):
    return {
        "event": "labeled",
        "label": {"name": label},
        "actor": {"login": actor, "type": actor_type},
        "created_at": created_at,
    }


def comment(login, body, created_at=AFTER):
    return {"user": {"login": login}, "body": body, "created_at": created_at}


def waive_line(cell=CELL):
    return f"waive: {cell[0]} {cell[1]} {cell[2]:g}"


def decide(events, comments, approvers=("cmlarsen",), head_date=HEAD, regressed=(CELL,), **kw):
    return waiver_approved(
        kw.pop("issue", issue()),
        events,
        comments,
        approvers=approvers,
        head_date=head_date,
        regressed=regressed,
        **kw,
    )


def test_listed_approver_waives_named_cells():
    ok, reason = decide([labeled("cmlarsen")], [comment("cmlarsen", waive_line())])
    assert ok and "cmlarsen" in reason


def test_actor_outside_allowlist_is_rejected():
    ok, _ = decide([labeled("mallory")], [comment("mallory", waive_line())])
    assert not ok


def test_empty_allowlist_is_rejected():
    ok, reason = decide([labeled("cmlarsen")], [comment("cmlarsen", waive_line())], approvers=())
    assert not ok and "approvers" in reason


def test_stale_label_event_is_rejected():
    ok, _ = decide([labeled("cmlarsen", created_at=BEFORE)], [comment("cmlarsen", waive_line())])
    assert not ok


def test_removed_label_is_not_a_waiver():
    ok, _ = decide(
        [labeled("cmlarsen")], [comment("cmlarsen", waive_line())], issue=issue(labels=())
    )
    assert not ok


def test_missing_comment_is_not_a_waiver():
    ok, _ = decide([labeled("cmlarsen")], [])
    assert not ok


def test_stale_comment_is_not_a_waiver():
    ok, _ = decide([labeled("cmlarsen")], [comment("cmlarsen", waive_line(), created_at=BEFORE)])
    assert not ok


def test_comment_by_another_actor_is_not_a_waiver():
    ok, _ = decide([labeled("cmlarsen")], [comment("mallory", waive_line())])
    assert not ok


def test_unnamed_regressed_cell_is_not_waived():
    other = ("cyl", "identity", 0.0)
    ok, reason = decide([labeled("cmlarsen")], [comment("cmlarsen", waive_line(other))])
    assert not ok and "box" in reason


def test_extra_named_cells_are_allowed():
    body = waive_line() + "\n" + waive_line(("cyl", "identity", 0.0))
    ok, _ = decide([labeled("cmlarsen")], [comment("cmlarsen", body)])
    assert ok


def test_cells_across_comments_combine():
    comments = [
        comment("cmlarsen", waive_line()),
        comment("cmlarsen", waive_line(("cyl", "x", 1.0))),
    ]
    ok, _ = decide([labeled("cmlarsen")], comments, regressed=(CELL, ("cyl", "x", 1.0)))
    assert ok


def test_bot_actor_is_rejected_even_when_listed():
    ok, _ = decide(
        [labeled("cmlarsen[bot]", "Bot")],
        [comment("cmlarsen[bot]", waive_line())],
        approvers=("cmlarsen[bot]",),
    )
    assert not ok


def test_bot_author_cannot_waive_its_own_pr():
    bot_pr = issue(author="greptile[bot]", author_type="Bot")
    ok, _ = decide(
        [labeled("greptile[bot]", "User")],
        [comment("greptile[bot]", waive_line())],
        approvers=("greptile[bot]",),
        issue=bot_pr,
    )
    assert not ok


def test_human_can_waive_a_bot_pr():
    bot_pr = issue(author="greptile[bot]", author_type="Bot")
    ok, _ = decide([labeled("cmlarsen")], [comment("cmlarsen", waive_line())], issue=bot_pr)
    assert ok


def test_unknown_head_date_is_not_a_waiver():
    ok, _ = decide([labeled("cmlarsen")], [comment("cmlarsen", waive_line())], head_date=None)
    assert not ok


def test_parse_waived_cells_ignores_garbage():
    body = "looks good\nwaive: box identity 0\nwaive: broken line\nwaive: a b notanumber\n"
    assert parse_waived_cells(body) == {CELL}
    assert parse_waived_cells(None) == set()


def load_script_module(name):
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader(name, "scripts/check-waiver")
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_gh_api_list_paginates(monkeypatch):
    module = load_script_module("check_waiver_pages")

    def fake_run(cmd, **kwargs):
        import re

        assert cmd[:2] == ["gh", "api"]
        page = int(re.search(r"[?&]page=(\d+)", cmd[2]).group(1))
        payload = [{"n": i} for i in range(100 if page == 1 else 3)]
        return types.SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert len(module.gh_api_list("repos/o/r/issues/1/events")) == 103


def test_read_approvers_skips_blanks_and_comments(monkeypatch):
    module = load_script_module("check_waiver_approvers")
    body = "# humans who may waive\n\ncmlarsen\n  \n"
    payload = {"content": base64.b64encode(body.encode()).decode(), "encoding": "base64"}
    monkeypatch.setattr(module, "gh_api", lambda path: payload)
    assert module.read_approvers("o/r", ".github/waiver-approvers", "main") == ["cmlarsen"]


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


def test_check_waiver_end_to_end_with_stubbed_api(tmp_path, monkeypatch):
    module = load_script_module("check_waiver_e2e")

    def cell(seed, f1):
        return {
            "part": "box",
            "operator": "identity",
            "severity": 0.0,
            "seed": seed,
            "converter": "unmesh",
            "git_sha": "s",
            "grid_hash": "g",
            "plugin_hash": "",
            "status": "ok",
            "f1": f1,
            "valid": True,
            "under_report": False,
            "fallback": False,
            "dev_input_max": 0.0,
            "dev_truth_max": 0.0,
        }

    run_a = tmp_path / "a.jsonl"
    run_b = tmp_path / "b.jsonl"
    run_a.write_text("\n".join(json.dumps(cell(i, 1.0)) for i in range(3)) + "\n")
    run_b.write_text("\n".join(json.dumps(cell(i, 0.9)) for i in range(3)) + "\n")
    state = {"body": waive_line()}

    def fake_gh_api(path):
        if path.endswith("/events?per_page=100&page=1"):
            return [labeled("cmlarsen")]
        if path.endswith("/comments?per_page=100&page=1"):
            return [comment("cmlarsen", state["body"])]
        if "/pulls/" in path:
            return {"head": {"sha": "abc"}}
        if "/commits/" in path:
            return {"commit": {"committer": {"date": HEAD}}}
        if "/contents/" in path:
            content = base64.b64encode(b"cmlarsen\n").decode()
            return {"content": content, "encoding": "base64"}
        return issue()

    monkeypatch.setattr(module, "gh_api", fake_gh_api)
    argv = [
        "check-waiver",
        "--repo",
        "o/r",
        "--pr",
        "7",
        "--run-a",
        str(run_a),
        "--run-b",
        str(run_b),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    state["body"] = waive_line(("other", "identity", 0.0))
    assert module.main() == 1
