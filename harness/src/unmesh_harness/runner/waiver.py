from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

WAIVER_LABEL = "waiver:approved"
DEFAULT_APPROVERS_PATH = ".github/waiver-approvers"


def _is_bot(login: str, actor_type: str) -> bool:
    return actor_type == "Bot" or login.endswith("[bot]")


def parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        current = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text[-1:] in ("Z", "z"):
            text = text[:-1] + "+00:00"
        try:
            current = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current


def parse_waived_cells(body: Any) -> set[tuple[str, str, float]]:
    cells: set[tuple[str, str, float]] = set()
    if not isinstance(body, str):
        return cells
    for line in body.splitlines():
        text = line.strip()
        if not text.lower().startswith("waive:"):
            continue
        parts = text.split(":", 1)[1].split()
        if len(parts) < 3:
            continue
        try:
            cells.add((parts[0], parts[1], float(parts[2])))
        except ValueError:
            continue
    return cells


def _normalize_cells(cells: Any) -> set[tuple[str, str, float]]:
    out: set[tuple[str, str, float]] = set()
    for cell in cells or ():
        if isinstance(cell, dict):
            try:
                out.add((str(cell["part"]), str(cell["operator"]), float(cell["severity"])))
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(cell, (list, tuple)) and len(cell) == 3:
            try:
                out.add((str(cell[0]), str(cell[1]), float(cell[2])))
            except (TypeError, ValueError):
                continue
    return out


def _named_cells(
    comments: list[dict[str, Any]], login: str, head: datetime
) -> set[tuple[str, str, float]]:
    named: set[tuple[str, str, float]] = set()
    for comment in comments or ():
        user = comment.get("user") or {}
        if str(user.get("login") or "").lower() != login.lower():
            continue
        when = parse_time(comment.get("created_at"))
        if when is None or when <= head:
            continue
        named |= parse_waived_cells(comment.get("body"))
    return named


def waiver_approved(
    issue: dict[str, Any],
    events: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    label: str = WAIVER_LABEL,
    approvers: Any = (),
    head_date: Any = None,
    regressed: Any = (),
) -> tuple[bool, str]:
    names = {entry.get("name") for entry in issue.get("labels") or [] if isinstance(entry, dict)}
    if label not in names:
        return False, f"label {label!r} is not on the PR"
    allowed = {str(a).strip().lower() for a in (approvers or ()) if str(a).strip()}
    if not allowed:
        return False, "no waiver approvers configured"
    head = parse_time(head_date)
    if head is None:
        return False, "PR head commit date is unknown"
    user = issue.get("user") or {}
    author, author_type = str(user.get("login") or ""), str(user.get("type") or "")
    author_is_bot = bool(author) and _is_bot(author, author_type)
    targets = _normalize_cells(regressed)
    problems = []
    for event in events or ():
        if event.get("event") != "labeled":
            continue
        if (event.get("label") or {}).get("name") != label:
            continue
        actor = event.get("actor") or {}
        login, actor_type = str(actor.get("login") or ""), str(actor.get("type") or "")
        if not login or _is_bot(login, actor_type) or login.lower() not in allowed:
            continue
        if author_is_bot and login.lower() == author.lower():
            continue
        when = parse_time(event.get("created_at"))
        if when is None or when <= head:
            problems.append(f"@{login} labeled before the current head commit")
            continue
        named = _named_cells(comments, login, head)
        missing = sorted(targets - named, key=str)
        if missing:
            problems.append(f"@{login} did not name waived cell(s) {missing}")
            continue
        return True, f"label {label!r} approved by @{login} covering {len(targets)} cell(s)"
    if problems:
        return False, "; ".join(problems)
    return False, f"label {label!r} has no qualifying approval from an approver"
