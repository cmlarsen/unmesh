from __future__ import annotations

from typing import Any

WAIVER_LABEL = "waiver:approved"


def _is_bot(login: str, actor_type: str) -> bool:
    return actor_type == "Bot" or login.endswith("[bot]")


def waiver_approved(
    issue: dict[str, Any], events: list[dict[str, Any]], label: str = WAIVER_LABEL
) -> tuple[bool, str]:
    names = {entry.get("name") for entry in issue.get("labels") or [] if isinstance(entry, dict)}
    if label not in names:
        return False, f"label {label!r} is not on the PR"
    user = issue.get("user") or {}
    author, author_type = str(user.get("login") or ""), str(user.get("type") or "")
    author_is_bot = bool(author) and _is_bot(author, author_type)
    labeled = [
        e
        for e in events
        if e.get("event") == "labeled" and (e.get("label") or {}).get("name") == label
    ]
    if not labeled:
        return False, f"label {label!r} is on the PR but no labeling event was found"
    for event in labeled:
        actor = event.get("actor") or {}
        login, actor_type = str(actor.get("login") or ""), str(actor.get("type") or "")
        if not login:
            continue
        if _is_bot(login, actor_type):
            continue
        if author_is_bot and login == author:
            continue
        return True, f"label {label!r} applied by human @{login}"
    return False, f"label {label!r} was only applied by bots"
