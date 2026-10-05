from __future__ import annotations

from typing import Any

STRATA_LIN = 0.01

FACE_BUCKETS = ((10, "0-10"), (50, "11-50"), (300, "51-300"), (float("inf"), "301+"))
FEATURE_BUCKETS = ((1, "<1"), (10, "1-10"), (100, "10-100"), (float("inf"), "100+"))


def face_bucket(face_count: int) -> str:
    for limit, label in FACE_BUCKETS:
        if face_count <= limit:
            return label
    raise AssertionError("unreachable")


def feature_bucket(ratio: float) -> str:
    for limit, label in FEATURE_BUCKETS:
        if ratio < limit:
            return label
    raise AssertionError("unreachable")


def min_edge_length(shape) -> float:
    edges = [e.length for e in shape.edges()]
    live = [e for e in edges if e > 1e-7]
    if not live:
        raise RuntimeError("shape has no non-degenerate edges")
    return min(live)


def compute_strata(shape, category: str) -> dict[str, Any]:
    faces = len(shape.faces())
    smallest = min_edge_length(shape)
    ratio = round(smallest / STRATA_LIN, 3)
    return {
        "category": category,
        "face_count": faces,
        "face_bucket": face_bucket(faces),
        "min_feature_ratio": ratio,
        "feature_bucket": feature_bucket(ratio),
    }
