import pytest

from unmesh_harness.corpus import load_manifest, select
from unmesh_harness.labels import DEFLECTION_SETTINGS


def is_planar(entry) -> bool:
    return entry["strata"].get("category", "planar") == "planar"


def smoke_entries(curved_slow: bool = False):
    marks = (pytest.mark.slow,)
    return [
        pytest.param(e, id=e["id"], marks=() if is_planar(e) or not curved_slow else marks)
        for e in select(load_manifest(), "smoke")
    ]


def smoke_by_deflection():
    finest = min(DEFLECTION_SETTINGS)
    return [
        pytest.param(
            e,
            lin,
            ang,
            id=f"{e['id']}-{lin}",
            marks=(pytest.mark.slow,) if not is_planar(e) and (lin, ang) == finest else (),
        )
        for e in select(load_manifest(), "smoke")
        for lin, ang in DEFLECTION_SETTINGS
    ]
