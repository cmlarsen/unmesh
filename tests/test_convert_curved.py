from collections import Counter
from pathlib import Path

import numpy as np

import unmesh

MESHES = Path(__file__).resolve().parents[1] / "fixtures" / "meshes"


def test_corner_fillet_linux_tessellation_recovers_every_face():
    with np.load(MESHES / "corner_fillet-0008-linux.npz") as z:
        tris, face_id = z["tris"], z["face_id"]
    ir, report = unmesh.convert(tris)
    assert report.region_counts == {"cylinder": 12, "plane": 6, "sphere": 8}
    owner = np.full(len(face_id), -1)
    kind = {}
    for region in ir.regions:
        owner[list(region.triangles)] = region.id
        kind[region.id] = type(region.surface).__name__
    dominant = set()
    for face in np.unique(face_id):
        held = Counter(owner[face_id == face].tolist())
        region, count = held.most_common(1)[0]
        dominant.add(region)
        if kind[region] == "Sphere":
            assert count >= 0.95 * (face_id == face).sum(), face
        else:
            assert len(held) == 1, face
    assert len(dominant) == len(np.unique(face_id))
