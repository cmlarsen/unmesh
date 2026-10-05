import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
import unmesh_harness.degrade.defects  # noqa: F401
import unmesh
from unmesh._writer import occ


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


gt = generate("revolved_cone", 1)
mesh = labels.tessellate(gt.solid, 0.01, 0.2)
deg = degrade_core.apply_chain(mesh, [("slivers", 1.0), ("noise_isotropic", 1.0)], 6001)
t = deg.tris
print("ntris", len(t), "signed", signed_volume(t), flush=True)
a, b, c = t[:, 0], t[:, 1], t[:, 2]
contrib = np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0
print("contrib>0:", int((contrib > 0).sum()), "contrib<0:", int((contrib < 0).sum()),
      "sum+:", round(contrib[contrib > 0].sum(), 3),
      "sum-:", round(contrib[contrib < 0].sum(), 3), flush=True)
ir, rep = unmesh.convert(t)
print("shells:", [(s.closed, s.role, s.parent, len(s.regions)) for s in ir.shells],
      "warn:", [str(x.code) for x in rep.warnings], flush=True)
nref = sum(len(ir.regions[r].triangles) for s in ir.shells for r in s.regions)
print("triangles referenced by IR:", nref, "of", len(t), flush=True)
ids = sorted(x for s in ir.shells for r in s.regions for x in ir.regions[r].triangles)
print("referenced signed vol:", signed_volume(t[ids]), flush=True)
shell = occ.build_triangle_shell(np.ascontiguousarray(t[ids]))
print("sewn volume:", occ.volume_of(shell), flush=True)
solid = occ.make_solid(shell, [])
print("solid volume:", occ.volume_of(solid), "valid:", occ.is_valid(solid), flush=True)
