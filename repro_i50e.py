import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
from unmesh._writer import occ


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


gt = generate("through_bore", 3)
mesh = labels.tessellate(gt.solid, 0.01, 0.2)
noisy = degrade_core.apply_chain(mesh, [("noise_isotropic", 1.0), ("noise_normal", 1.0)], 77)
for name, t in [
    ("clean", mesh.tris),
    ("noisy", noisy.tris),
    ("noisy-inverted", noisy.tris[:, ::-1]),
]:
    print(name, "numpy", signed_volume(t), flush=True)
    shell = occ.build_triangle_shell(np.ascontiguousarray(t))
    print("  after sewing+orient:", occ.volume_of(shell), flush=True)
    solid = occ.make_solid(shell, [])
    print("  after make_solid:", occ.volume_of(solid), flush=True)
    fixed, _ = occ.fix_shape(solid)
    print("  after fix_shape:", occ.volume_of(fixed), "valid:", occ.is_valid(fixed), flush=True)
