import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
import unmesh_harness.degrade.defects  # noqa: F401
import unmesh
from unmesh._writer import occ
from OCP.TopoDS import TopoDS


def rv(sh):
    return round(occ.volume_of(sh), 3)


gt = generate("revolved_cone", 1)
mesh = labels.tessellate(gt.solid, 0.01, 0.2)
deg = degrade_core.apply_chain(mesh, [("slivers", 1.0), ("noise_isotropic", 1.0)], 6001)
ir, rep = unmesh.convert(deg.tris)
ids = sorted(x for s in ir.shells for r in s.regions for x in ir.regions[r].triangles)
t = np.ascontiguousarray(deg.tris[ids])

shell = occ.build_triangle_shell(t)
solid = occ.make_solid(shell, [])
print("A solid:", rv(solid), flush=True)
occ.write_step(solid, "mA.step")
print("A roundtrip:", rv(occ.read_step("mA.step")), flush=True)

fixed, _ = occ.fix_shape(solid)
print("A fixed:", rv(fixed), flush=True)
occ.write_step(fixed, "mAf.step")
print("Af roundtrip:", rv(occ.read_step("mAf.step")), flush=True)

rev = TopoDS.Solid_s(fixed.Reversed())
print("A fixed+rev:", rv(rev), flush=True)
occ.write_step(rev, "mAr.step")
print("Ar roundtrip:", rv(occ.read_step("mAr.step")), flush=True)

tneg = np.ascontiguousarray(t[:, ::-1])
shell2 = occ.build_triangle_shell(tneg)
print("B sewn-from-flipped:", rv(shell2), flush=True)
solid2 = occ.make_solid(shell2, [])
print("B solid:", rv(solid2), flush=True)
fixed2, _ = occ.fix_shape(solid2)
print("B fixed:", rv(fixed2), flush=True)
occ.write_step(fixed2, "mB.step")
print("B roundtrip:", rv(occ.read_step("mB.step")), flush=True)
