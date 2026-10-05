import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels, oracle
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
from unmesh import step as step_mod


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


gt = generate("through_bore", 3)
mesh = labels.tessellate(gt.solid, 0.01, 0.2)
noisy = degrade_core.apply_chain(
    mesh, [("noise_isotropic", 1.0), ("noise_normal", 1.0)], 77
).tris.copy()
ir = oracle.build_oracle_ir(mesh)
print("ntris", len(noisy), "clean signed", round(signed_volume(mesh.tris), 3), flush=True)

rng = np.random.default_rng(0)
for ndeg in [0, 1, 5, 20, 60]:
    t = noisy.copy()
    picks = rng.choice(len(t), size=ndeg, replace=False)
    for p in picks:
        t[p, 2] = t[p, 0]
    sv = signed_volume(t)
    rep = step_mod.write(ir, f"repro5_{ndeg}.step", mesh=t)
    vols = [(s.shell, s.kind, s.valid, s.volume) for s in rep.shells]
    print(f"ndeg {ndeg} signed {round(sv,3)} valid {rep.valid} fallback {rep.fallback}",
          vols, rep.issues if not rep.valid else "", flush=True)
