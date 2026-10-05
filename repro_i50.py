import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels, oracle
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401  (registers operators)
from unmesh import step as step_mod


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


for family in ["through_bore", "round_boss", "revolved_dome"]:
    for seed in range(6):
        gt = generate(family, seed)
        mesh = labels.tessellate(gt.solid, 0.01, 0.2)
        noisy = degrade_core.apply_chain(
            mesh, [("noise_isotropic", 1.0), ("noise_normal", 1.0)], 1000 + seed
        )
        ir = oracle.build_oracle_ir(mesh)
        rep = step_mod.write(ir, f"repro_i50_{family}_{seed}.step", mesh=noisy.tris)
        sv = signed_volume(noisy.tris)
        vols = [(s.shell, s.kind, s.valid, s.volume) for s in rep.shells]
        print(
            family, seed, "ntris", len(mesh.tris), "signed", round(sv, 3),
            "valid", rep.valid, "fallback", rep.fallback, vols,
            (rep.issues[:1] + rep.shells[0].issues[:1]) if not rep.valid else "",
            flush=True,
        )
