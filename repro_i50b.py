import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
import unmesh
from unmesh import step as step_mod


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


chains = {
    "iso": [("noise_isotropic", 1.0)],
    "normal": [("noise_normal", 1.0)],
    "iso+normal": [("noise_isotropic", 1.0), ("noise_normal", 1.0)],
}
families = ["through_bore", "round_boss", "revolved_dome", "counterbore", "revolved_cone"]
for family in families:
    for seed in range(10):
        gt = generate(family, seed)
        mesh = labels.tessellate(gt.solid, 0.1, 0.5)
        for cname, chain in chains.items():
            noisy = degrade_core.apply_chain(mesh, chain, 2000 + seed)
            sv = signed_volume(noisy.tris)
            try:
                ir, _ = unmesh.convert(noisy.tris)
                rep = step_mod.write(ir, f"repro2_{family}_{seed}_{cname}.step", mesh=noisy.tris)
                vols = [(s.shell, s.kind, s.valid, s.volume) for s in rep.shells]
                flag = "" if (rep.valid and all(
                    v is None or abs(v - sv) <= 1e-9 * abs(sv) for _, _, _, v in vols) or
                    all(v is None for _, _, _, v in vols)) else "  <<< CHECK"
                print(family, seed, cname, "ntris", len(mesh.tris),
                      "signed", round(sv, 3), "valid", rep.valid,
                      "fallback", rep.fallback, vols, flag, flush=True)
            except Exception as e:
                print(family, seed, cname, "CONVERT-RAISE", type(e).__name__, e, flush=True)
