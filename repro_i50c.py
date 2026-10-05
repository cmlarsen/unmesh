import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
import unmesh_harness.degrade.faults  # noqa: F401
import unmesh
from unmesh import step as step_mod


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


families = ["through_bore", "round_boss", "revolved_dome", "counterbore"]
chains = {
    "flip": [("flipped_facets", 1.0)],
    "flip+iso": [("flipped_facets", 1.0), ("noise_isotropic", 1.0)],
    "iso+flip": [("noise_isotropic", 1.0), ("flipped_facets", 1.0)],
    "flip+iso+normal": [
        ("flipped_facets", 1.0),
        ("noise_isotropic", 1.0),
        ("noise_normal", 1.0),
    ],
}
for family in families:
    for seed in range(6):
        gt = generate(family, seed)
        mesh = labels.tessellate(gt.solid, 0.01, 0.2)
        for cname, chain in chains.items():
            noisy = degrade_core.apply_chain(mesh, chain, 3000 + seed)
            sv = signed_volume(noisy.tris)
            try:
                ir, rep = unmesh.convert(noisy.tris)
                w = [str(x.code) for x in rep.warnings]
                out = step_mod.write(ir, f"repro3_{family}_{seed}_{cname}.step", mesh=noisy.tris)
                vols = [(s.shell, s.kind, s.valid, s.volume) for s in out.shells]
                print(family, seed, cname, "ntris", len(noisy.tris),
                      "signed", round(sv, 3), "valid", out.valid,
                      "fallback", out.fallback, vols, "warn", w,
                      "" if out.valid else f"ISSUES {out.issues} {out.shells[0].issues}",
                      flush=True)
            except Exception as e:
                print(family, seed, cname, "RAISE", type(e).__name__, e, flush=True)
