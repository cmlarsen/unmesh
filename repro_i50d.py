import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
import unmesh_harness.degrade.faults  # noqa: F401
import unmesh_harness.degrade.precision  # noqa: F401
import unmesh_harness.degrade.defects  # noqa: F401
import unmesh
from unmesh import step as step_mod


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


chains = {
    "trunc": [("truncated_digits", 1.0)],
    "trunc+iso": [("truncated_digits", 1.0), ("noise_isotropic", 1.0)],
    "iso+trunc": [("noise_isotropic", 1.0), ("truncated_digits", 1.0)],
    "hole": [("hole_patch", 1.0)],
    "hole+iso": [("hole_patch", 1.0), ("noise_isotropic", 1.0)],
    "gap": [("unwelded_gap", 1.0)],
    "gap+iso": [("unwelded_gap", 1.0), ("noise_isotropic", 1.0)],
    "far": [("far_translation", 1.0)],
    "far+iso": [("far_translation", 1.0), ("noise_isotropic", 1.0)],
    "dup": [("duplicate_facets", 1.0)],
    "dup+iso": [("duplicate_facets", 1.0), ("noise_isotropic", 1.0)],
    "stray": [("stray_shells", 1.0)],
    "stray+iso": [("stray_shells", 1.0), ("noise_isotropic", 1.0)],
    "fin": [("nonmanifold_fin", 1.0)],
    "fin+iso": [("nonmanifold_fin", 1.0), ("noise_isotropic", 1.0)],
}
for family in ["through_bore", "revolved_dome"]:
    for seed in range(4):
        gt = generate(family, seed)
        mesh = labels.tessellate(gt.solid, 0.1, 0.5)
        for cname, chain in chains.items():
            try:
                deg = degrade_core.apply_chain(mesh, chain, 4000 + seed)
            except Exception as e:
                print(family, seed, cname, "DEGRADE-RAISE", type(e).__name__, e, flush=True)
                continue
            sv = signed_volume(deg.tris)
            try:
                ir, rep = unmesh.convert(deg.tris)
                w = [str(x.code) for x in rep.warnings]
                out = step_mod.write(ir, f"repro4_{family}_{seed}_{cname}.step", mesh=deg.tris)
                vols = [(s.shell, s.kind, s.valid, s.volume) for s in out.shells]
                bad = (not out.valid) or any(
                    v is not None and v < 0 for _, _, _, v in vols
                )
                if bad:
                    print(family, seed, cname, "ntris", len(deg.tris),
                          "signed", round(sv, 3), "valid", out.valid,
                          "fallback", out.fallback, vols, "warn", w,
                          f"ISSUES {out.issues} {[s.issues for s in out.shells]}",
                          flush=True)
            except Exception as e:
                print(family, seed, cname, "RAISE", type(e).__name__, e, flush=True)
        print(family, seed, "done", flush=True)
print("SWEEP DONE", flush=True)
