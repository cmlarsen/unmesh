import numpy as np

from unmesh_harness.degrade import core as degrade_core
from unmesh_harness import labels
from unmesh_harness.groundtruth import generate
import unmesh_harness.degrade.noise  # noqa: F401
import unmesh_harness.degrade.coarsen  # noqa: F401
import unmesh_harness.degrade.retriangulate  # noqa: F401
import unmesh_harness.degrade.defects  # noqa: F401
import unmesh_harness.degrade.chords  # noqa: F401
import unmesh
from unmesh import step as step_mod


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


chains = {
    "slivers+iso": [("slivers", 1.0), ("noise_isotropic", 1.0)],
    "slivers+iso+normal": [("slivers", 1.0), ("noise_isotropic", 1.0), ("noise_normal", 1.0)],
    "coarsen+iso": [("coarsen", 1.0), ("noise_isotropic", 1.0)],
    "retri+iso": [("retriangulate", 1.0), ("noise_isotropic", 1.0)],
    "chords+iso": [("nonuniform_chords", 1.0), ("noise_isotropic", 1.0)],
    "tjunct+iso": [("t_junctions", 1.0), ("noise_isotropic", 1.0)],
}
families = ["through_bore", "round_boss", "revolved_dome", "counterbore", "revolved_cone"]
for family in families:
    for seed in range(6):
        try:
            gt = generate(family, seed)
            mesh = labels.tessellate(gt.solid, 0.01, 0.2)
        except Exception as e:
            print(family, seed, "SETUP-RAISE", type(e).__name__, str(e)[:120], flush=True)
            continue
        for cname, chain in chains.items():
            try:
                deg = degrade_core.apply_chain(mesh, chain, 6000 + seed)
                sv = signed_volume(deg.tris)
                ir, rep = unmesh.convert(deg.tris)
                w = [str(x.code) for x in rep.warnings]
                out = step_mod.write(ir, f"repro7_{family}_{seed}_{cname}.step", mesh=deg.tris)
                vols = [(s.shell, s.kind, s.valid, s.volume) for s in out.shells]
                bad = (not out.valid) or any(v is not None and v < 0 for _, _, _, v in vols)
                if bad:
                    print(family, seed, cname, "ntris", len(deg.tris),
                          "signed", round(sv, 3), "valid", out.valid,
                          "fallback", out.fallback, vols, "warn", w,
                          f"ISSUES {out.issues} {[s.issues for s in out.shells]}",
                          flush=True)
            except Exception as e:
                print(family, seed, cname, "RAISE", type(e).__name__, str(e)[:160], flush=True)
        print(family, seed, "done", flush=True)
print("SWEEP DONE", flush=True)
