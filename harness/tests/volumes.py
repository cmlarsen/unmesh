import math


def _shoelace(p):
    n = len(p)
    return abs(sum(p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1] for i in range(n))) / 2


def _area(x):
    return x["size"][0] * x["size"][1]


def expected_volume(gt):
    f, p, feats = gt.family, gt.parameters, gt.features
    if f in ("plate_pockets", "rotated_pockets", "square_slots"):
        length, width, t = p["plate"]
        return length * width * t - sum(_area(x) * x["depth"] for x in feats)
    if f == "through_cuts":
        length, width, t = p["plate"]
        return length * width * t - sum(_area(x) * t for x in feats)
    if f == "boss_plate":
        length, width, t = p["plate"]
        return length * width * t + sum(_area(x) * x["height"] for x in feats)
    if f in ("polygon_prism", "lshape_outline"):
        return _shoelace(feats[0]["points"]) * feats[0]["height"]
    if f == "thin_walls":
        length, width, height = p["block"]
        pocket = feats[0]
        v = length * width * height - _area(pocket) * pocket["depth"]
        for x in feats:
            if x["type"] == "thin_rib":
                v += x["length"] * x["thickness"] * x["height"]
        return v
    if f == "stepped_block":
        length, width, height = p["block"]
        xs = [-length / 2] + [x["x_start"] for x in feats] + [length / 2]
        hs = [height] + [x["top_z"] for x in feats]
        return sum((xs[i + 1] - xs[i]) * hs[i] for i in range(len(hs))) * width
    if f == "through_bore":
        length, width, t = p["plate"]
        return length * width * t - sum(math.pi * x["radius"] ** 2 * t for x in feats)
    if f == "blind_bore":
        length, width, t = p["plate"]
        return length * width * t - sum(math.pi * x["radius"] ** 2 * x["depth"] for x in feats)
    if f == "round_boss":
        length, width, t = p["plate"]
        return length * width * t + sum(math.pi * x["radius"] ** 2 * x["height"] for x in feats)
    if f == "counterbore":
        length, width, t = p["plate"]
        return length * width * t - sum(
            math.pi * x["bore_radius"] ** 2 * x["bore_depth"]
            + math.pi * x["shaft_radius"] ** 2 * (t - x["bore_depth"])
            for x in feats
        )
    if f == "countersink":
        length, width, t = p["plate"]
        return length * width * t - sum(
            math.pi
            * x["cone_depth"]
            / 3
            * (
                x["mouth_radius"] ** 2
                + x["mouth_radius"] * x["shaft_radius"]
                + x["shaft_radius"] ** 2
            )
            + math.pi * x["shaft_radius"] ** 2 * (t - x["cone_depth"])
            for x in feats
        )
    if f in ("round_slot_through", "round_slot_blind"):
        length, width, t = p["plate"]
        return length * width * t - sum(
            (x["span"] * 2 * x["radius"] + math.pi * x["radius"] ** 2) * x["depth"] for x in feats
        )
    if f == "revolved_cone":
        r1, h1, r2, h2 = p["r_base"], p["h_base"], p["r_top"], p["h_cone"]
        return math.pi * r1**2 * h1 + math.pi * h2 / 3 * (r1**2 + r1 * r2 + r2**2)
    if f == "revolved_dome":
        r, h = p["radius"], p["h_stem"]
        return math.pi * r**2 * h + 2 / 3 * math.pi * r**3
    if f == "revolved_torus":
        return 2 * math.pi**2 * p["major_radius"] * p["minor_radius"] ** 2
    raise KeyError(f)
