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
    raise KeyError(f)
