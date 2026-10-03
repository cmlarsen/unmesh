import math

from unmesh_harness.groundtruth.complex import (
    assembly_volume,
    finned_volume,
    mixed_volume,
    void_volume,
)


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
    if f == "planar_chamfer":
        length, width, height = p["box"]
        c = p["chamfer"]
        return length * width * height - c * c * (length + width) + 4 * c**3 / 3
    if f == "straight_fillet":
        length, width, height = p["box"]
        r = p["radius"]
        return length * width * height - 4 * (1 - math.pi / 4) * r * r * height
    if f == "circular_fillet":
        radius, height = p["disc"]
        r = p["fillet"]
        return (
            math.pi * radius * radius * height
            - 2 * math.pi * radius * r * r * (1 - math.pi / 4)
            + r**3 * (5 * math.pi / 3 - math.pi**2 / 2)
        )
    if f == "bore_chamfer":
        c = p["chamfer"]
        if p["variant"] == "bore":
            length, width, height = p["plate"]
            bore = p["bore"]
            return (
                length * width * height
                - math.pi * bore * bore * height
                - (math.pi * c * c * bore + math.pi * c**3 / 3)
            )
        radius, height = p["disc"]
        return math.pi * radius * radius * height - (math.pi * c * c * radius - math.pi * c**3 / 3)
    if f == "corner_fillet":
        a, b, height = p["box"]
        r = p["radius"]
        u, v, w = a - 2 * r, b - 2 * r, height - 2 * r
        return (
            u * v * w
            + 2 * r * (u * v + u * w + v * w)
            + math.pi * r * r * (u + v + w)
            + 4 / 3 * math.pi * r**3
        )
    if f == "ngon_prism":
        n, radius, height = p["n"], p["radius"], p["height"]
        return n / 2 * radius**2 * math.sin(2 * math.pi / n) * height
    if f == "coarse_cylinder_prism":
        return math.pi * p["radius"] ** 2 * p["height"]
    if f == "one_segment_fillet":
        length, width, height = p["box"]
        r = p["radius"]
        return length * width * height - (1 - math.pi / 4) * r * r * height
    if f == "chamfer_same_chord":
        length, width, height = p["box"]
        c = p["chamfer"]
        return length * width * height - c * c / 2 * height
    if f == "two_segment_fillet":
        length, width, height = p["box"]
        r = p["radius"]
        return length * width * height - (1 - math.pi / 4) * r * r * height
    if f == "two_planes":
        length, width, height = p["box"]
        corner = [(length / 2, width / 2)] + [tuple(q) for q in feats[0]["points"]]
        return length * width * height - _shoelace(corner) * height
    if f == "complex_mixed":
        return mixed_volume(p, feats)
    if f == "complex_thin":
        return finned_volume(p, feats)
    if f == "complex_void":
        return void_volume(p, feats)
    if f == "complex_assembly":
        return assembly_volume(p, feats)
    raise KeyError(f)
