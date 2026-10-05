from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .topology import BuildError

PLANE_TOLERANCE = 1e-12
UNCERTAINTY = 1e-7
TIMESTAMP = "2000-01-01T00:00:00"
_DIRECTIONS = np.array(
    [
        (x, y, z)
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
        for z in (-1.0, 0.0, 1.0)
        if (x, y, z) != (0.0, 0.0, 0.0)
    ]
)


@dataclass
class Face:
    loop: list[int]
    normal: np.ndarray


@dataclass
class ShellFaces:
    faces: list[Face]
    probes: int = 0


@dataclass
class Body:
    outer: ShellFaces
    voids: list[ShellFaces] = field(default_factory=list)


def _unit_normals(vertices: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = vertices[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    length = np.linalg.norm(n, axis=1)
    unit = np.divide(n, length[:, None], out=np.zeros_like(n), where=length[:, None] > 0)
    return unit, length


def _edge_owner(loops: list[list[int]]) -> dict[tuple[int, int], int]:
    owner = {}
    for i, loop in enumerate(loops):
        if loop:
            for k in range(len(loop)):
                owner[(loop[k], loop[(k + 1) % len(loop)])] = i
    return owner


def _absorb_degenerate(
    vertices: np.ndarray, loops: list[list[int]], degenerate: np.ndarray
) -> None:
    for i in np.flatnonzero(degenerate):
        a, b, c = loops[i]
        p = vertices[[a, b, c]]
        d = p - p[0]
        axis = d[np.argmax(np.linalg.norm(d, axis=1))]
        t = p @ axis
        lo, hi = int(np.argmin(t)), int(np.argmax(t))
        mid = 3 - lo - hi
        tri = [a, b, c]
        u, v, m = tri[lo], tri[hi], tri[mid]
        if tri[(tri.index(u) + 1) % 3] != v:
            u, v = v, u
        owner = _edge_owner(loops)
        j = owner.get((v, u))
        if j is None or j == i or degenerate[j]:
            raise BuildError("a zero-area triangle has no neighbour to absorb it")
        loop = loops[j]
        k = loop.index(v)
        loop.insert(k + 1, m)
        loops[i] = []


def _hull_faces(vertices: np.ndarray, tris: np.ndarray, outward: np.ndarray, ok: np.ndarray):
    pts = vertices[np.unique(tris)]
    scale = float(np.linalg.norm(np.ptp(pts, axis=0))) or 1.0
    extreme = np.unique(np.argmax(pts @ _DIRECTIONS.T, axis=0))
    ids = np.unique(tris)[extreme]
    candidates = np.flatnonzero(np.isin(tris, ids).any(axis=1) & ok)
    hulls = []
    for c in candidates:
        height = (pts - vertices[tris[c, 0]]) @ outward[c]
        if height.max() <= PLANE_TOLERANCE * scale:
            hulls.append(int(c))
    return hulls


def _merge_coplanar(vertices, loops, normal, area, scale):
    owner = _edge_owner(loops)
    parent = list(range(len(loops)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), i in owner.items():
        j = owner.get((b, a))
        if j is None or j <= i:
            continue
        if normal[i] @ normal[j] < 1 - 1e-12:
            continue
        off = (vertices[loops[j]] - vertices[loops[i][0]]) @ normal[i]
        if np.abs(off).max() <= PLANE_TOLERANCE * scale:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i, loop in enumerate(loops):
        if loop:
            groups.setdefault(find(i), []).append(i)
    faces = []
    for members in groups.values():
        merged = None
        if len(members) > 1:
            merged = _boundary_loop(vertices, loops, normal, area, members, scale)
        if merged is None:
            faces.extend((i, [i], Face(loops[i], normal[i])) for i in members)
        else:
            faces.append((min(members), members, merged))
    faces.sort(key=lambda f: f[0])
    return faces


def _boundary_loop(vertices, loops, normal, area, members, scale):
    ref = max(members, key=lambda i: area[i])
    edges = {
        (loop[k], loop[(k + 1) % len(loop)])
        for loop in (loops[i] for i in members)
        for k in range(len(loop))
    }
    nxt: dict[int, int] = {}
    for a, b in edges:
        if (b, a) in edges:
            continue
        if a in nxt:
            return None
        nxt[a] = b
    start = min(nxt)
    loop = [start]
    while True:
        v = nxt[loop[-1]]
        if v == start:
            break
        loop.append(v)
        if len(loop) > len(nxt):
            return None
    if len(loop) != len(nxt):
        return None
    off = (vertices[loop] - vertices[loops[ref][0]]) @ normal[ref]
    if np.abs(off).max() > PLANE_TOLERANCE * scale:
        return None
    return Face(loop, normal[ref])


def shell_faces(vertices: np.ndarray, tris: np.ndarray, *, outer: bool) -> ShellFaces:
    tris = np.asarray(tris, dtype=np.int64)
    if len(tris) == 0:
        raise BuildError("shell has no triangles")
    normal, area = _unit_normals(vertices, tris)
    ok = area > 0
    loops = [[int(v) for v in t] for t in tris]
    _absorb_degenerate(vertices, loops, ~ok)
    pts = vertices[np.unique(tris)]
    scale = float(np.linalg.norm(np.ptp(pts, axis=0))) or 1.0
    probes = set(_hull_faces(vertices, tris, normal, ok)) if outer else set()
    merged = _merge_coplanar(vertices, loops, normal, area, scale)
    first = [f for _, members, f in merged if probes.intersection(members)]
    rest = [f for _, members, f in merged if not probes.intersection(members)]
    return ShellFaces(first + rest, len(first))


def open_faces(vertices: np.ndarray, tris: np.ndarray) -> ShellFaces:
    tris = np.asarray(tris, dtype=np.int64)
    normal, area = _unit_normals(vertices, tris)
    keep = area > 0
    faces = zip(tris[keep], normal[keep], strict=True)
    return ShellFaces([Face([int(v) for v in t], n) for t, n in faces])


def _real(x: float) -> str:
    s = repr(float(x) + 0.0)
    if "e" in s:
        mantissa, exponent = s.split("e")
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{exponent}"
    return s


def _string(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class _Writer:
    def __init__(self, vertices: np.ndarray):
        self.vertices = vertices
        self.lines: list[str] = []
        self.shared: dict[str, int] = {}
        self.points: dict[int, int] = {}

    def add(self, text: str) -> int:
        self.lines.append(text)
        return len(self.lines)

    def once(self, text: str) -> int:
        ref = self.shared.get(text)
        if ref is None:
            ref = self.shared[text] = self.add(text)
        return ref

    def point(self, v: int) -> int:
        ref = self.points.get(v)
        if ref is None:
            x, y, z = self.vertices[v]
            ref = self.points[v] = self.once(
                f"CARTESIAN_POINT('',({_real(x)},{_real(y)},{_real(z)}))"
            )
        return ref

    def direction(self, d) -> int:
        return self.once(f"DIRECTION('',({_real(d[0])},{_real(d[1])},{_real(d[2])}))")

    def shell(self, shell: ShellFaces, kind: str, reverse: bool = False) -> int:
        vertex_points: dict[int, int] = {}
        edges: dict[tuple[int, int], int] = {}

        def vertex(v: int) -> int:
            ref = vertex_points.get(v)
            if ref is None:
                ref = vertex_points[v] = self.add(f"VERTEX_POINT('',#{self.point(v)})")
            return ref

        def oriented_edge(a: int, b: int) -> int:
            lo, hi = (a, b) if a < b else (b, a)
            ref = edges.get((lo, hi))
            if ref is None:
                d = self.vertices[hi] - self.vertices[lo]
                length = float(np.linalg.norm(d))
                vec = self.add(f"VECTOR('',#{self.direction(d / length)},{_real(length)})")
                line = self.add(f"LINE('',#{self.point(lo)},#{vec})")
                ref = edges[(lo, hi)] = self.add(
                    f"EDGE_CURVE('',#{vertex(lo)},#{vertex(hi)},#{line},.T.)"
                )
            sense = ".T." if a == lo else ".F."
            return self.add(f"ORIENTED_EDGE('',*,*,#{ref},{sense})")

        faces = []
        for face in shell.faces:
            loop = face.loop[::-1] if reverse else face.loop
            normal = -face.normal if reverse else face.normal
            oes = [oriented_edge(loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))]
            edge_loop = self.add("EDGE_LOOP('',(" + ",".join(f"#{e}" for e in oes) + "))")
            bound = self.add(f"FACE_OUTER_BOUND('',#{edge_loop},.T.)")
            u = self.vertices[loop[1]] - self.vertices[loop[0]]
            u = u - (u @ normal) * normal
            u /= np.linalg.norm(u)
            axis = self.add(
                f"AXIS2_PLACEMENT_3D('',#{self.point(loop[0])},"
                f"#{self.direction(normal)},#{self.direction(u)})"
            )
            plane = self.add(f"PLANE('',#{axis})")
            faces.append(self.add(f"ADVANCED_FACE('',(#{bound}),#{plane},.T.)"))
        return self.add(f"{kind}('',(" + ",".join(f"#{f}" for f in faces) + "))")


def write(
    path: str | os.PathLike,
    vertices: np.ndarray,
    bodies: list[Body],
    open_shells: list[ShellFaces],
    name: str = "unmesh",
) -> None:
    w = _Writer(np.asarray(vertices, dtype=float))
    apc = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{apc})"
    )
    pc = w.add(f"PRODUCT_CONTEXT('',#{apc},'mechanical')")
    product = w.add(f"PRODUCT({_string(name)},{_string(name)},'',(#{pc}))")
    w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{product}))")
    pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    pdc = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{apc},'design')")
    pd = w.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")
    pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    length = w.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
    angle = w.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    solid_angle = w.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    uncertainty = w.add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({_real(UNCERTAINTY)}),#{length},"
        "'distance_accuracy_value','confusion accuracy')"
    )
    ctx = w.add(
        "( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle})) "
        "REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )"
    )
    zero = w.once("CARTESIAN_POINT('',(0.,0.,0.))")
    origin = w.add(
        f"AXIS2_PLACEMENT_3D('',#{zero},"
        f"#{w.direction((0.0, 0.0, 1.0))},#{w.direction((1.0, 0.0, 0.0))})"
    )
    items = []
    for body in bodies:
        outer = w.shell(body.outer, "CLOSED_SHELL")
        if not body.voids:
            items.append(w.add(f"MANIFOLD_SOLID_BREP('',#{outer})"))
            continue
        voids = []
        for void in body.voids:
            closed = w.shell(void, "CLOSED_SHELL", reverse=True)
            voids.append(w.add(f"ORIENTED_CLOSED_SHELL('',*,#{closed},.F.)"))
        items.append(
            w.add(f"BREP_WITH_VOIDS('',#{outer},(" + ",".join(f"#{v}" for v in voids) + "))")
        )
    reps = []
    if items:
        reps.append(
            w.add(
                "ADVANCED_BREP_SHAPE_REPRESENTATION('',("
                + ",".join(f"#{i}" for i in [origin, *items])
                + f"),#{ctx})"
            )
        )
    if open_shells:
        shells = [w.shell(s, "OPEN_SHELL") for s in open_shells]
        model = w.add("SHELL_BASED_SURFACE_MODEL('',(" + ",".join(f"#{s}" for s in shells) + "))")
        reps.append(w.add(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',(#{origin},#{model}),#{ctx})"))
    if not reps:
        raise BuildError("nothing to write")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{reps[0]})")
    for rep in reps[1:]:
        w.add(f"SHAPE_REPRESENTATION_RELATIONSHIP('','',#{rep},#{reps[0]})")
    header = (
        "ISO-10303-21;\nHEADER;\n"
        "FILE_DESCRIPTION(('unmesh faceted solid'),'2;1');\n"
        f"FILE_NAME({_string(name)},'{TIMESTAMP}',(''),(''),'unmesh','unmesh','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
        "ENDSEC;\nDATA;\n"
    )
    body_text = "".join(f"#{i} = {line};\n" for i, line in enumerate(w.lines, start=1))
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(header + body_text + "ENDSEC;\nEND-ISO-10303-21;\n")
