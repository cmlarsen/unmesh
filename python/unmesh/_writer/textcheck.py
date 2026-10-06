from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

CHUNK = 8 << 20
_ID = r"\n#(\d+) = "
_REF = r"#(\d+)"
_LIST = r"\(([#\d,]+)\)"
_REAL = r"([-+0-9.Ee]+)"
_BOOL = r"\.([TF])\."

_STRING = r"'(?:[^'\n\\]|'')*'"
_UNITS = r"\(\(#(\d+),#(\d+),#(\d+)\)\)"

_KINDS = {
    "point": rf"CARTESIAN_POINT\('',\({_REAL},{_REAL},{_REAL}\)\)",
    "direction": rf"DIRECTION\('',\({_REAL},{_REAL},{_REAL}\)\)",
    "vector": rf"VECTOR\('',{_REF},{_REAL}\)",
    "line": rf"LINE\('',{_REF},{_REF}\)",
    "vertex": rf"VERTEX_POINT\('',{_REF}\)",
    "edge": rf"EDGE_CURVE\('',{_REF},{_REF},{_REF},{_BOOL}\)",
    "oriented_edge": rf"ORIENTED_EDGE\('',\*,\*,{_REF},{_BOOL}\)",
    "loop": rf"EDGE_LOOP\('',{_LIST}\)",
    "bound": rf"FACE_OUTER_BOUND\('',{_REF},{_BOOL}\)",
    "axis": rf"AXIS2_PLACEMENT_3D\('',{_REF},{_REF},{_REF}\)",
    "plane": rf"PLANE\('',{_REF}\)",
    "face": rf"ADVANCED_FACE\('',\({_REF}\),{_REF},{_BOOL}\)",
    "closed_shell": rf"CLOSED_SHELL\('',{_LIST}\)",
    "open_shell": rf"OPEN_SHELL\('',{_LIST}\)",
    "oriented_shell": rf"ORIENTED_CLOSED_SHELL\('',\*,{_REF},{_BOOL}\)",
    "manifold": rf"MANIFOLD_SOLID_BREP\('',{_REF}\)",
    "voids": rf"BREP_WITH_VOIDS\('',{_REF},{_LIST}\)",
    "brep_rep": rf"ADVANCED_BREP_SHAPE_REPRESENTATION\('',{_LIST},{_REF}\)",
    "surface_model": rf"SHELL_BASED_SURFACE_MODEL\('',{_LIST}\)",
    "surface_rep": rf"MANIFOLD_SURFACE_SHAPE_REPRESENTATION\('',{_LIST},{_REF}\)",
    "application": r"APPLICATION_CONTEXT\('core data for automotive mechanical design processes'\)",
    "protocol": rf"APPLICATION_PROTOCOL_DEFINITION\('international standard',"
    rf"'automotive_design',2000,{_REF}\)",
    "product_context": rf"PRODUCT_CONTEXT\('',{_REF},'mechanical'\)",
    "product": rf"PRODUCT\({_STRING},{_STRING},'',\({_REF}\)\)",
    "category": rf"PRODUCT_RELATED_PRODUCT_CATEGORY\('part',\$,\({_REF}\)\)",
    "formation": rf"PRODUCT_DEFINITION_FORMATION\('','',{_REF}\)",
    "definition_context": rf"PRODUCT_DEFINITION_CONTEXT\('part definition',{_REF},'design'\)",
    "definition": rf"PRODUCT_DEFINITION\('design','',{_REF},{_REF}\)",
    "definition_shape": rf"PRODUCT_DEFINITION_SHAPE\('','',{_REF}\)",
    "uncertainty": rf"UNCERTAINTY_MEASURE_WITH_UNIT\(LENGTH_MEASURE\({_REAL}\),{_REF},"
    r"'distance_accuracy_value','confusion accuracy'\)",
    "shape_definition": rf"SHAPE_DEFINITION_REPRESENTATION\({_REF},{_REF}\)",
    "relationship": rf"SHAPE_REPRESENTATION_RELATIONSHIP\('','',{_REF},{_REF}\)",
    "length_unit": r"\( LENGTH_UNIT\(\) NAMED_UNIT\(\*\) SI_UNIT\(\.MILLI\.,\.METRE\.\) \)",
    "angle_unit": r"\( NAMED_UNIT\(\*\) PLANE_ANGLE_UNIT\(\) SI_UNIT\(\$,\.RADIAN\.\) \)",
    "solid_angle_unit": r"\( NAMED_UNIT\(\*\) SI_UNIT\(\$,\.STERADIAN\.\) SOLID_ANGLE_UNIT\(\) \)",
    "context": rf"\( GEOMETRIC_REPRESENTATION_CONTEXT\(3\) "
    rf"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT\(\({_REF}\)\) GLOBAL_UNIT_ASSIGNED_CONTEXT{_UNITS} "
    r"REPRESENTATION_CONTEXT\('Context #1','3D Context with UNIT and UNCERTAINTY'\) \)",
}
_COMPLEX = ("length_unit", "angle_unit", "solid_angle_unit", "context")
_SINGLE = (
    "application",
    "protocol",
    "product_context",
    "product",
    "category",
    "formation",
    "definition_context",
    "definition",
    "definition_shape",
    "uncertainty",
    "shape_definition",
    *_COMPLEX,
)
_PATTERNS = {k: re.compile(_ID + v + r";(?=\n)") for k, v in _KINDS.items()}
_NAMES = {v.split("\\(")[0]: k for k, v in _KINDS.items() if k not in _COMPLEX}
_COMMON = list(_KINDS)[:12]
_RARE = re.compile(
    _ID
    + "("
    + "|".join([*(n for n, k in _NAMES.items() if k not in _COMMON), ""])
    + r")\(([^\n]*)\);(?=\n)"
)
_HEADER = re.compile(
    r"ISO-10303-21;\nHEADER;\n"
    r"FILE_DESCRIPTION\(\('unmesh faceted solid'\),'2;1'\);\n"
    rf"FILE_NAME\({_STRING},'[0-9T:-]+',\(''\),\(''\),'unmesh','unmesh',''\);\n"
    r"FILE_SCHEMA\(\('AUTOMOTIVE_DESIGN \{ 1 0 10303 214 1 1 1 1 \}'\)\);\n"
    r"ENDSEC;\nDATA;\n"
)
_FOOTER = b"ENDSEC;\nEND-ISO-10303-21;\n"
_LISTS = {
    "loop": 1,
    "closed_shell": 1,
    "open_shell": 1,
    "voids": 2,
    "brep_rep": 1,
    "surface_model": 1,
    "surface_rep": 1,
}
_FLOATS = {"point", "direction", "vector", "uncertainty"}


class _Bad(Exception):
    pass


@dataclass
class TextSolid:
    volume: float
    valid: bool
    tolerance: float
    shells: int
    issues: list[str] = field(default_factory=list)


@dataclass
class TextRead:
    solids: list[TextSolid]
    shells: int
    issues: list[str] = field(default_factory=list)


@dataclass
class _Table:
    ids: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    lists: list = field(default_factory=list)


def _sections(path) -> tuple[int, int]:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(4096).decode("ascii", "replace")
        fh.seek(max(size - len(_FOOTER), 0))
        tail = fh.read()
    m = _HEADER.match(head)
    if m is None:
        raise _Bad("the file does not start with the faceted writer's Part 21 header")
    if tail != _FOOTER or size - len(_FOOTER) < m.end():
        raise _Bad("the file does not end with ENDSEC; END-ISO-10303-21; (truncated?)")
    return m.end() - 1, size - len(_FOOTER)


def _chunks(path, start: int, end: int):
    with open(path, "rb") as fh:
        fh.seek(start)
        left = end - start
        tail = b""
        while left > 0:
            block = tail + fh.read(min(CHUNK, left))
            left = end - fh.tell()
            cut = block.rfind(b"\n") if left > 0 else len(block) - 1
            if cut <= 0:
                tail = block
                continue
            yield block[: cut + 1].decode("ascii")
            tail = block[cut:]


def _parse(path) -> tuple[dict[str, _Table], int, int]:
    tables = {k: _Table() for k in _KINDS}
    total = matched = 0
    start, end = _sections(path)
    for text in _chunks(path, start, end):
        if not text.startswith("\n") or not text.endswith("\n"):
            raise _Bad("the DATA section is not made of whole lines")
        total += text.count("\n") - 1
        for kind in _COMMON:
            rows = _PATTERNS[kind].findall(text)
            matched += len(rows)
            _add(tables[kind], kind, rows)
        rare: dict[str, list] = {}
        for ref, name, args in _RARE.findall(text):
            line = f"\n#{ref} = {name}({args});\n"
            kinds = [_NAMES[name]] if name else _COMPLEX
            for kind in kinds:
                rows = _PATTERNS[kind].findall(line)
                if rows:
                    matched += len(rows)
                    rare.setdefault(kind, []).extend(rows)
                    break
        for kind, rows in rare.items():
            _add(tables[kind], kind, rows)
    return tables, total, matched


def _add(t: _Table, kind: str, rows: list) -> None:
    if not rows:
        return
    if isinstance(rows[0], str):
        rows = [(r,) for r in rows]
    if kind in _LISTS:
        at = _LISTS[kind]
        t.ids.append(np.array([r[0] for r in rows], dtype=np.int64))
        t.lists.extend(r[at] for r in rows)
        rest = [r[1:at] + r[at + 1 :] for r in rows]
        if rest[0]:
            t.cols.append(np.array(rest, dtype=np.int64))
        return
    arr = np.array(rows)
    t.ids.append(arr[:, 0].astype(np.int64))
    body = arr[:, 1:]
    if kind in _FLOATS:
        t.cols.append(body.astype(float))
    else:
        body[body == "T"] = "1"
        body[body == "F"] = "0"
        t.cols.append(body.astype(np.int64))


class _Index:
    def __init__(self, tables):
        self.ids = {}
        self.cols = {}
        self.lists = {}
        top = 0
        for kind in list(tables):
            t = tables.pop(kind)
            ids = np.concatenate(t.ids) if t.ids else np.zeros(0, dtype=np.int64)
            self.ids[kind] = ids
            if t.cols:
                self.cols[kind] = np.concatenate(t.cols)
            if kind in _LISTS:
                self.lists[kind] = _flatten(t.lists)
            top = max(top, int(ids.max(initial=0)))
        self.kind = np.full(top + 1, -1, dtype=np.int16)
        self.row = np.full(top + 1, -1, dtype=np.int64)
        self.names = list(self.ids)
        for k, kind in enumerate(self.names):
            ids = self.ids[kind]
            if (self.kind[ids] != -1).any() or len(np.unique(ids)) != len(ids):
                raise _Bad("an entity id is defined twice")
            self.kind[ids] = k
            self.row[ids] = np.arange(len(ids))

    def rows(self, refs, kind, what):
        refs = np.asarray(refs, dtype=np.int64)
        if len(refs) and (refs.min() < 0 or refs.max() >= len(self.kind)):
            raise _Bad(f"{what} refers to an entity that is not in the file")
        if (self.kind[refs] != self.names.index(kind)).any():
            raise _Bad(f"{what} refers to an entity that is not a {kind.replace('_', ' ')}")
        return self.row[refs]


def _flatten(lists: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not lists:
        return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    counts = np.array([s.count(",") + 1 for s in lists], dtype=np.int64)
    joined = "," + ",".join(lists)
    if joined.count(",#") != counts.sum() or joined.count("#") != counts.sum():
        raise _Bad("a list of references is malformed")
    flat = np.array(joined[1:].replace("#", "").split(","), dtype=np.int64)
    return flat, np.concatenate([[0], np.cumsum(counts)])


def _require(ok, what):
    if not np.all(ok):
        raise _Bad(what)


def _unit(v):
    n = np.linalg.norm(v, axis=1)
    _require(n > 0, "a direction has zero length")
    return v / n[:, None]


def _line_distance(p, origin, direction):
    d = p - origin
    return np.linalg.norm(d - (d * direction).sum(axis=1)[:, None] * direction, axis=1)


def read(path: str | os.PathLike) -> TextRead:
    try:
        tables, total, matched = _parse(path)
        if matched != total:
            raise _Bad(
                f"{total - matched} lines of the DATA section are not entities the faceted"
                " writer emits"
            )
        ix = _Index(tables)
        _structure(ix)
        return _check(ix)
    except _Bad as e:
        return TextRead([], 0, [str(e)])
    except (KeyError, IndexError, ValueError) as e:
        return TextRead([], 0, [f"the file could not be parsed: {type(e).__name__}: {e}"])


_CHAIN = (
    ("protocol", 0, "application"),
    ("product_context", 0, "application"),
    ("product", 0, "product_context"),
    ("category", 0, "product"),
    ("formation", 0, "product"),
    ("definition_context", 0, "application"),
    ("definition", 0, "formation"),
    ("definition", 1, "definition_context"),
    ("definition_shape", 0, "definition"),
    ("uncertainty", 1, "length_unit"),
    ("context", 0, "uncertainty"),
    ("context", 1, "length_unit"),
    ("context", 2, "angle_unit"),
    ("context", 3, "solid_angle_unit"),
    ("shape_definition", 0, "definition_shape"),
)


def _members(ix: _Index, kind: str, allowed: tuple[str, ...]) -> list[str]:
    flat, _ = ix.lists[kind]
    if len(flat) and (flat.max() >= len(ix.kind) or flat.min() < 0):
        raise _Bad(f"a {kind.replace('_', ' ')} item is not in the file")
    names = [ix.names[k] if k >= 0 else None for k in ix.kind[flat]]
    if any(n not in allowed for n in names):
        raise _Bad(f"a {kind.replace('_', ' ')} item is not one of {', '.join(allowed)}")
    return names


def _structure(ix: _Index) -> None:
    for kind in _SINGLE:
        if len(ix.ids[kind]) != 1:
            raise _Bad(f"the file has {len(ix.ids[kind])} {kind.replace('_', ' ')} entities, not 1")
    for kind, col, target in _CHAIN:
        ix.rows(ix.cols[kind][:, col].astype(np.int64), target, f"the {kind.replace('_', ' ')}")
    if not ix.cols["uncertainty"][0, 0] > 0:
        raise _Bad("the length uncertainty is not positive")
    breps, surfaces = len(ix.ids["brep_rep"]), len(ix.ids["surface_rep"])
    if breps > 1 or surfaces > 1 or breps + surfaces == 0:
        raise _Bad(f"the file has {breps} brep and {surfaces} surface representations")
    for kind in ("brep_rep", "surface_rep"):
        if len(ix.ids[kind]):
            ix.rows(ix.cols[kind][:, 0], "context", f"the {kind.replace('_', ' ')}")
    main = "brep_rep" if breps else "surface_rep"
    ix.rows(ix.cols["shape_definition"][:, 1], main, "the shape definition representation")
    rel = ix.cols.get("relationship", np.zeros((0, 2), dtype=np.int64))
    if len(rel) != (1 if breps and surfaces else 0):
        raise _Bad("the representation relationships do not link the surface representation")
    if len(rel):
        ix.rows(rel[:, 0], "surface_rep", "the representation relationship")
        ix.rows(rel[:, 1], "brep_rep", "the representation relationship")
    items = _members(ix, "brep_rep", ("axis", "manifold", "voids"))
    solids = len(ix.ids["manifold"]) + len(ix.ids["voids"])
    if items.count("manifold") + items.count("voids") != solids or len(
        set(ix.lists["brep_rep"][0].tolist())
    ) != len(items):
        raise _Bad("a solid is not in the brep representation exactly once")
    models = _members(ix, "surface_rep", ("axis", "surface_model"))
    if models.count("surface_model") != len(ix.ids["surface_model"]) or len(
        set(ix.lists["surface_rep"][0].tolist())
    ) != len(models):
        raise _Bad("a surface model is not in the surface representation exactly once")


def _check(ix: _Index) -> TextRead:
    pts = ix.cols["point"]
    pts = pts - 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    dirs = _unit(ix.cols["direction"])
    vertex_pt = pts[ix.rows(ix.cols["vertex"][:, 0], "point", "a vertex")]

    vec = ix.cols["vector"]
    vec_dir = dirs[ix.rows(vec[:, 0].astype(np.int64), "direction", "a vector")]
    line = ix.cols["line"]
    line_origin = pts[ix.rows(line[:, 0], "point", "a line")]
    line_dir = vec_dir[ix.rows(line[:, 1], "vector", "a line")]

    edge = ix.cols["edge"]
    _require(edge[:, 3] == 1, "an edge curve has same_sense .F.")
    v0 = ix.rows(edge[:, 0], "vertex", "an edge")
    v1 = ix.rows(edge[:, 1], "vertex", "an edge")
    _require(v0 != v1, "an edge starts and ends at the same vertex")
    lr = ix.rows(edge[:, 2], "line", "an edge")
    a, b = vertex_pt[v0], vertex_pt[v1]
    o, d = line_origin[lr], line_dir[lr]
    deviation = np.maximum(_line_distance(a, o, d), _line_distance(b, o, d))
    _require(((b - a) * d).sum(axis=1) > 0, "an edge runs against its line's direction")

    oe = ix.cols["oriented_edge"]
    oe_edge = ix.rows(oe[:, 0], "edge", "an oriented edge")
    forward = oe[:, 1] == 1
    oe_start = np.where(forward, v0[oe_edge], v1[oe_edge])
    oe_end = np.where(forward, v1[oe_edge], v0[oe_edge])

    flat, offsets = ix.lists["loop"]
    counts = np.diff(offsets)
    _require(counts >= 3, "an edge loop has fewer than three edges")
    use = ix.rows(flat, "oriented_edge", "an edge loop")
    _require(np.bincount(use, minlength=len(oe)) == 1, "an oriented edge is not used exactly once")
    loop_of_use = np.repeat(np.arange(len(counts)), counts)
    nxt = np.arange(len(use)) + 1
    nxt[offsets[1:] - 1] = offsets[:-1]
    _require(oe_end[use] == oe_start[use[nxt]], "an edge loop is not connected")
    corner = loop_of_use * len(vertex_pt) + oe_start[use]
    _require(len(np.unique(corner)) == len(corner), "an edge loop passes through a vertex twice")

    bound = ix.cols["bound"]
    _require(bound[:, 1] == 1, "a face bound is reversed")
    bound_loop = ix.rows(bound[:, 0], "loop", "a face bound")
    axis = ix.cols["axis"]
    axis_origin = pts[ix.rows(axis[:, 0], "point", "a placement")]
    axis_normal = dirs[ix.rows(axis[:, 1], "direction", "a placement")]
    plane = ix.cols["plane"][:, 0]
    plane_axis = ix.rows(plane, "axis", "a plane")

    face = ix.cols["face"]
    _require(face[:, 2] == 1, "a face has same_sense .F.")
    face_loop = bound_loop[ix.rows(face[:, 0], "bound", "a face")]
    _require(np.bincount(face_loop, minlength=len(counts)) == 1, "an edge loop is not used once")
    face_axis = plane_axis[ix.rows(face[:, 1], "plane", "a face")]
    face_of_loop = np.full(len(counts), -1, dtype=np.int64)
    face_of_loop[face_loop] = np.arange(len(face))
    face_of_use = face_of_loop[loop_of_use]

    p = vertex_pt[oe_start[use]]
    cross = np.cross(p, p[nxt]) / 2.0
    area = np.stack(
        [np.bincount(face_of_use, weights=cross[:, k], minlength=len(face)) for k in range(3)],
        axis=1,
    )
    del cross
    face_normal = axis_normal[face_axis]
    height = (axis_origin[face_axis] * face_normal).sum(axis=1)
    off_plane = np.abs(np.einsum("ij,ij->i", p, face_normal[face_of_use]) - height[face_of_use])
    winds = (area * face_normal).sum(axis=1) > 0
    first = p[offsets[face_loop]]
    face_volume = (area * first).sum(axis=1) / 3.0
    face_tolerance = np.zeros(len(face))
    np.maximum.at(face_tolerance, face_of_use, np.maximum(off_plane, deviation[oe_edge[use]]))

    shells = {}
    for kind, closed in (("closed_shell", True), ("open_shell", False)):
        sflat, soff = ix.lists[kind]
        rows = ix.rows(sflat, "face", f"a {kind.replace('_', ' ')}")
        shells[kind] = (rows, soff, closed)
    every = np.concatenate([shells[k][0] for k in shells])
    _require(np.bincount(every, minlength=len(face)) == 1, "a face is not in exactly one shell")

    sense = forward[use]
    edge_of_use = oe_edge[use]
    report = {}
    for kind, (rows, soff, closed) in shells.items():
        for s in range(len(soff) - 1):
            members = rows[soff[s] : soff[s + 1]]
            report[(kind, s)] = _shell(
                members, closed, face_of_use, edge_of_use, sense, winds, face_volume, face_tolerance
            )
    return _solids(ix, shells, report)


def _shell(members, closed, face_of_use, edge_of_use, sense, winds, face_volume, face_tolerance):
    issues = []
    mine = np.isin(face_of_use, members)
    edges = edge_of_use[mine]
    senses = sense[mine]
    uses = np.bincount(edges)
    forward = np.bincount(edges, weights=senses)
    used = uses > 0
    if closed:
        bad = int(np.count_nonzero(used & ((uses != 2) | (forward != 1))))
        if bad:
            issues.append(f"{bad} edges are not used once in each direction")
    else:
        bad = int(np.count_nonzero(used & ((uses > 2) | ((uses == 2) & (forward != 1)))))
        if bad:
            issues.append(f"{bad} edges are used twice in the same direction or more than twice")
    against = int(np.count_nonzero(~winds[members]))
    if against:
        issues.append(f"{against} faces wind against their plane's normal")
    return (
        float(face_volume[members].sum()),
        float(face_tolerance[members].max(initial=0.0)),
        issues,
        edges,
    )


def _solids(ix: _Index, shells, report) -> TextRead:
    issues = []
    closed_rows = ix.ids["closed_shell"]
    claimed = np.zeros(len(closed_rows), dtype=np.int64)
    shared = [r[3] for r in report.values()]
    if shared:
        all_edges = np.concatenate(shared)
        owners = np.concatenate([np.full(len(e), i) for i, e in enumerate(shared)])
        lo = np.full(int(all_edges.max(initial=-1)) + 1, len(shared))
        hi = np.full(len(lo), -1)
        np.minimum.at(lo, all_edges, owners)
        np.maximum.at(hi, all_edges, owners)
        if np.any((hi >= 0) & (lo != hi)):
            issues.append("an edge is shared between two shells")

    def closed(ref, what):
        row = int(ix.rows([ref], "closed_shell", what)[0])
        claimed[row] += 1
        return report[("closed_shell", row)]

    solid_rows = {}
    for kind in ("manifold", "voids"):
        for row, ref in enumerate(ix.ids[kind]):
            solid_rows[int(ref)] = (kind, row)
    solids = []
    items, ioff = ix.lists["brep_rep"]
    if len(ix.ids["brep_rep"]) > 1:
        issues.append("more than one advanced brep representation")
    for ref in items:
        if ix.kind[ref] == ix.names.index("axis"):
            continue
        if int(ref) not in solid_rows:
            raise _Bad("a representation item is not a solid")
        kind, row = solid_rows[int(ref)]
        if kind == "manifold":
            outer, voids = ix.cols["manifold"][row, 0], []
        else:
            outer = ix.cols["voids"][row, 0]
            vflat, voff = ix.lists["voids"]
            voids = vflat[voff[row] : voff[row + 1]]
        volume, tolerance, mine, _ = closed(outer, "a solid")
        mine = list(mine)
        if not volume > 0:
            mine.append(f"the outer shell encloses a non-positive volume {volume:.6g}")
        for v in voids:
            orow = int(ix.rows([v], "oriented_shell", "a void")[0])
            ref_shell, same = ix.cols["oriented_shell"][orow]
            vv, vt, vissues, _ = closed(ref_shell, "an oriented shell")
            vv = vv if same == 1 else -vv
            mine.extend(vissues)
            if not vv < 0:
                mine.append(f"a void encloses a non-negative volume {vv:.6g}")
            volume += vv
            tolerance = max(tolerance, vt)
        solids.append(TextSolid(volume, not mine, tolerance, 1 + len(voids), mine))
    if np.any(claimed != 1):
        issues.append("a closed shell is not used by exactly one solid")
    sflat, _ = ix.lists["surface_model"]
    open_rows = ix.rows(sflat, "open_shell", "a surface model")
    for row in open_rows:
        issues.extend(report[("open_shell", int(row))][2])
    if len(open_rows) != len(ix.ids["open_shell"]) or len(np.unique(open_rows)) != len(open_rows):
        issues.append("an open shell is not used by exactly one surface model")
    count = sum(s.shells for s in solids) + len(open_rows)
    return TextRead(solids, count, issues)
