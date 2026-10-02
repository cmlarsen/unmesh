from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import unmesh
from unmesh.ir import (
    Adjacency,
    Boundary,
    Cone,
    Cylinder,
    Ir,
    Plane,
    Region,
    Residual,
    Shell,
    Source,
    Sphere,
    Tolerances,
    Torus,
    Vertex,
)

from .labels import TANGENT_THRESHOLD_DEG, EdgeAdjacency, FaceInfo, LabeledMesh, distance_to_surface


def _surface(face: FaceInfo):
    p = face.params
    orientation = "reversed" if face.reversed else "same"
    if face.surface == "plane":
        return Plane(tuple(p["origin"]), tuple(p["normal"]))
    if face.surface == "cylinder":
        return Cylinder(tuple(p["origin"]), tuple(p["axis"]), p["radius"], orientation)
    if face.surface == "cone":
        return Cone(tuple(p["apex"]), tuple(p["axis"]), p["half_angle"], orientation)
    if face.surface == "sphere":
        return Sphere(tuple(p["center"]), p["radius"], orientation)
    if face.surface == "torus":
        return Torus(
            tuple(p["center"]),
            tuple(p["axis"]),
            p["major_radius"],
            p["minor_radius"],
            orientation,
        )
    raise NotImplementedError(f"oracle IR has no region for surface type {face.surface!r}")


def _residual(face: FaceInfo, tris: np.ndarray) -> Residual:
    pts = np.concatenate([tris.reshape(-1, 3), tris.mean(axis=1)])
    d = distance_to_surface(face, pts)
    return Residual(float(np.sqrt(np.mean(d**2))), float(d.max()))


class _Directed:
    def __init__(self, adj: EdgeAdjacency):
        fwd = adj.forward_in_a
        self.points = adj.points if fwd else adj.points[::-1]
        self.start = adj.start_vertex if fwd else adj.end_vertex
        self.end = adj.end_vertex if fwd else adj.start_vertex
        self.dihedral = math.degrees(adj.dihedral)


def _walk(first, outgoing, used, roles):
    chain = [first]
    used.add(id(first))
    while chain[-1].end not in roles:
        nxt = [e for e in outgoing[chain[-1].end] if id(e) not in used]
        if not nxt:
            break
        used.add(id(nxt[0]))
        chain.append(nxt[0])
    return chain


def _merge(chain):
    pts = list(chain[0].points)
    for e in chain[1:]:
        pts.extend(e.points[1:])
    return pts, float(np.median([e.dihedral for e in chain]))


def build_oracle_ir(mesh: LabeledMesh, tangent_threshold_deg: float = TANGENT_THRESHOLD_DEG) -> Ir:
    regions = []
    for face in mesh.faces:
        tris = mesh.face_tris(face.id)
        regions.append(
            Region(
                face.id,
                _surface(face),
                np.nonzero(mesh.face_id == face.id)[0].tolist(),
                _residual(face, tris),
            )
        )

    def kind_of(deg: float) -> str:
        return "tangent" if deg < tangent_threshold_deg else "transversal"

    by_pair: dict[tuple[int, int], list[_Directed]] = defaultdict(list)
    faces_at: dict[int, set[int]] = defaultdict(set)
    ends_at: dict[int, list[str]] = defaultdict(list)
    for adj in mesh.adjacency:
        d = _Directed(adj)
        by_pair[(adj.face_a, adj.face_b)].append(d)
        kind = kind_of(d.dihedral)
        for v in (adj.start_vertex, adj.end_vertex):
            faces_at[v].update((adj.face_a, adj.face_b))
            ends_at[v].append(kind)

    roles: dict[int, str] = {}
    for v, faces in faces_at.items():
        if len(faces) >= 3:
            roles[v] = "junction"
        elif len(faces) == 2 and sorted(ends_at[v]) == ["tangent", "transversal"]:
            roles[v] = "kind_change"

    order = sorted(roles, key=lambda v: tuple(mesh.vertices[v]))
    vertex_id = {v: i for i, v in enumerate(order)}
    vertices = [
        Vertex(
            vertex_id[v],
            roles[v],
            tuple(mesh.vertices[v]),
            sorted(faces_at[v]),
            [tuple(mesh.vertices[v])],
        )
        for v in order
    ]

    adjacencies = []
    for pair in sorted(by_pair):
        edges = by_pair[pair]
        outgoing: dict[int, list[_Directed]] = defaultdict(list)
        for e in edges:
            outgoing[e.start].append(e)
        used: set[int] = set()
        boundaries = []

        for e in edges:
            if id(e) in used or e.start not in roles:
                continue
            chain = _walk(e, outgoing, used, roles)
            pts, deg = _merge(chain)
            boundaries.append(
                Boundary(
                    kind_of(deg),
                    deg,
                    False,
                    vertex_id[chain[0].start],
                    vertex_id[chain[-1].end],
                    [tuple(p) for p in pts],
                )
            )
        for e in edges:
            if id(e) in used:
                continue
            chain = _walk(e, outgoing, used, roles)
            pts, deg = _merge(chain)
            if pts[0] == pts[-1]:
                pts = pts[:-1]
            boundaries.append(
                Boundary(kind_of(deg), deg, True, None, None, [tuple(p) for p in pts])
            )
        boundaries.sort(key=lambda b: b.points[0])
        adjacencies.append(Adjacency(pair, boundaries))

    if mesh.shells:
        shells = [Shell(s.closed, s.role, s.parent, list(s.faces)) for s in mesh.shells]
    else:
        shells = [Shell(True, "outer", None, [f.id for f in mesh.faces])]

    verts, _, _, _ = unmesh.weld(mesh.tris, 1e-6)
    return Ir(
        Tolerances(linear=mesh.linear_deflection, tangent_threshold_deg=tangent_threshold_deg),
        Source(len(mesh.tris), len(verts)),
        shells,
        regions,
        adjacencies,
        vertices,
    )
