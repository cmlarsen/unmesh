from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

METRICS = (
    ("f1", "F1"),
    ("mean_triangle_iou", "IoU"),
    ("dev_input_p99", "deviation p99 (mm)"),
    ("valid", "validity"),
    ("under_report", "under-report"),
)

RATE_METRICS = ("valid", "under_report")

WORST_N = 20
VIEWER_MAX_TRIS = 15000


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    import math

    return float(value) if math.isfinite(value) else None


def _strata(record: dict[str, Any]) -> dict[str, Any]:
    strata = record.get("strata")
    return strata if isinstance(strata, dict) else {}


def _average_metrics(rs: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"n": len(rs)}
    for name, _ in METRICS:
        if name in RATE_METRICS:
            row[name] = sum(1 for r in rs if r.get(name) is True) / len(rs)
            continue
        vals = [metric_value(r, name) for r in rs]
        vals = [v for v in vals if v is not None]
        if vals:
            row[name] = sum(vals) / len(vals)
    return row


def heatmap_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        s = _strata(r)
        key = (
            r.get("converter", "?"),
            str(r.get("family", "?")),
            str(s.get("face_bucket", "?")),
            str(s.get("feature_bucket", "?")),
            r["operator"],
            float(r["severity"]),
        )
        groups.setdefault(key, []).append(r)
    rows = []
    for (converter, family, face_bucket, feature_bucket, operator, severity), rs in sorted(
        groups.items()
    ):
        row = _average_metrics(rs)
        row.update(
            {
                "converter": converter,
                "family": family,
                "face_bucket": face_bucket,
                "feature_bucket": feature_bucket,
                "operator": operator,
                "severity": severity,
            }
        )
        rows.append(row)
    return rows


def heatmap_overall(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        key = (r.get("converter", "?"), r["operator"], float(r["severity"]))
        groups.setdefault(key, []).append(r)
    rows = []
    for (converter, operator, severity), rs in sorted(groups.items()):
        row = _average_metrics(rs)
        row.update({"converter": converter, "operator": operator, "severity": severity})
        rows.append(row)
    return rows


def curve_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[float]] = {}
    counts: dict[tuple, int] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        for name, _ in METRICS:
            v = metric_value(r, name)
            if v is None:
                continue
            key = (
                r.get("converter", "?"),
                str(r.get("family", "?")),
                r["operator"],
                float(r["severity"]),
                name,
            )
            groups.setdefault(key, []).append(v)
            counts[key] = counts.get(key, 0) + 1
    return [
        {
            "converter": c,
            "family": f,
            "operator": o,
            "severity": s,
            "metric": m,
            "mean": sum(v) / len(v),
            "n": len(v),
        }
        for (c, f, o, s, m), v in sorted(groups.items())
    ]


def metric_value(record: dict[str, Any], name: str) -> float | None:
    if name == "valid":
        return 1.0 if record.get("valid") is True else 0.0
    if name == "under_report":
        return 0.0 if record.get("under_report") is False else 1.0
    value = _num(record.get(name))
    if value is None:
        segmentation = record.get("segmentation")
        if isinstance(segmentation, dict):
            value = _num(segmentation.get(name))
    return value


def worst_parts(records: list[dict[str, Any]], n: int = WORST_N) -> list[dict[str, Any]]:
    ok = [r for r in records if r.get("status") == "ok" and _num(r.get("f1")) is not None]
    return sorted(ok, key=lambda r: float(r["f1"]))[:n]


def failed_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("status") not in ("ok", "skipped")]


def viewer_payload(
    part_id: str,
    steps: list[tuple[str, float]],
    seed: int,
    cache: Path,
    max_tris: int = VIEWER_MAX_TRIS,
) -> dict[str, Any] | None:
    import numpy as np

    from ..degrade import chain
    from ..labels import LabeledMesh

    labeled_path = cache / f"{part_id}.labeled.npz"
    if not labeled_path.is_file():
        return None
    try:
        degraded = chain(LabeledMesh.load(labeled_path), steps, seed)
        tris = np.asarray(degraded.tris, dtype=np.float64).reshape(-1, 3, 3)
    except Exception:
        return None
    n = len(tris)
    _, owner = np.unique(np.asarray(degraded.face_id).reshape(-1), return_inverse=True)
    stride = max(1, (n + max_tris - 1) // max_tris)
    take = np.arange(0, n, stride, dtype=np.int64)
    sub = tris[take]
    flat = sub.reshape(-1, 3)
    rounded = np.round(flat, 3)
    uniq, index = np.unique(rounded, axis=0, return_inverse=True)
    return {
        "v": [[float(a), float(b), float(c)] for a, b, c in uniq.tolist()],
        "t": index.reshape(-1, 3).tolist(),
        "r": [int(x) for x in owner[take].tolist()],
        "ntris": n,
    }


def build_html(
    records: list[dict[str, Any]],
    grid_name: str,
    viewers: dict[str, dict[str, Any]] | None = None,
) -> str:
    heat = heatmap_rows(records)
    curves = curve_rows(records)
    worst = worst_parts(records)
    viewers = viewers or {}
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    converters = sorted({str(r.get("converter", "?")) for r in records})
    data = {
        "grid": grid_name,
        "converters": converters,
        "cells": len(records),
        "ok": len(ok),
        "skipped": len(skipped),
        "heat": heat,
        "heat_all": heatmap_overall(records),
        "curves": curves,
        "worst": [
            {
                "part": r["part"],
                "family": str(r.get("family", "?")),
                "operator": r["operator"],
                "severity": r["severity"],
                "seed": r["seed"],
                "converter": r.get("converter", "?"),
                "f1": _num(r.get("f1")),
                "dev_input_p99": _num(r.get("dev_input_p99")),
                "valid": r.get("valid"),
            }
            for r in worst
        ],
        "failed": [
            {
                "part": r["part"],
                "family": str(r.get("family", "?")),
                "operator": r.get("operator", "?"),
                "severity": r.get("severity", "?"),
                "seed": r.get("seed", "?"),
                "converter": r.get("converter", "?"),
                "status": r.get("status", "?"),
                "error": str(r.get("error", ""))[:300],
            }
            for r in failed_cells(records)[:WORST_N]
        ],
        "failed_total": len(failed_cells(records)),
        "viewers": {
            viewer_key(r): viewers[viewer_key(r)] for r in worst if viewer_key(r) in viewers
        },
    }
    payload = json.dumps(data, sort_keys=True).replace("<", "\\u003c")
    return _PAGE.replace("__DATA__", payload).replace("__GRID__", html.escape(grid_name))


def viewer_key(record: dict[str, Any]) -> str:
    return f"{record['part']}|{record['operator']}|{float(record['severity']):g}|{record['seed']}"


_PAGE = """<!DOCTYPE html>
<meta charset="utf-8">
<title>unmesh harness report __GRID__</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}
table{border-collapse:collapse;margin:1em 0;font-size:13px}
th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}
th{background:#f2f2f2}
td.op,th.op{text-align:left}
select,button{margin:0 .4em .6em 0}
canvas.mesh{border:1px solid #999;background:#111;cursor:grab}
.card{border:1px solid #ddd;margin:1em 0;padding:1em}
.mono{font-family:ui-monospace,monospace}
</style>
<h1>unmesh harness report <span id="grid"></span></h1>
<p id="meta"></p>
<h2>Heatmaps: operator x severity</h2>
<p>
<label>converter <select id="f-conv"></select></label>
<label>family <select id="f-fam"></select></label>
<label>face bucket <select id="f-face"></select></label>
<label>feature bucket <select id="f-feat"></select></label>
</p>
<div id="heats"></div>
<h2>Breaking-point curves</h2>
<p><label>metric <select id="c-metric"></select></label></p>
<div id="curves"></div>
<h2>Worst parts by F1</h2>
<div id="worst"></div>
<h2>Timeouts and errors</h2>
<div id="failed"></div>
<script>
"use strict";
const DATA = __DATA__;
const METRICS = [
  "f1", "mean_triangle_iou", "dev_input_p99", "valid", "under_report"
];
const HIGHER_BETTER = {
  "f1": 1, "mean_triangle_iou": 1, "valid": 1,
  "dev_input_p99": 0, "under_report": 0
};
function uniq(rows, key) {
  return [...new Set(rows.map((r) => String(r[key])))].sort();
}
function fill(sel, vals) {
  const el = document.getElementById(sel);
  el.innerHTML = "";
  for (const v of ["(all)", ...vals]) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    el.appendChild(o);
  }
}
function curFilters() {
  const g = (id) => document.getElementById(id).value;
  const pairs = [
    ["f-conv", "converter"], ["f-fam", "family"],
    ["f-face", "face_bucket"], ["f-feat", "feature_bucket"]
  ];
  const f = {};
  for (const [id, k] of pairs) {
    if (g(id) !== "(all)") f[k] = g(id);
  }
  return f;
}
function matchFilt(r, f) {
  for (const k in f) {
    if (String(r[k]) !== f[k]) return false;
  }
  return true;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
function weightedMean(rows, metric) {
  let num = 0, den = 0;
  for (const r of rows) {
    if (r[metric] === undefined || r[metric] === null) continue;
    const n = r.n || 1;
    num += r[metric] * n;
    den += n;
  }
  return den ? num / den : undefined;
}
function cellColor(metric, v, lo, hi) {
  let t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
  if (!HIGHER_BETTER[metric]) t = 1 - t;
  const r = Math.round(215 - 170 * t);
  const g = Math.round(50 + 150 * t);
  const b = Math.round(60 + 40 * t);
  return "background:rgb(" + r + "," + g + "," + b + ")";
}
function fmt(metric, v) {
  if (v === undefined || v === null) return "-";
  if (metric === "dev_input_p99") return (v * 1000).toFixed(1);
  if (metric === "f1" || metric === "mean_triangle_iou") {
    return v.toFixed(3);
  }
  return v.toFixed(2);
}
function renderHeats() {
  const f = curFilters();
  const div = document.getElementById("heats");
  div.innerHTML = "";
  const unfiltered = Object.keys(f).length === 0 && DATA.heat_all;
  for (const metric of METRICS) {
    let table;
    if (unfiltered) {
      table = DATA.heat_all
        .filter((r) => r[metric] !== undefined)
        .map((r) => ({operator: r.operator, severity: r.severity, value: r[metric]}));
    } else {
      const rows = DATA.heat.filter(
        (r) => matchFilt(r, f) && r[metric] !== undefined
      );
      if (!rows.length) continue;
      const ops = uniq(rows, "operator");
      const sevs = [...new Set(rows.map((r) => r.severity))];
      sevs.sort((a, b) => a - b);
      table = [];
      for (const o of ops) {
        for (const s of sevs) {
          const v = weightedMean(
            rows.filter((r) => r.operator === o && r.severity === s), metric
          );
          if (v !== undefined) table.push({operator: o, severity: s, value: v});
        }
      }
    }
    if (!table.length) continue;
    const ops = [...new Set(table.map((c) => c.operator))].sort();
    const sevs = [...new Set(table.map((c) => c.severity))];
    sevs.sort((a, b) => a - b);
    const vals = table.map((c) => c.value);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    let h = "<table><tr><th class=op>operator / severity</th>";
    for (const s of sevs) h += "<th>" + esc(s) + "</th>";
    h += "</tr>";
    for (const o of ops) {
      h += "<tr><td class=op>" + esc(o) + "</td>";
      for (const s of sevs) {
        const hit = table.find((c) => c.operator === o && c.severity === s);
        if (hit) {
          const style = cellColor(metric, hit.value, lo, hi);
          h += '<td style="' + style + '">';
          h += esc(fmt(metric, hit.value)) + "</td>";
        } else {
          h += "<td>-</td>";
        }
      }
      h += "</tr>";
    }
    const d = document.createElement("div");
    d.innerHTML = "<h3>" + esc(metric) + "</h3>" + h + "</table>";
    div.appendChild(d);
  }
}
function curveSvg(o, m, famRows, fams, sevs) {
  const vals = famRows.map((r) => r.mean);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi <= lo) hi = lo + 1;
  const W = 520, H = 200, P = 44;
  const span = sevs[sevs.length - 1] - sevs[0] || 1;
  const X = (s) => P + (sevs.length < 2 ? W / 2 : ((s - sevs[0]) / span) * (W - P - 8));
  const Y = (v) => H - 28 - ((v - lo) / (hi - lo)) * (H - 60);
  let s = '<svg width="' + W + '" height="' + H + '">';
  s += '<text x="6" y="16">min ' + fmt(m, lo) + "</text>";
  s += '<text x="6" y="' + (H - 30) + '">max ' + fmt(m, hi) + "</text>";
  fams.forEach((fam, i) => {
    const hue = (i * 137) % 360;
    const pts = famRows
      .filter((r) => r.family === fam)
      .sort((a, b) => a.severity - b.severity);
    const pl = pts
      .map((p) => X(p.severity).toFixed(1) + "," + Y(p.mean).toFixed(1))
      .join(" ");
    s += '<polyline points="' + pl + '" fill="none" ';
    s += 'stroke="hsl(' + hue + ',70%,40%)" stroke-width="2"/>';
    for (const p of pts) {
      s += '<circle cx="' + X(p.severity).toFixed(1) + '" ';
      s += 'cy="' + Y(p.mean).toFixed(1) + '" r="3" ';
      s += 'fill="hsl(' + hue + ',70%,40%)"><title>' + esc(fam) + " ";
      s += p.severity + ": " + fmt(m, p.mean) + "</title></circle>";
    }
  });
  return s + "</svg>";
}
function renderCurves() {
  const conv = document.getElementById("f-conv").value;
  const fam = document.getElementById("f-fam").value;
  const m = document.getElementById("c-metric").value;
  const div = document.getElementById("curves");
  div.innerHTML = "";
  const rows = DATA.curves.filter(
    (r) => r.metric === m
      && (conv === "(all)" || String(r.converter) === conv)
      && (fam === "(all)" || String(r.family) === fam)
  );
  for (const o of uniq(rows, "operator")) {
    const famRows = rows.filter((r) => r.operator === o);
    const fams = uniq(famRows, "family");
    const sevs = [...new Set(famRows.map((r) => r.severity))];
    sevs.sort((a, b) => a - b);
    const d = document.createElement("div");
    const h3 = document.createElement("h3");
    h3.textContent = o + " (" + m + ")";
    d.appendChild(h3);
    const wrap = document.createElement("div");
    wrap.innerHTML = curveSvg(o, m, famRows, fams, sevs);
    d.appendChild(wrap);
    const p = document.createElement("p");
    p.textContent = fams.join(", ");
    d.appendChild(p);
    div.appendChild(d);
  }
}
function rotMatrix(ax, ay) {
  const ca = Math.cos(ax), sa = Math.sin(ax);
  const cb = Math.cos(ay), sb = Math.sin(ay);
  return [[cb, 0, sb], [sa * sb, ca, -sa * cb], [-ca * sb, sa, ca * cb]];
}
function projectAll(mesh, M, cx, cy, cz) {
  const P = new Array(mesh.v.length);
  const box = [1e9, -1e9, 1e9, -1e9];
  for (let i = 0; i < mesh.v.length; i++) {
    const v = mesh.v[i];
    const x = v[0] - cx, y = v[1] - cy, z = v[2] - cz;
    const px = M[0][0] * x + M[0][1] * y + M[0][2] * z;
    const py = M[1][0] * x + M[1][1] * y + M[1][2] * z;
    const pz = M[2][0] * x + M[2][1] * y + M[2][2] * z;
    P[i] = [px, py, pz];
    if (px < box[0]) box[0] = px;
    if (px > box[1]) box[1] = px;
    if (py < box[2]) box[2] = py;
    if (py > box[3]) box[3] = py;
  }
  return [P, box];
}
function drawViewer(cv, mesh, ax, ay, scale, c) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);
  const [P, box] = projectAll(mesh, rotMatrix(ax, ay), c[0], c[1], c[2]);
  const sc = Math.min((W - 20) / (box[1] - box[0] || 1),
    (H - 20) / (box[3] - box[2] || 1)) * scale;
  const T = mesh.t;
  const order = T.map((t, i) => i).sort((a, b) => {
    const ta = T[a], tb = T[b];
    const za = (P[ta[0]][2] + P[ta[1]][2] + P[ta[2]][2]) / 3;
    const zb = (P[tb[0]][2] + P[tb[1]][2] + P[tb[2]][2]) / 3;
    return za - zb;
  });
  for (const i of order) {
    const t = T[i];
    ctx.beginPath();
    for (let k = 0; k < 3; k++) {
      const sx = W / 2 + P[t[k]][0] * sc;
      const sy = H / 2 - P[t[k]][1] * sc;
      if (k === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.closePath();
    ctx.fillStyle = "hsl(" + ((mesh.r[i] * 137) % 360) + ",65%,45%)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,.55)";
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }
}
function meshCenter(mesh) {
  const n = mesh.v.length;
  const c = [0, 0, 0];
  for (const v of mesh.v) {
    c[0] += v[0];
    c[1] += v[1];
    c[2] += v[2];
  }
  return [c[0] / n, c[1] / n, c[2] / n];
}
function renderWorst() {
  const div = document.getElementById("worst");
  div.innerHTML = "";
  DATA.worst.forEach((w, i) => {
    const key = w.part + "|" + w.operator + "|" + w.severity + "|" + w.seed;
    const d = document.createElement("div");
    d.className = "card";
    const f1 = w.f1 === null ? "-" : w.f1.toFixed(3);
    const head = document.createElement("div");
    const rank = document.createElement("b");
    rank.textContent = (i + 1) + ". " + w.part;
    const detail = document.createElement("span");
    detail.textContent = " " + w.converter + " " + w.operator + "@" + w.severity +
      " seed " + w.seed + " F1 " + f1;
    head.appendChild(rank);
    head.appendChild(detail);
    d.appendChild(head);
    const mesh = DATA.viewers[key];
    if (mesh) {
      const cv = document.createElement("canvas");
      cv.width = 460;
      cv.height = 340;
      cv.className = "mesh";
      d.appendChild(document.createElement("br"));
      d.appendChild(cv);
      const c = meshCenter(mesh);
      let ax = 0.6, ay = 0.8;
      drawViewer(cv, mesh, ax, ay, 1.0, c);
      let drag = null;
      cv.addEventListener("mousedown", (e) => {
        drag = [e.clientX, e.clientY, ax, ay];
      });
      cv.addEventListener("mousemove", (e) => {
        if (!drag) return;
        ay = drag[3] + (e.clientX - drag[0]) * 0.01;
        ax = drag[2] + (e.clientY - drag[1]) * 0.01;
        drawViewer(cv, mesh, ax, ay, 1.0, c);
      });
      cv.addEventListener("mouseup", () => {
        drag = null;
      });
      cv.addEventListener("mouseleave", () => {
        drag = null;
      });
      const p = document.createElement("p");
      p.textContent = "drag to rotate: input mesh colored by ground-truth faces (" +
        mesh.r.length + " tris shown of " + mesh.ntris + ")";
      d.appendChild(p);
    } else {
      const p = document.createElement("p");
      p.textContent = "mesh unavailable for viewer";
      d.appendChild(p);
    }
    div.appendChild(d);
  });
}
document.getElementById("grid").textContent = DATA.grid;
document.getElementById("meta").textContent = DATA.cells + " cells, " +
  DATA.ok + " ok, " + (DATA.skipped || 0) + " skipped, " +
  (DATA.failed_total || 0) + " failed, converters: " +
  DATA.converters.join(", ");
fill("f-conv", uniq(DATA.heat, "converter"));
fill("f-fam", uniq(DATA.heat, "family"));
fill("f-face", uniq(DATA.heat, "face_bucket"));
fill("f-feat", uniq(DATA.heat, "feature_bucket"));
const cm = document.getElementById("c-metric");
for (const m of METRICS) {
  const o = document.createElement("option");
  o.value = m;
  o.textContent = m;
  cm.appendChild(o);
}
for (const id of ["f-conv", "f-fam", "f-face", "f-feat"]) {
  document.getElementById(id).addEventListener("change", () => {
    renderHeats();
    renderCurves();
  });
}
cm.addEventListener("change", renderCurves);
renderHeats();
renderCurves();
renderWorst();
renderFailed();
function renderFailed() {
  const div = document.getElementById("failed");
  div.innerHTML = "";
  const rows = DATA.failed || [];
  const note = document.createElement("p");
  note.textContent = (DATA.failed_total || 0) + " failed cells" +
    (rows.length < (DATA.failed_total || 0)
      ? ", showing first " + rows.length : "") + ".";
  div.appendChild(note);
  if (!rows.length) return;
  const t = document.createElement("table");
  const head = document.createElement("tr");
  for (const c of ["part", "converter", "operator", "severity", "seed", "status", "error"]) {
    const th = document.createElement("th");
    th.textContent = c;
    head.appendChild(th);
  }
  t.appendChild(head);
  for (const r of rows) {
    const tr = document.createElement("tr");
    for (const c of ["part", "converter", "operator", "severity", "seed", "status", "error"]) {
      const td = document.createElement("td");
      td.textContent = String(r[c]);
      if (c === "part" || c === "operator" || c === "status") td.className = "op";
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
  div.appendChild(t);
}
</script>
"""
