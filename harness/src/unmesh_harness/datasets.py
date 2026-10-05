from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

USER_AGENT = "unmesh-fetch-datasets/0.1 (+https://github.com/cmlarsen/unmesh)"
REDISTRIBUTABLE = "redistributable"
DOWNLOAD_ONLY = "download-only"
MANIFEST_NAME = "manifest.json"
REQUEST_DELAY = 0.25

THINGI_REDISTRIBUTABLE = {
    "Public Domain": "Public-Domain",
    "Creative Commons - Public Domain Dedication": "CC0-1.0",
    "Creative Commons - Attribution": "CC-BY",
    "BSD License": "BSD",
}
THINGI_LICENSE_URLS = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-SA": "https://creativecommons.org/licenses/by-sa/",
    "CC-BY-ND": "https://creativecommons.org/licenses/by-nd/",
    "GPL": "https://www.gnu.org/licenses/gpl-3.0.html",
    "LGPL": "https://www.gnu.org/licenses/lgpl-3.0.html",
}
THINGI_DOWNLOAD_ONLY = {
    "Creative Commons - Attribution - Share Alike": "CC-BY-SA",
    "GNU - GPL": "GPL",
    "GNU - LGPL": "LGPL",
    "Creative Commons - Attribution - No Derivatives": "CC-BY-ND",
}

NIST_URL = "https://www.nist.gov/system/files/documents/noindex/2024/06/19/NIST-PMI-STEP-Files.zip"
NIST_SHA256 = "8fa78429e6d8d9b0d7681d223b6aa9ec98c3772185c55b1a0e3679b21c181911"
FREECAD_REPO = "FreeCAD/FreeCAD-library"
FREECAD_COMMIT = "544a254e090eaf7bfbb6a9b69e249dc0d3d29d67"
FREECAD_MAX_BYTES = 5_000_000
IMPORTED_TIER_FETCH = {
    "nist-pmi": {"limit": 60},
    "freecad-library": {"limit": 200},
}
IMPORTED_TIER_COMMAND = "uv run scripts/fetch-datasets --manifest corpus/v0.json"
ABC_ARCHIVES = {
    "step": (
        "https://archive.nyu.edu/rest/bitstreams/88598/retrieve",
        1594129754,
        "695388be7a278c7798c8c8ae239772ac",
    ),
    "meta": (
        "https://archive.nyu.edu/rest/bitstreams/88595/retrieve",
        594388,
        "7a489538af88b2788b19e916ccdcabbc",
    ),
}
THINGI_COMMIT = "2d5d3b2f3cd3711028ad75b12788c13b25559ec6"
THINGI_BASE = f"https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/{THINGI_COMMIT}"


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    tier: str
    license: str
    license_url: str
    license_quote: str
    source_url: str
    default: bool


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset(
            "nist-pmi",
            "NIST MBE PMI test models (STEP)",
            REDISTRIBUTABLE,
            "NIST: use without restriction",
            "https://www.nist.gov/ctl/smart-connected-systems-division/"
            "smart-connected-manufacturing-systems-group/mbe-pmi-0",
            "The test cases, CAD models, and STEP files can be used without any restrictions.",
            NIST_URL,
            True,
        ),
        Dataset(
            "freecad-library",
            "FreeCAD parts library (STEP + STL pairs)",
            REDISTRIBUTABLE,
            "CC-BY-3.0",
            f"https://raw.githubusercontent.com/{FREECAD_REPO}/{FREECAD_COMMIT}/LICENSE-Assets",
            "All of it is licensed under the Creative Commons Attribution 3.0 Unported license "
            "(SPDX identifier: CC-BY-3.0)",
            f"https://github.com/{FREECAD_REPO}/tree/{FREECAD_COMMIT}",
            True,
        ),
        Dataset(
            "thingi10k",
            "Thingi10K (STL, per-file license)",
            REDISTRIBUTABLE,
            "per file (Thingiverse license field)",
            "https://github.com/Thingi10K/Thingi10K",
            'Each "thing" in the dataset has its own license. Please refer to the `license` '
            "field associated with each entry in the dataset.",
            "https://huggingface.co/datasets/Thingi10K/Thingi10K",
            True,
        ),
        Dataset(
            "abc",
            "ABC dataset chunk 0 (STEP)",
            DOWNLOAD_ONLY,
            "per model (Onshape public document terms)",
            "https://www.onshape.com/en/legal/terms-of-use",
            "The copyright of the CAD models is owned by their creators. For licensing details, "
            "see Onshape Terms of Use 1.g.ii.",
            "https://deep-geometry.github.io/abc-dataset/",
            False,
        ),
    ]
}


def default_datasets_dir() -> Path:
    override = os.environ.get("UNMESH_CACHE_DIR")
    if override:
        return Path(override) / "datasets"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "unmesh" / "datasets"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


_TMP_SUFFIX = ".unmesh-tmp"


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + _TMP_SUFFIX)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _clean_stale_tmp_files(root: Path) -> None:
    for tmp in root.rglob("*" + _TMP_SUFFIX):
        try:
            tmp.unlink()
        except OSError:
            pass


def classify_thingi_license(name: str) -> tuple[str, str | None]:
    if name in THINGI_REDISTRIBUTABLE:
        return REDISTRIBUTABLE, THINGI_REDISTRIBUTABLE[name]
    if name in THINGI_DOWNLOAD_ONLY:
        return DOWNLOAD_ONLY, THINGI_DOWNLOAD_ONLY[name]
    return "excluded", None


def license_allowed(file_tier: str, wanted: str) -> bool:
    if file_tier == REDISTRIBUTABLE:
        return True
    return file_tier == DOWNLOAD_ONLY and wanted == DOWNLOAD_ONLY


def stl_bytes(vertices: np.ndarray, facets: np.ndarray) -> bytes:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(facets, dtype=np.int64)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    return b"unmesh fetch-datasets".ljust(80, b"\0") + struct.pack("<I", len(f)) + rec.tobytes()


def select_freecad_pairs(
    tree: Iterable[dict[str, Any]], max_bytes: int = FREECAD_MAX_BYTES
) -> list[dict[str, Any]]:
    blobs = {t["path"]: t for t in tree if t.get("type") == "blob"}
    pairs = []
    for path, node in blobs.items():
        stem, dot, ext = path.rpartition(".")
        if not dot or ext.lower() not in ("step", "stp"):
            continue
        if not path.startswith("Mechanical Parts/"):
            continue
        stl = next((p for p in (f"{stem}.stl", f"{stem}.STL") if p in blobs), None)
        if stl is None:
            continue
        if max(node.get("size", 0), blobs[stl].get("size", 0)) > max_bytes:
            continue
        pairs.append({"id": stem, "step": node, "stl": blobs[stl]})
    pairs.sort(key=lambda p: p["id"])
    return pairs


def select_thingi(
    summary_rows: Iterable[dict[str, str]],
    context_rows: Iterable[dict[str, str]],
    wanted_tier: str,
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    context = {r["Thing ID"]: r for r in context_rows}
    picked = []
    for row in summary_rows:
        tier, spdx = classify_thingi_license(row["License"])
        if spdx is None or not license_allowed(tier, wanted_tier):
            continue
        if row.get("Closed") != "TRUE" or row.get("Edge manifold") != "TRUE":
            continue
        ctx = context.get(row["Thing ID"], {})
        if categories and ctx.get("Category") not in categories:
            continue
        picked.append(
            {
                "file_id": row["ID"],
                "thing_id": row["Thing ID"],
                "license": spdx,
                "license_text": row["License"],
                "tier": tier,
                "name": ctx.get("Name", "").strip(),
                "author": ctx.get("Author", ""),
                "category": ctx.get("Category", ""),
            }
        )
    picked.sort(key=lambda p: int(p["file_id"]))
    return picked


@dataclass
class Manifest:
    dataset: str
    tier: str
    license: str
    license_url: str
    license_quote: str
    source_url: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        self.tier = overall_tier(self.entries, self.tier)

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "tier": self.tier,
            "license": self.license,
            "license_url": self.license_url,
            "license_quote": self.license_quote,
            "source_url": self.source_url,
            "params": self.params,
            "entries": self.entries,
        }

    def write(self, root: Path) -> Path:
        path = root / MANIFEST_NAME
        body = (json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n").encode()
        _atomic_write_bytes(path, body)
        return path

    @staticmethod
    def read(root: Path) -> dict[str, Any]:
        return json.loads((root / MANIFEST_NAME).read_text())


def overall_tier(entries: Iterable[dict[str, Any]], default: str) -> str:
    tiers = {e["license_tier"] for e in entries if "license_tier" in e}
    if not tiers:
        return default
    return REDISTRIBUTABLE if tiers == {REDISTRIBUTABLE} else DOWNLOAD_ONLY


def file_record(root: Path, path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def verify_manifest(root: Path) -> list[str]:
    problems = []
    manifest = Manifest.read(root)
    for entry in manifest["entries"]:
        for kind, rec in entry["files"].items():
            path = root / rec["path"]
            if not path.is_file():
                problems.append(f"{entry['id']}/{kind}: missing {rec['path']}")
            elif path.stat().st_size != rec["bytes"] or sha256_file(path) != rec["sha256"]:
                problems.append(f"{entry['id']}/{kind}: checksum mismatch {rec['path']}")
    return problems


def _open(url: str, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=60)


def http_bytes(url: str, headers: dict[str, str] | None = None, retries: int = 3) -> bytes:
    for attempt in range(retries):
        time.sleep(REQUEST_DELAY)
        try:
            with _open(url, headers) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ):
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def http_download(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        have = part.stat().st_size if part.exists() else 0
        if expected_size is not None and have == expected_size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _open(url, headers) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(part, mode) as f:
                    shutil.copyfileobj(r, f, 1 << 20)
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
            continue
        if expected_size is None or part.stat().st_size == expected_size:
            break
    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(f"{url}: expected {expected_size} bytes, got {part.stat().st_size}")
    part.replace(dest)


def _manifest_for(ds: Dataset, **params: Any) -> Manifest:
    return Manifest(
        ds.name,
        ds.tier,
        ds.license,
        ds.license_url,
        ds.license_quote,
        ds.source_url,
        params=params,
    )


def _nist_archive(root: Path) -> Path:
    archive = root / "_archive" / "NIST-PMI-STEP-Files.zip"
    if not archive.is_file() or sha256_file(archive) != NIST_SHA256:
        http_download(NIST_URL, archive)
    if sha256_file(archive) != NIST_SHA256:
        raise RuntimeError("NIST archive checksum mismatch")
    return archive


def _nist_entry(ds: Dataset, root: Path, dest: Path) -> dict[str, Any]:
    return {
        "id": f"nist-pmi/{dest.stem}",
        "license": ds.license,
        "license_tier": ds.tier,
        "source_url": NIST_URL,
        "files": {"step": file_record(root, dest)},
    }


def fetch_nist_pmi(ds: Dataset, root: Path, limit: int, **_: Any) -> Manifest:
    manifest = _manifest_for(ds, limit=limit)
    archive = _nist_archive(root)
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith((".stp", ".step")))[:limit]
        for name in names:
            dest = root / "step" / Path(name).name
            _atomic_write_bytes(dest, z.read(name))
            manifest.entries.append(_nist_entry(ds, root, dest))
    shutil.rmtree(root / "_archive")
    return manifest


def _prior_entries(root: Path) -> list[dict[str, Any]]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        return []
    try:
        record = Manifest.read(root)
    except ValueError as e:
        print(
            f"warning: ignoring unreadable cache manifest {path}: {e}; rebuilding",
            file=sys.stderr,
        )
        return []
    entries = record.get("entries", []) if isinstance(record, dict) else None
    if not isinstance(entries, list):
        print(f"warning: ignoring malformed cache manifest {path}; rebuilding", file=sys.stderr)
        return []
    return entries


def _cached_step_ok(root: Path, entry: dict[str, Any] | None, sha: str) -> bool:
    if entry is None:
        return False
    rec = entry.get("files", {}).get("step")
    if rec is None:
        return False
    dest = root / rec["path"]
    return dest.is_file() and sha256_file(dest) == sha


def _step_record_for(root: Path, dest: Path, prior: dict[str, Any] | None, **extra: Any):
    kept = {k: v for k, v in (prior or {}).items() if k not in ("path", "bytes", "sha256")}
    kept.update(extra)
    return file_record(root, dest, **kept)


def _kept_entry(root: Path, prior: dict[str, Any], sha: str, **extra: Any) -> dict[str, Any]:
    dest = root / prior["files"]["step"]["path"]
    entry = dict(prior)
    files = dict(prior.get("files", {}))
    files["step"] = _step_record_for(root, dest, prior["files"]["step"], **extra)
    entry["files"] = files
    return entry


def _with_preserved_extras(
    manifest: Manifest, prior: list[dict[str, Any]], wanted: set[str]
) -> Manifest:
    manifest.entries.extend(e for e in prior if e.get("id") not in wanted)
    return manifest


def fetch_nist_refs(ds: Dataset, root: Path, wanted: dict[str, str], **_: Any) -> Manifest:
    manifest = _manifest_for(ds, mode="manifest", refs=len(wanted))
    _clean_stale_tmp_files(root)
    prior = _prior_entries(root)
    by_id = {e["id"]: e for e in prior if "id" in e}
    missing = sorted(
        fid for fid in wanted if not _cached_step_ok(root, by_id.get(fid), wanted[fid])
    )
    fetched: dict[str, bytes] = {}
    names: dict[str, str] = {}
    if missing:
        stems = {fid.split("/", 1)[1]: fid for fid in wanted}
        archive = _nist_archive(root)
        with zipfile.ZipFile(archive) as z:
            members = sorted(n for n in z.namelist() if n.lower().endswith((".stp", ".step")))
            found: dict[str, str] = {}
            for name in members:
                found.setdefault(Path(name).stem, name)
            absent = [fid for stem, fid in stems.items() if stem not in found]
            if absent:
                raise RuntimeError(
                    f"nist-pmi: {len(absent)} manifest file(s) vanished upstream: {absent[:5]}"
                )
            for fid in missing:
                stem = fid.split("/", 1)[1]
                names[fid] = Path(found[stem]).name
                fetched[fid] = z.read(found[stem])
        shutil.rmtree(root / "_archive", ignore_errors=True)
    for fid in sorted(wanted):
        prior_entry = by_id.get(fid)
        if fid in fetched:
            if prior_entry is not None:
                dest = root / prior_entry["files"]["step"]["path"]
            else:
                dest = root / "step" / names[fid]
            _atomic_write_bytes(dest, fetched[fid])
            if sha256_file(dest) != wanted[fid]:
                raise RuntimeError(f"{fid}: sha256 mismatch against corpus manifest")
            base = prior_entry or _nist_entry(ds, root, dest)
            manifest.entries.append(_kept_entry(root, base, wanted[fid]))
        else:
            manifest.entries.append(_kept_entry(root, prior_entry, wanted[fid]))
    return _with_preserved_extras(manifest, prior, set(wanted))


def _freecad_tree() -> list[dict[str, Any]]:
    api = f"https://api.github.com/repos/{FREECAD_REPO}/git/trees/{FREECAD_COMMIT}?recursive=1"
    tree = json.loads(http_bytes(api, {"Accept": "application/vnd.github+json"}))
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing truncated")
    return tree["tree"]


def _download_freecad_blob(node: dict[str, Any]) -> bytes:
    raw = f"https://raw.githubusercontent.com/{FREECAD_REPO}/{FREECAD_COMMIT}/"
    data = http_bytes(raw + urllib.parse.quote(node["path"]))
    if git_blob_sha1(data) != node["sha"]:
        raise RuntimeError(f"git blob checksum mismatch for {node['path']}")
    return data


def fetch_freecad_library(ds: Dataset, root: Path, limit: int, **_: Any) -> Manifest:
    manifest = _manifest_for(ds, limit=limit, commit=FREECAD_COMMIT)
    tree = _freecad_tree()
    raw = f"https://raw.githubusercontent.com/{FREECAD_REPO}/{FREECAD_COMMIT}/"
    for pair in round_robin(select_freecad_pairs(tree), limit):
        index = len(manifest.entries)
        files = {}
        for kind in ("step", "stl"):
            node = pair[kind]
            dest = root / kind / f"{index:04d}_{Path(node['path']).name}"
            if not dest.is_file():
                data = http_bytes(raw + urllib.parse.quote(node["path"]))
                if git_blob_sha1(data) != node["sha"]:
                    raise RuntimeError(f"git blob checksum mismatch for {node['path']}")
                _atomic_write_bytes(dest, data)
            files[kind] = file_record(root, dest, source_path=node["path"], git_blob=node["sha"])
        manifest.entries.append(
            {
                "id": f"freecad-library/{pair['id']}",
                "license": "CC-BY-3.0",
                "license_tier": ds.tier,
                "attribution": None,
                "attribution_note": "author unresolved: take it from the git history of "
                + pair["step"]["path"]
                + " at the pinned commit before publishing",
                "source_url": f"https://github.com/{FREECAD_REPO}/blob/{FREECAD_COMMIT}/"
                + urllib.parse.quote(pair["step"]["path"]),
                "files": files,
            }
        )
    return manifest


def _fresh_freecad_dest(root: Path, node: dict[str, Any], taken: set[str]) -> Path:
    base = Path(node["path"]).name
    index = 0
    while True:
        name = f"{index:04d}_{base}"
        if name not in taken and not (root / "step" / name).exists():
            taken.add(name)
            return root / "step" / name
        index += 1


def fetch_freecad_refs(ds: Dataset, root: Path, wanted: dict[str, str], **_: Any) -> Manifest:
    manifest = _manifest_for(ds, mode="manifest", commit=FREECAD_COMMIT, refs=len(wanted))
    _clean_stale_tmp_files(root)
    prior = _prior_entries(root)
    by_id = {e["id"]: e for e in prior if "id" in e}
    missing = sorted(
        fid for fid in wanted if not _cached_step_ok(root, by_id.get(fid), wanted[fid])
    )
    fetched: dict[str, bytes] = {}
    nodes: dict[str, dict[str, Any]] = {}
    if missing:
        pairs = {p["id"]: p for p in select_freecad_pairs(_freecad_tree())}
        absent = sorted(fid for fid in wanted if fid.partition("/")[2] not in pairs)
        if absent:
            raise RuntimeError(
                f"freecad-library: {len(absent)} manifest file(s) vanished upstream: {absent[:5]}"
            )
        for fid in missing:
            node = pairs[fid.partition("/")[2]]["step"]
            nodes[fid] = node
            fetched[fid] = _download_freecad_blob(node)
    taken = {Path(e["files"]["step"]["path"]).name for e in prior if "step" in e.get("files", {})}
    for fid in sorted(wanted):
        prior_entry = by_id.get(fid)
        if fid in fetched:
            node = nodes[fid]
            if prior_entry is not None:
                dest = root / prior_entry["files"]["step"]["path"]
            else:
                dest = _fresh_freecad_dest(root, node, taken)
            _atomic_write_bytes(dest, fetched[fid])
            if sha256_file(dest) != wanted[fid]:
                raise RuntimeError(f"{fid}: sha256 mismatch against corpus manifest")
            base: dict[str, Any] = (
                dict(prior_entry)
                if prior_entry is not None
                else {
                    "id": fid,
                    "license": "CC-BY-3.0",
                    "license_tier": ds.tier,
                    "attribution": None,
                    "attribution_note": "author unresolved: take it from the git history of "
                    + node["path"]
                    + " at the pinned commit before publishing",
                    "source_url": f"https://github.com/{FREECAD_REPO}/blob/{FREECAD_COMMIT}/"
                    + urllib.parse.quote(node["path"]),
                    "files": {},
                }
            )
            files = {k: v for k, v in base.get("files", {}).items() if k != "step"}
            files["step"] = _step_record_for(
                root, dest, None, source_path=node["path"], git_blob=node["sha"]
            )
            base["files"] = files
            manifest.entries.append(_kept_entry(root, base, wanted[fid]))
        else:
            manifest.entries.append(_kept_entry(root, prior_entry, wanted[fid]))
    return _with_preserved_extras(manifest, prior, set(wanted))


def fetch_thingi10k(
    ds: Dataset,
    root: Path,
    limit: int,
    license_tier: str = REDISTRIBUTABLE,
    categories: set[str] | None = None,
    **_: Any,
) -> Manifest:
    manifest = _manifest_for(
        ds,
        limit=limit,
        commit=THINGI_COMMIT,
        license_tier=license_tier,
        categories=sorted(categories or []),
    )
    summary = http_bytes(f"{THINGI_BASE}/metadata/input_summary.csv").decode()
    context = http_bytes(f"{THINGI_BASE}/metadata/contextual_data.csv").decode()
    picked = select_thingi(
        csv.DictReader(io.StringIO(summary)),
        csv.DictReader(io.StringIO(context)),
        license_tier,
        categories,
    )
    for item in picked:
        if len(manifest.entries) >= limit:
            break
        try:
            blob = http_bytes(f"{THINGI_BASE}/npz/{item['file_id']}.npz")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        with np.load(io.BytesIO(blob)) as z:
            data = stl_bytes(z["vertices"], z["facets"])
        thing_url = f"https://www.thingiverse.com/thing:{item['thing_id']}"
        dest = root / "stl" / f"{item['file_id']}.stl"
        _atomic_write_bytes(dest, data)
        manifest.entries.append(
            {
                "id": f"thingi10k/{item['file_id']}",
                "license": item["license"],
                "license_url": THINGI_LICENSE_URLS.get(item["license"], thing_url),
                "license_version": None,
                "license_text": item["license_text"],
                "license_tier": item["tier"],
                "attribution": f"{item['name']} by {item['author']}",
                "category": item["category"],
                "source_url": thing_url,
                "modified": "rebuilt as binary STL from the Thingi10K npz mesh",
                "files": {
                    "stl": file_record(
                        root, dest, source_npz_sha256=hashlib.sha256(blob).hexdigest()
                    )
                },
            }
        )
    return manifest


def strided(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return []
    stride = max(1, len(items) // limit)
    return items[::stride][:limit]


def round_robin(pairs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        family = "/".join(pair["id"].split("/")[:2])
        groups.setdefault(family, []).append(pair)
    queues = [strided(groups[k], len(groups[k])) for k in sorted(groups)]
    out: list[dict[str, Any]] = []
    depth = 0
    while len(out) < limit and any(depth < len(q) for q in queues):
        out.extend(q[depth] for q in queues if depth < len(q))
        depth += 1
    return sorted(out[:limit], key=lambda p: p["id"])


def meta_field(text: str, key: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(f"{key}:"):
            value = line.partition(":")[2].strip()
            return None if value in ("", "null") else value
    return None


def parse_sevenzip_listing(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        if line.startswith("Path = "):
            names.append(line[len("Path = ") :])
    return names


def _sevenzip() -> str:
    for exe in ("7zz", "7z", "7za"):
        found = shutil.which(exe)
        if found:
            return found
    raise RuntimeError("7zz (or 7z) is required to read ABC archives; install 7-zip")


def _abc_archive(root: Path, kind: str) -> Path:
    url, size, md5 = ABC_ARCHIVES[kind]
    dest = root / "_archive" / f"abc_0000_{kind}_v00.7z"
    if not dest.is_file():
        http_download(url, dest, size)
    if md5_file(dest) != md5:
        dest.rename(dest.with_name(dest.name + ".bad"))
        raise RuntimeError(f"{dest.name}: md5 mismatch (expected {md5})")
    return dest


def _sevenzip_list(archive: Path) -> list[str]:
    out = subprocess.run(
        [_sevenzip(), "l", "-slt", str(archive)], check=True, capture_output=True, text=True
    ).stdout
    return parse_sevenzip_listing(out.split("----------", 1)[1])


def _sevenzip_extract(archive: Path, names: list[str], dest: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        lst.write("\n".join(names))
    try:
        subprocess.run(
            [_sevenzip(), "x", "-y", f"-o{dest}", str(archive), f"@{lst.name}"],
            check=True,
            capture_output=True,
        )
    finally:
        os.unlink(lst.name)


def fetch_abc(
    ds: Dataset, root: Path, limit: int, keep_archives: bool = False, **_: Any
) -> Manifest:
    manifest = _manifest_for(ds, limit=limit, chunk="abc_0000_v00")
    step_archive = _abc_archive(root, "step")
    meta_archive = _abc_archive(root, "meta")
    shutil.rmtree(root / "step", ignore_errors=True)
    shutil.rmtree(root / "meta", ignore_errors=True)
    steps = sorted(n for n in _sevenzip_list(step_archive) if n.endswith(".step"))
    steps = strided(steps, limit)
    ids = {Path(n).parent.name for n in steps}
    metas = [
        n for n in _sevenzip_list(meta_archive) if n.endswith(".yml") and Path(n).parent.name in ids
    ]
    _sevenzip_extract(step_archive, steps, root / "step")
    _sevenzip_extract(meta_archive, metas, root / "meta")
    meta_by_id = {Path(n).parent.name: n for n in metas}
    for name in steps:
        model = Path(name).parent.name
        files = {"step": file_record(root, root / "step" / name)}
        extra: dict[str, Any] = {}
        if model in meta_by_id:
            meta_path = root / "meta" / meta_by_id[model]
            files["meta"] = file_record(root, meta_path)
            text = meta_path.read_text()
            extra = {
                "onshape_created_at": meta_field(text, "createdAt"),
                "onshape_name": meta_field(text, "name"),
            }
        manifest.entries.append(
            {
                "id": f"abc/{model}",
                "license": "per model: Onshape public document terms; not recorded per file",
                "license_tier": ds.tier,
                "source_url": ds.source_url,
                **extra,
                "files": files,
            }
        )
    if not keep_archives:
        shutil.rmtree(root / "_archive")
    return manifest


FETCHERS: dict[str, Callable[..., Manifest]] = {
    "nist-pmi": fetch_nist_pmi,
    "freecad-library": fetch_freecad_library,
    "thingi10k": fetch_thingi10k,
    "abc": fetch_abc,
}

REFETCHERS: dict[str, Callable[..., Manifest]] = {
    "nist-pmi": fetch_nist_refs,
    "freecad-library": fetch_freecad_refs,
}


def wanted_imported_refs(corpus_manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    for entry in corpus_manifest.get("entries", []):
        if entry.get("tier") != "imported" or "source" not in entry:
            continue
        src = entry["source"]
        refs.setdefault(src["dataset"], {})[src["file_id"]] = src["sha256"]
    return refs


def fetch_manifest_refs(corpus_manifest: dict[str, Any], base: Path) -> dict[str, int]:
    refs = wanted_imported_refs(corpus_manifest)
    unknown = [n for n in refs if n not in REFETCHERS or n not in DATASETS]
    if unknown:
        raise RuntimeError(
            f"--manifest cannot fetch dataset(s): {', '.join(sorted(unknown))}; "
            "only the imported-tier datasets are rebuildable "
            f"({', '.join(sorted(REFETCHERS))})"
        )
    counts = {}
    for name in sorted(refs):
        root = base / name
        root.mkdir(parents=True, exist_ok=True)
        manifest = REFETCHERS[name](DATASETS[name], root, refs[name])
        manifest.finalize()
        path = manifest.write(root)
        total = sum(f["bytes"] for e in manifest.entries for f in e["files"].values())
        print(f"{name}: {len(manifest.entries)} entries, {total / 1e6:.1f} MB, manifest {path}")
        counts[name] = len(manifest.entries)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="fetch-datasets",
        description="Download pinned subsets of open CAD/mesh datasets into a local cache.",
    )
    p.add_argument(
        "--dataset", action="append", default=[], help="dataset name or 'all' (defaults)"
    )
    p.add_argument("--limit", type=int, default=25, help="models per dataset (stable id order)")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument(
        "--license-tier",
        choices=[REDISTRIBUTABLE, DOWNLOAD_ONLY],
        default=REDISTRIBUTABLE,
        help="thingi10k: also include download-only licenses (SA, GPL, LGPL, ND)",
    )
    p.add_argument("--category", action="append", default=[], help="thingi10k category filter")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--verify", action="store_true", help="re-hash cached files against manifests")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="corpus manifest (e.g. corpus/v0.json): fetch exactly the imported "
        "STEP files it references, ignoring --dataset/--limit",
    )
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for ds in DATASETS.values():
            flag = "default" if ds.default else "opt-in"
            print(f"{ds.name:24} {ds.tier:16} {flag:8} {ds.license}")
        return 0

    base = (args.cache_dir or default_datasets_dir()).expanduser()

    if args.manifest is not None:
        if args.verify:
            p.error("--manifest cannot be combined with --verify")
        corpus = json.loads(args.manifest.read_text())
        fetch_manifest_refs(corpus, base)
        return 0

    names = args.dataset or ["all"]
    if "all" in names:
        names = [n for n in names if n != "all"] + [d.name for d in DATASETS.values() if d.default]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        p.error(f"unknown dataset(s): {', '.join(unknown)}")

    status = 0
    for name in dict.fromkeys(names):
        ds = DATASETS[name]
        root = base / name
        if args.verify:
            problems = verify_manifest(root)
            print(f"{name}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
            for problem in problems:
                print(f"  {problem}")
            status |= bool(problems)
            continue
        root.mkdir(parents=True, exist_ok=True)
        manifest = FETCHERS[name](
            ds,
            root,
            args.limit,
            license_tier=args.license_tier,
            categories=set(args.category) or None,
            keep_archives=args.keep_archives,
        )
        manifest.finalize()
        path = manifest.write(root)
        total = sum(f["bytes"] for e in manifest.entries for f in e["files"].values())
        print(f"{name}: {len(manifest.entries)} entries, {total / 1e6:.1f} MB, manifest {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
