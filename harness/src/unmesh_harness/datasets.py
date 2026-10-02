from __future__ import annotations

import argparse
import csv
import hashlib
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

THINGI_REDISTRIBUTABLE = {
    "Public Domain": "Public-Domain",
    "Creative Commons - Public Domain Dedication": "CC0-1.0",
    "Creative Commons - Attribution": "CC-BY",
    "BSD License": "BSD",
}
THINGI_DOWNLOAD_ONLY = {
    "Creative Commons - Attribution - Share Alike": "CC-BY-SA",
    "GNU - GPL": "GPL",
    "GNU - LGPL": "LGPL",
    "Creative Commons - Attribution - Non-Commercial": "CC-BY-NC",
    "Attribution - Non-Commercial - Share Alike": "CC-BY-NC-SA",
    "Attribution - Non-Commercial - No Derivatives": "CC-BY-NC-ND",
    "Creative Commons - Attribution - No Derivatives": "CC-BY-ND",
}

NIST_URL = "https://www.nist.gov/system/files/documents/noindex/2024/06/19/NIST-PMI-STEP-Files.zip"
NIST_SHA256 = "8fa78429e6d8d9b0d7681d223b6aa9ec98c3772185c55b1a0e3679b21c181911"
FREECAD_REPO = "FreeCAD/FreeCAD-library"
FREECAD_COMMIT = "544a254e090eaf7bfbb6a9b69e249dc0d3d29d67"
FREECAD_MAX_BYTES = 5_000_000
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
FUSION_URL = (
    "https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/"
    "segmentation/s2.0.1/s2.0.1_extended_step.zip"
)
FUSION_SIZE = 506333119
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
    terms_key: str | None = None


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
            "https://raw.githubusercontent.com/FreeCAD/FreeCAD-library/master/LICENSE-Assets",
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
            "each thing in the dataset has its own license; refer to the license field",
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
        Dataset(
            "fusion360-segmentation",
            "Fusion 360 Gallery segmentation, extended STEP",
            DOWNLOAD_ONLY,
            "Autodesk Fusion 360 Gallery Dataset License (non-commercial research)",
            "https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md",
            "You may access, use, reproduce and modify the Dataset, in each case, only for "
            "non-commercial research purposes.",
            "https://github.com/AutodeskAILab/Fusion360GalleryDataset",
            False,
            terms_key="fusion360-gallery",
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


class RangeFile(io.RawIOBase):
    def __init__(self, size: int, fetch: Callable[[int, int], bytes]):
        self._size = size
        self._fetch = fetch
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self._size - self._pos)
        if n <= 0:
            return 0
        data = self._fetch(self._pos, self._pos + n - 1)
        buf[: len(data)] = data
        self._pos += len(data)
        return len(data)


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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")
        return path

    @staticmethod
    def read(root: Path) -> dict[str, Any]:
        return json.loads((root / MANIFEST_NAME).read_text())


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
        try:
            with _open(url, headers) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
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
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
            continue
        if expected_size is None or part.stat().st_size == expected_size:
            break
    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(f"{url}: expected {expected_size} bytes, got {part.stat().st_size}")
    part.replace(dest)


def ranged_zip(url: str, expected_size: int | None = None) -> zipfile.ZipFile:
    with _open(url, {"Range": "bytes=0-0"}) as r:
        total = int(r.headers["Content-Range"].rpartition("/")[2])
    if expected_size is not None and total != expected_size:
        raise RuntimeError(f"{url}: expected {expected_size} bytes, server reports {total}")

    def fetch(start: int, end: int) -> bytes:
        return http_bytes(url, {"Range": f"bytes={start}-{end}"})

    return zipfile.ZipFile(io.BufferedReader(RangeFile(total, fetch), 1 << 22))


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


def fetch_nist_pmi(ds: Dataset, root: Path, limit: int, **_: Any) -> Manifest:
    manifest = _manifest_for(ds, limit=limit)
    archive = root / "_archive" / "NIST-PMI-STEP-Files.zip"
    if not archive.is_file() or sha256_file(archive) != NIST_SHA256:
        http_download(NIST_URL, archive)
    if sha256_file(archive) != NIST_SHA256:
        raise RuntimeError("NIST archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith((".stp", ".step")))[:limit]
        for name in names:
            dest = root / "step" / Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            manifest.entries.append(
                {
                    "id": f"nist-pmi/{dest.stem}",
                    "license": ds.license,
                    "license_tier": ds.tier,
                    "source_url": NIST_URL,
                    "files": {"step": file_record(root, dest)},
                }
            )
    shutil.rmtree(root / "_archive")
    return manifest


def fetch_freecad_library(ds: Dataset, root: Path, limit: int, **_: Any) -> Manifest:
    manifest = _manifest_for(ds, limit=limit, commit=FREECAD_COMMIT)
    api = f"https://api.github.com/repos/{FREECAD_REPO}/git/trees/{FREECAD_COMMIT}?recursive=1"
    tree = json.loads(http_bytes(api, {"Accept": "application/vnd.github+json"}))
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing truncated")
    raw = f"https://raw.githubusercontent.com/{FREECAD_REPO}/{FREECAD_COMMIT}/"
    for pair in round_robin(select_freecad_pairs(tree["tree"]), limit):
        index = len(manifest.entries)
        files = {}
        for kind in ("step", "stl"):
            node = pair[kind]
            dest = root / kind / f"{index:04d}_{Path(node['path']).name}"
            if not dest.is_file():
                data = http_bytes(raw + urllib.parse.quote(node["path"]))
                if git_blob_sha1(data) != node["sha"]:
                    raise RuntimeError(f"git blob checksum mismatch for {node['path']}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            files[kind] = file_record(root, dest, source_path=node["path"], git_blob=node["sha"])
        manifest.entries.append(
            {
                "id": f"freecad-library/{pair['id']}",
                "license": "CC-BY-3.0",
                "license_tier": ds.tier,
                "attribution": "FreeCAD-library contributors; see git history at the pinned commit",
                "source_url": f"https://github.com/{FREECAD_REPO}/blob/{FREECAD_COMMIT}/"
                + urllib.parse.quote(pair["step"]["path"]),
                "files": files,
            }
        )
    return manifest


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
        dest = root / "stl" / f"{item['file_id']}.stl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest.entries.append(
            {
                "id": f"thingi10k/{item['file_id']}",
                "license": item["license"],
                "license_text": item["license_text"],
                "license_tier": item["tier"],
                "attribution": f"{item['name']} by {item['author']}",
                "category": item["category"],
                "source_url": f"https://www.thingiverse.com/thing:{item['thing_id']}",
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


def fetch_fusion360(ds: Dataset, root: Path, limit: int, **_: Any) -> Manifest:
    manifest = _manifest_for(ds, limit=limit, archive_bytes=FUSION_SIZE)
    z = ranged_zip(FUSION_URL, FUSION_SIZE)
    names = sorted(i.filename for i in z.infolist() if i.filename.endswith(".stp"))[:limit]
    for name in names:
        dest = root / "step" / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(name))
        manifest.entries.append(
            {
                "id": f"fusion360-segmentation/{dest.stem}",
                "license": ds.license,
                "license_tier": ds.tier,
                "source_url": ds.source_url,
                "files": {"step": file_record(root, dest)},
            }
        )
    return manifest


FETCHERS: dict[str, Callable[..., Manifest]] = {
    "nist-pmi": fetch_nist_pmi,
    "freecad-library": fetch_freecad_library,
    "thingi10k": fetch_thingi10k,
    "abc": fetch_abc,
    "fusion360-segmentation": fetch_fusion360,
}


def check_terms(ds: Dataset, accepted: set[str]) -> None:
    if ds.terms_key and ds.terms_key not in accepted:
        raise SystemExit(
            f"{ds.name}: license is {ds.license!r} ({ds.license_url}). "
            f"Review it, then pass --accept-terms {ds.terms_key}"
        )


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
    p.add_argument("--accept-terms", action="append", default=[])
    p.add_argument(
        "--license-tier",
        choices=[REDISTRIBUTABLE, DOWNLOAD_ONLY],
        default=REDISTRIBUTABLE,
        help="thingi10k: also include download-only licenses (SA, GPL, NC, ND)",
    )
    p.add_argument("--category", action="append", default=[], help="thingi10k category filter")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--verify", action="store_true", help="re-hash cached files against manifests")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for ds in DATASETS.values():
            flag = "default" if ds.default else "opt-in"
            print(f"{ds.name:24} {ds.tier:16} {flag:8} {ds.license}")
        return 0

    names = args.dataset or ["all"]
    if "all" in names:
        names = [n for n in names if n != "all"] + [d.name for d in DATASETS.values() if d.default]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        p.error(f"unknown dataset(s): {', '.join(unknown)}")
    base = (args.cache_dir or default_datasets_dir()).expanduser()

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
        check_terms(ds, set(args.accept_terms))
        root.mkdir(parents=True, exist_ok=True)
        manifest = FETCHERS[name](
            ds,
            root,
            args.limit,
            license_tier=args.license_tier,
            categories=set(args.category) or None,
            keep_archives=args.keep_archives,
        )
        path = manifest.write(root)
        total = sum(f["bytes"] for e in manifest.entries for f in e["files"].values())
        print(f"{name}: {len(manifest.entries)} entries, {total / 1e6:.1f} MB, manifest {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
