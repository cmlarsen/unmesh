# Datasets for the imported-STEP tier, the real-exporter row and the hold-out

Decision rule (owner, [#22](https://github.com/cmlarsen/unmesh/issues/22)):

- **Redistributable** (may be committed or published as derived meshes): CC0, CC-BY, MIT, BSD, Apache or public domain only.
- **Download-only** (fetched by `scripts/fetch-datasets`, never committed): any other license that allows research or benchmark use.
- **Excluded**: terms forbid automated download or benchmark use, or the license cannot be established. A license that is unclear is excluded.

Nothing in this document is committed as data. `scripts/fetch-datasets` writes to `$UNMESH_CACHE_DIR/datasets` or `~/.cache/unmesh/datasets`, one directory per dataset id, each with a `manifest.json` (per-file path, bytes, sha256, license, source URL, attribution). Evidence was read on 2026-10-02.

## Decision table

| Dataset | Contents | License (operative text) | Tier | Decision |
| --- | --- | --- | --- | --- |
| [NIST MBE PMI test models](#nist-mbe-pmi-test-models) | 33 STEP AP242 files (CTC/FTC/STC), mechanical, 54 MB | "can be used without any restrictions" | redistributable | **Use** (`nist-pmi`) |
| [FreeCAD parts library](#freecad-parts-library) | 1,970 STEP+STL pairs under `Mechanical Parts/` (of 2,814 STEP, 2,553 STL), mechanical | CC-BY-3.0 | redistributable | **Use** (`freecad-library`) |
| [Thingi10K](#thingi10k) | 10,000 STL (3,142 CC0/CC-BY/PD/BSD, 6,854 other open licenses), mixed mechanical and organic, no STEP | per file, Thingiverse license field | per file | **Use** (`thingi10k`), redistributable tier by default |
| [ABC](#abc) | 1M STEP, plus Parasolid and STL, Onshape public documents, mostly mechanical | copyright with creators, Onshape ToU 1.g.ii | download-only | **Use**, opt-in (`abc`) |
| [Fusion 360 Gallery](#fusion-360-gallery) | 42,912 STEP (segmentation, extended), 8,625 reconstruction sequences | non-commercial research only | download-only | **Use**, opt-in, needs `--accept-terms` (`fusion360-segmentation`) |
| [DeepCAD](#deepcad) | 178k construction-sequence JSON, no STEP | data license not stated | n/a | Excluded: unclear, and derived from ABC |
| [MCB](#mcb-mechanical-components-benchmark) | 58,696 STEP/STL/OFF components | code MIT, data license not stated | n/a | Excluded: unclear, scraped from third-party sites |
| [CADParser](#cadparser) | ~40,000 STEP | no license found | n/a | Excluded: unclear |
| [Objaverse](#objaverse) | 800k+ glTF objects | ODC-By overall, per-object CC | n/a | Not used: no STEP, mostly organic |
| [Thingiverse / Printables direct](#thingiverse-and-printables-direct) | per-model downloads | per model | n/a | Excluded: no sanctioned bulk access; Thingi10K is the sanctioned archive |

## Sources

### NIST MBE PMI test models

- Source: <https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/mbe-pmi-0>. File fetched: <https://www.nist.gov/system/files/documents/noindex/2024/06/19/NIST-PMI-STEP-Files.zip> (14 MB, sha256 pinned in the script).
- License: "The test cases, CAD models, and STEP files can be used without any restrictions." (NIST Disclaimer, same page.) NIST is a US government agency, so public domain or unrestricted. NIST asks for acknowledgment.
- Contents: 33 STEP AP242/AP203 files, the CTC, FTC and STC test parts (machined prismatic parts with PMI), 11 PDF drawings. All mechanical. Few parts (33), so it adds breadth, not volume.
- Automated download: a single static file link. NIST rejects requests without a `User-Agent` header (HTTP 403), so the script sends one.
- Decision: use. Redistributable.

### FreeCAD parts library

- Source: <https://github.com/FreeCAD/FreeCAD-library>, pinned to commit `544a254e090eaf7bfbb6a9b69e249dc0d3d29d67`.
- License ([LICENSE-Assets](https://raw.githubusercontent.com/FreeCAD/FreeCAD-library/master/LICENSE-Assets)): "All of it is licensed under the Creative Commons Attribution 3.0 Unported license (SPDX identifier: CC-BY-3.0)", covering "the FreeCAD documents (.FCStd), their exported counterparts (.stp/.step, .brp/.brep, .stl, .dxf, .wrl)". "Each part is copyrighted by its own author, not by the FreeCAD project." Attribution is by git history, which the manifest points at.
- Contents: 1,970 STEP+STL pairs (same directory and stem) under `Mechanical Parts/` of at most 5 MB each: fasteners (1,027), EN profiles (895), bearings, mountings, pulleys, couplings, chain links. Simple to medium parts (3 to ~4,000 faces in the sample). Fasteners and profiles dominate the library, so `fetch-datasets` takes a round-robin over families to keep small families represented.
- STL provenance: exported by FreeCAD, which tessellates with OCCT. The STL is a pair for the STEP, but it is not independent of the OCCT-based truth tessellation the harness already uses.
- Automated download: `raw.githubusercontent.com` at the pinned commit. Each file is verified against its git blob sha1.
- Decision: use. Redistributable with attribution.

### Thingi10K

- Source: <https://github.com/Thingi10K/Thingi10K>, mirrored by the authors at <https://huggingface.co/datasets/Thingi10K/Thingi10K> (pinned to commit `2d5d3b2f3cd3711028ad75b12788c13b25559ec6`).
- License: "each thing in the dataset has its own license, refer to the license field" (README of the `thingi10k` package; the package code is Apache-2.0). Per-file license is in `metadata/input_summary.csv`, with author and name in `metadata/contextual_data.csv`. The HF dataset card carries no license tag of its own.
- Count by license over the 10,000 files: CC-BY-SA 3,680; CC-BY 2,945; CC-BY-NC 1,581; CC-BY-NC-SA 975; CC-BY-NC-ND 330; GPL 202; CC0 99; Public Domain 88; CC-BY-ND 84; BSD 10; LGPL 2; unknown 4. Redistributable (CC0, CC-BY, PD, BSD): 3,142 files, 2,481 of them closed and edge-manifold. Download-only: 6,854. Excluded (unknown): 4.
- Contents: STL meshes of 3D-printing models from Thingiverse, 2009 to 2015. Mixed mechanical and organic; Thingiverse categories are noisy (462 are "None"). No STEP. 9.6 GB for the full archive.
- Automated download: the full archive is one 9.6 GB tarball. `fetch-datasets` instead reads the per-file `npz/<id>.npz` (vertices and facets) from the pinned HF commit and writes a binary STL, so a subset costs only what it selects. The STL is rebuilt from the welded mesh, not the original Thingiverse bytes. The original S3 links in `input_summary.csv` are Thingiverse assets and are not used.
- Decision: use. The default selects only the redistributable tier and only closed, edge-manifold meshes; `--license-tier download-only` adds SA, GPL, NC and ND files. `unknown_license` is always excluded. CC-BY files need attribution, which the manifest carries (name, author, Thingiverse URL).

### ABC

- Source: <https://deep-geometry.github.io/abc-dataset/>. Chunks of 10,000 models on NYU: <https://archive.nyu.edu/handle/2451/61215>. MD5 and size per chunk are published (`md5.yml`, `size.yml`) and pinned in the script.
- License: "The copyright of the CAD models is owned by their creators. For licensing details, see Onshape Terms of Use 1.g.ii." Onshape ToU 1.g.ii ([terms](https://www.onshape.com/en/legal/terms-of-use)): "For any Public Document owned by a Free Plan User created on or after August 7, 2018, or any Public Document created prior to that date without a LICENSE tab, Customer grants a worldwide, royalty-free and non-exclusive license to any End User or third party accessing the Public Document to use the intellectual property contained in Customer's Public Document without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies". It continues: documents "which contained a tab called LICENSE reserving rights greater than the foregoing, those greater reserved rights will continue to apply". The ABC metadata does not say whether a document had a LICENSE tab or a Free Plan owner, so permissiveness cannot be verified per file.
- Contents: 1M models; formats `step`, `para`, `stl`/`stl2`, `obj`, `feat`, `meta`. Mostly mechanical CAD, many duplicates and trivial parts. STEP chunk 1.6 GB (10,000 models), STL chunk 9.4 GB.
- Automated download: the site documents `wget`/`curl` bulk download of chunks, so it is sanctioned. Onshape's own ToU 4.b.ix forbids scraping the Onshape service; that does not apply to the ABC archive hosted by NYU. A chunk is a solid 7z, so a subset still costs the whole 1.6 GB transfer; `fetch-datasets` downloads it, extracts a strided sample of K models plus their metadata, and deletes the archive (`--keep-archives` keeps it). Requires `7zz`/`7z`.
- Decision: download-only, not redistributable (per-file license unverifiable). The manifest records Onshape `createdAt` per model. Opt-in via `--dataset abc`.

### Fusion 360 Gallery

- Source: <https://github.com/AutodeskAILab/Fusion360GalleryDataset>.
- License ([LICENSE.md](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md)): "You may access, use, reproduce and modify the Dataset, in each case, only for non-commercial research purposes." Also: "You may not redistribute or make available to others the Dataset in its entirety" and modified portions only under the same non-commercial terms; "If you are employed by a for-profit, commercial entity, your employer shall also be bound by this License".
- Contents: reconstruction (8,625 sketch-and-extrude sequences, 2.0 GB), segmentation (35,680 parts, 3.1 GB), assembly (154,468 parts). The `s2.0.1_extended_step.zip` (483 MB) holds 42,912 STEP files with per-face modeling-operation labels. Mechanical-ish user designs, many simple.
- Automated download: public S3 URLs with HTTP Range support, linked from the README. `fetch-datasets` reads the zip directory by range requests and downloads only the K selected STEP files.
- Decision: download-only. Opt-in, and refuses to run without `--accept-terms fusion360-gallery`. See "Owner judgment" below.

### DeepCAD

- Source: <https://github.com/ChrisWu1997/DeepCAD>. Code license MIT. Data (<http://www.cs.columbia.edu/cg/deepcad/data.tar>, 208 MB): "The data we used are parsed from Onshape public documents with links from ABC dataset."
- Contents: construction-sequence JSON and vectors, no STEP or B-rep; getting geometry means replaying sequences with a kernel. The README states no data license.
- Decision: excluded. License of the data is unstated, and the geometry is a subset of ABC.

### MCB (Mechanical Components Benchmark)

- Source: <https://github.com/stnoah1/mcb>. The repository `LICENSE` is MIT and covers the code. The README says the models were collected "from online 3D CAD repositories" (its config names GrabCAD, 3D ContentCentral and TraceParts) and links the data on Box.
- Contents: 58,696 components in 68 classes, STEP/STL/OFF.
- Decision: excluded. The data license is not stated, the models come from third-party sites with their own terms, and the Box links are not a scriptable endpoint.

### CADParser

- Source: <https://github.com/spicywagyu04/CADParser> (reimplementation) and the paper. The ~40,000-model STEP data is shared on Google Drive.
- Decision: excluded. No license found for the data and no stable scriptable download.

### Objaverse

- Source: <https://huggingface.co/datasets/allenai/objaverse>. "The use of the dataset as a whole is licensed under the ODC-By v1.0 license." Per-object: CC-BY 721K, CC-BY-NC 25K, CC-BY-NC-SA 52K, CC-BY-SA 16K, CC0 3.5K.
- Contents: glTF/GLB, 8.9 TB, overwhelmingly organic and artistic; no STEP and little CAD-exported mechanical geometry.
- Decision: not used. The license metadata is good, but the content is the wrong domain.

### Thingiverse and Printables direct

- Thingiverse robots.txt disallows `/*/zip` and `/download:*`, and the site has no sanctioned bulk path; Thingi10K is the researchers' archive of the same material. Printables models are individually licensed, and its [terms](https://www.prusa3d.com/page/terms-of-service-of-printables-com_231249/) were read without finding a bulk-download grant. I did not find a scraper clause there either, but there is no API for bulk file access, so crawling is not pursued.
- Decision: excluded for automation. The owner can still curate individual CC0 or CC-BY models by hand for the hold-out (#13).

## Recommended use

**(a) Imported-STEP tier (#25).** Ground-truth STEP we tessellate ourselves.

- Always on: `freecad-library` (1,970 pairs available, simple to medium mechanical parts) and `nist-pmi` (33 machined parts). Both are redistributable, so manifests may name them and CI may fetch them. This alone gives well over the 200 imported parts #25 asks for.
- Opt-in for breadth and complexity: `abc` (complex, real-world, noisy) and `fusion360-segmentation` (labeled faces). Manifest entries reference them by dataset id and model id, and the standard grid must treat their absence as a skip, not a failure.
- Strata (face count, smallest feature) are computed from the STEP after download, not stored here.

**(b) Real-exporter row (#29).** STL written by a CAD tool, ideally with its STEP.

- `freecad-library` STL: true pairs, written by FreeCAD, which uses OCCT. Cheap and useful, but not independent of the kernel the harness tessellates with.
- `thingi10k`, redistributable tier: STL from many unknown CAD tools and slicers, no STEP. It is real-world mesh quality (sliver triangles, non-manifold edges), which is the point of the row. Pick `--category tools --category gadgets --category hobby --category household` for the functional share.
- ABC STL (`stl2`, Onshape/Parasolid tessellation, paired with ABC STEP) would be the best real pair set, but a chunk is 9.4 GB and not fetched here. If wanted, it is a follow-up using `--keep-archives` style handling.
- No set we can fetch has Fusion, SolidWorks or Inventor exports with their STEP. The owner's own exports remain the only source for those.

**(c) Private hold-out (#13).** CC0 or CC-BY only, unless the owner consents.

- Draw from `thingi10k` CC0/CC-BY/PD/BSD files (3,142, 2,481 clean). Curate 30 to 60 by hand from functional categories, store them in the private repo or bucket, and record the Thingiverse URL and author for attribution.
- Exclude every id that the agent-visible `standard` manifest uses. The hold-out must not be reachable through `fetch-datasets` output that agents run.
- Reference STEP does not exist for Thingi10K; the rubric path in #13 (measured dimensions, per-face check) applies.
- `freecad-library` is deliberately not hold-out material: it is the agent-visible imported tier.

## Owner judgment needed

1. **Fusion 360 Gallery is non-commercial research only**, and clause 11 binds a for-profit employer. unmesh is Apache-2.0 and public, but if it feeds a commercial product the benchmark use may not qualify. It is included as download-only behind `--accept-terms fusion360-gallery` and no code, test or manifest depends on it. Drop it if there is doubt.
2. **ABC license cannot be verified per file.** Onshape's grant is MIT-like for most public documents but not those with a LICENSE tab. Kept download-only. If the owner accepts the ToU grant as sufficient, it could be promoted to redistributable, but nothing needs it.
3. **Thingi10K NC and ND files** (CC-BY-NC, NC-SA, NC-ND, ND) allow use but not commercial use or derivatives. They are off by default and only fetched with `--license-tier download-only`. The same commercial-use question as Fusion applies.
4. **Thingi10K STL are rebuilt from the HF npz mesh**, not the original bytes. If bit-exact exporter output matters for #29, the 9.6 GB tarball is the alternative.

## Using the script

```sh
uv run scripts/fetch-datasets --list
uv run scripts/fetch-datasets --dataset all --limit 60
uv run scripts/fetch-datasets --dataset thingi10k --category tools --limit 100
uv run scripts/fetch-datasets --dataset abc --limit 50
uv run scripts/fetch-datasets --dataset fusion360-segmentation --accept-terms fusion360-gallery
uv run scripts/fetch-datasets --verify --dataset freecad-library
```

`all` means the datasets marked default (`nist-pmi`, `freecad-library`, `thingi10k`). Selection is deterministic: strided, round-robin over families, or ascending id, so the same `--limit` gives the same models for a pinned source version. `--verify` rehashes every file against its manifest.
