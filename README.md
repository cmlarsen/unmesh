# unmesh

Turn a triangle mesh exported from CAD back into a STEP solid with real analytic faces (planes,
cylinders, cones, spheres, tori) and a measured fidelity report.

Early development. See [docs/plan.md](docs/plan.md).

## Use

```sh
pip install unmesh[step]
unmesh convert part.stl part.step --report part.json
```

`--unit in` reads inch coordinates (the STEP is always in mm), `--tolerance` sets the linear
tolerance in the input's unit, and OBJ input works by its `.obj` suffix. The exit code says what was
written:

| code | meaning |
|---|---|
| 0 | every face analytic |
| 1 | analytic faces plus triangle (`facets`) regions |
| 2 | a whole-part faceted solid (fallback) |
| 3 | error: nothing usable written |

The JSON report records the measured deviation to the input mesh overall and per face, the faces left
faceted and why, validity and verification, and runtime per stage. Its
schema is in [docs/api.md](docs/api.md#fidelity-report) and
[docs/fidelity.schema.json](docs/fidelity.schema.json). From Python, `unmesh.convert_to_step(mesh,
path)` does the same.

## Layout

- `crates/unmesh-core`: the Rust geometry core.
- `crates/unmesh-py`: PyO3 bindings, built as `unmesh._core`.
- `python/unmesh`: the Python package. `pip install unmesh[step]` adds the STEP writer's OCP dependency.
- `harness/`: the separate `unmesh-harness` evaluation package.

## Develop

```sh
uv sync
scripts/check.sh
```

See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0
