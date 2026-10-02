# unmesh

Turn a triangle mesh exported from CAD back into a STEP solid with real analytic faces (planes,
cylinders, cones, spheres, tori) and a measured fidelity report.

Early development. See [docs/plan.md](docs/plan.md).

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
