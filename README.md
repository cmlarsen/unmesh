# unmesh

Turn a triangle mesh exported from CAD back into a STEP solid with real analytic faces (planes,
cylinders, cones, spheres, tori) and a measured fidelity report.

Early development. See [docs/plan.md](docs/plan.md).

## Layout

- `crates/unmesh-core`: the Rust geometry core.

## Develop

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## License

Apache-2.0
