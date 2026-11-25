# MOCAT-6RD with pySSEM

Reproduce the model from *Effects of Orbit Raising and Deorbiting in Source-Sink
Evolutionary Models* using pySSEM, then extend the launch-rate terms.

## Build / run

```sh
./build.sh              # C++ release build  (debug | --clean | --test | --python)
./build.sh --python     # also set up .venv with pySSEM
./run.sh                # run C++ executable (build/bin/mocat6rd)
./run.sh py run_baseline.py
```

## Layout

- `cpp/` – C++ core (`include/mocat/`, `src/`, `apps/`, `tests/`)
- `python/mocat6rd/` – pySSEM wrapper + launch/orbit-raising/deorbit terms
- `python/scripts/` – entry-point scripts
- `configs/` – scenario configs
- `data/`, `results/` – inputs and outputs
