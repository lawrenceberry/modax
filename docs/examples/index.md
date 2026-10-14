# Examples

Worked end-to-end problems live in `examples/` (each with its own README):

- [`bbn_estimation/`](bbn-estimation.md) — toy Big Bang Nucleosynthesis network with nested-sampling
  parameter estimation and a modax/Diffrax/scipy solver benchmark;
- [`21cm_igm_evolution/`](21cm-igm-evolution.md) — toy global 21cm IGM thermal/ionisation history;
- [`mukhanov_sasaki/`](mukhanov-sasaki.md) — Mukhanov–Sasaki mode evolution.

Scaling, dimensionality, divergence, Jacobian-density and gradient benchmarks
are under `benchmarks/`, each a script that caches its timings in
`results.json` and writes a CSV and a plot named after the GPU. Every point
runs in a child process capped at 180 s, compilation included; one that
overruns is recorded as a timeout and left off the plot.

Each one runs on a GPU, compares against a Diffrax (and, where it is tractable,
a scipy) baseline built from the same right-hand side, and is checked without a
device by `tests/test_examples.py`.
