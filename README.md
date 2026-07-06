# HRT/Partcl Macro Placement Hackathon — Team Submission

A Quantum-Annealing-Inspired SA macro placer built for the [Partcl/HRT $20K competition](CHALLENGE.md).  
Competition deadline: **May 21, 2026**.

---

## Submission File

```
submissions/our_submission/qsa_contest.py
```

Self-contained single file — no external submission dependencies.

Run with the official evaluator:
```bash
uv run evaluate submissions/our_submission/qsa_contest.py -b ibm01
uv run evaluate submissions/our_submission/qsa_contest.py --all
```

---

## What We Built

We implemented a **QSA (Quantum-Annealing-Inspired Simulated Annealing)** placer that positions chip macros on a floorplan to minimise the competition proxy cost (wirelength + 0.5×density + 0.5×congestion).

**Algorithm highlights:**

1. **Smart initialisation** — starts from the benchmark's hand-crafted `macro_positions` layout, legalised via a tetris-style spiral search (same approach as rank-11 entry). Falls back to shelf-pack only if legalisation fails. Starts SA from a much better position than greedy methods.

2. **Vectorised HPWL** — padded net tensor allows fully batched HPWL computation (~38× faster than a Python net-loop). More SA moves per second within the same time budget.

3. **Proxy-cost objective** — SA directly minimises `WL + 0.5×D + 0.5×C`, i.e. the exact competition scoring metric, not just raw wirelength.

4. **Move mix** — 40% pairwise swaps + 60% random perturbations, with connectivity-weighted macro selection (macros in more nets are perturbed more often).

5. **Classical reheating** — up to 4 reheats on stall; restores best-known placement and raises temperature to escape local optima.

6. **Time-based density resync** — `compute_proxy_cost` is called every N wall-clock seconds (calibrated to 3× the measured call latency). Prevents ibm17-style stalls where a 166s/call evaluation would otherwise dominate runtime.

7. **plc auto-loading** — if the evaluator does not pass a `PlacementCost` object, the placer loads it from disk automatically, enabling full proxy-cost tracking in all run modes.

**Results (full 55-minute runs, all 17 IBM benchmarks):**

| Metric | Value |
|---|---|
| Average proxy cost | **1.5382** |
| vs SA baseline (2.1251) | +27.6% better |
| vs RePlAce baseline (1.4578) | −5.5% |
| Valid placements (0 overlaps) | 17 / 17 |
| Runtime per benchmark | ~3300 s (55 min) |

---

## Repository Layout

```
submissions/our_submission/
    qsa_contest.py     <- SUBMISSION FILE (self-contained)
    parallel_eval.py   <- local harness: run all 17 benchmarks in parallel
    quick_eval.py      <- local harness: quick iteration at reduced time budget

macro_place/           <- competition evaluation framework (do not modify)
    benchmark.py
    loader.py
    objective.py
    utils.py

external/MacroPlacement/ <- TILOS evaluator submodule
CHALLENGE.md             <- competition rules, prizes, baselines
SETUP.md                 <- competition API reference
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/KrishVenky/HRT_Partcl_Hackathon.git
cd HRT_Partcl_Hackathon
git submodule update --init external/MacroPlacement
uv sync
```

---

## Running

### Official evaluator (single benchmark)
```bash
uv run evaluate submissions/our_submission/qsa_contest.py -b ibm01
```

### All 17 benchmarks in parallel (4 workers, ~14 hours)
```bash
uv run python submissions/our_submission/parallel_eval.py --workers 4
```

### Quick sanity check (60s per benchmark)
```bash
uv run python submissions/our_submission/parallel_eval.py \
    --placer submissions/our_submission/qsa_contest.py \
    --workers 2 --time 60 -b ibm01 ibm10 ibm18
```

---

## Competition Context

**Proxy cost formula** (lower is better):
```
Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```

See [CHALLENGE.md](CHALLENGE.md) for full rules, prize structure, and leaderboard.
