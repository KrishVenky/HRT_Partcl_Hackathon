# HRT/Partcl Macro Placement Hackathon — Team Submission

A Simulated Annealing macro placer built for the [Partcl/HRT $20K competition](CHALLENGE.md).  
Competition deadline: **May 21, 2026**.

---

## What We Built

We implemented a custom **Simulated Annealing (SA)** placer that positions chip macros (large memory blocks, IPs, etc.) on a floorplan to minimise wirelength, density, and routing congestion.

**Algorithm highlights:**
- Greedy shelf-pack initialisation — guaranteed zero overlaps from the start
- Two move types: random perturbation (60%) and pairwise swap (40%)
- Connectivity-biased macro selection — macros in more nets are moved more often
- Time-based temperature schedule — robust across all 17 benchmarks regardless of design size
- Periodic reheating (up to 5×) to escape local optima
- Full 55-minute budget per benchmark (competition allows 1 hour each)

**Score target:** Beat the SA baseline (avg `2.1251`) and approach RePlAce (`1.4578`) on all 17 IBM benchmarks. First confirmed run on ibm01 (60s, 1.8% of budget): proxy cost `1.4187`, zero overlaps.

---

## Repository Layout

```
submissions/our_submission/
    sa_placer.py       ← our placer (what gets submitted)
    compare.py         ← harness to compare multiple placers side-by-side

macro_place/           ← competition evaluation framework (do not modify)
    benchmark.py       ← Benchmark dataclass (PyTorch tensors)
    loader.py          ← loads IBM/NG45 benchmarks from disk
    objective.py       ← proxy cost computation (WL + density + congestion)
    utils.py           ← validate_placement, visualize_placement

submissions/examples/  ← reference placers provided by competition organisers
external/MacroPlacement/ ← TILOS evaluator submodule (git submodule)

CHALLENGE.md           ← original competition README (prizes, rules, baselines)
SETUP.md               ← competition API reference
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone this repo
git clone https://github.com/KrishVenky/HRT_Partcl_Hackathon.git
cd HRT_Partcl_Hackathon

# 2. Pull the TILOS evaluator (needed to load benchmarks and compute cost)
git submodule update --init external/MacroPlacement

# 3. Install dependencies
uv sync
```

---

## Running the Placer

### Quick validity check on ibm01 (~1 min)

```bash
uv run python - << 'EOF'
import sys; sys.path.insert(0, '.')
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
from submissions.our_submission.sa_placer import SAPlacer
import time

bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
placer = SAPlacer(time_limit=60.0)   # short run for quick check
placement = placer.place(bench)

valid, violations = validate_placement(placement, bench)
costs = compute_proxy_cost(placement, bench, plc)
print(f"Valid: {valid}  |  Overlaps: {costs['overlap_count']}  |  Proxy: {costs['proxy_cost']:.4f}")
EOF
```

### Compare against competition baselines across benchmarks

```bash
# Single benchmark
python submissions/our_submission/compare.py submissions/our_submission/sa_placer.py -b ibm01

# Subset (fast iteration)
python submissions/our_submission/compare.py submissions/our_submission/sa_placer.py -b ibm01 ibm03 ibm09

# All 17 IBM benchmarks (takes ~15 hours with full time limit)
python submissions/our_submission/compare.py submissions/our_submission/sa_placer.py --all
```

Output columns: `proxy cost | overlaps | wirelength | density | congestion` vs SA and RePlAce baselines.

### Official evaluator

```bash
uv run evaluate submissions/our_submission/sa_placer.py -b ibm01
uv run evaluate submissions/our_submission/sa_placer.py --all
uv run evaluate submissions/our_submission/sa_placer.py --all --vis   # saves visualisations
```

---

## Key Parameters (sa_placer.py)

| Parameter | Default | Effect |
|---|---|---|
| `time_limit` | `3300.0` s | Wall-clock budget per benchmark (rule: 3600 s max) |
| `cooling_alpha` | `0.001` | T drops to 0.1% of T₀ by end of run |
| `swap_prob` | `0.40` | 40% of moves are pairwise swaps |
| `max_reheats` | `5` | Max reheat events per benchmark |
| `perturb_frac_init` | `0.30` | Initial perturbation radius (30% of canvas) |
| `perturb_frac_final` | `0.005` | Final perturbation radius (0.5% of canvas) |

For quick local testing, pass `SAPlacer(time_limit=60.0)` to get a result in ~1 minute.

---

## Competition Context

See [CHALLENGE.md](CHALLENGE.md) for the full competition rules, prize structure, evaluation details, and current leaderboard.

**Proxy cost formula** (lower is better):
```
Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```

Published baselines:
- SA baseline: avg `2.1251` across 17 benchmarks
- RePlAce baseline: avg `1.4578` (the target to beat)
- Top leaderboard (DREAMPlace variants): ~`1.40`

**Merch:** Every valid submission (zero overlaps, within runtime) gets HRT swag.

---

## Team

| Name | Role |
|---|---|
| Krishna | Lead — algorithm, infrastructure |
| *(add teammates)* | |
| *(add teammates)* | |
| *(add teammates)* | |
| *(add teammates)* | |

---

## Submitting

One submission per team via the [Google Form](https://forms.gle/YDRtYV5Vq68SZgKW9) — deadline May 21, 2026.  
Private repos must be shared with judges for evaluation.
