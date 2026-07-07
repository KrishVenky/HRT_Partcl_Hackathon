# QSA - Macro Placer · Team AxeCap

**Partcl × HRT Macro Placement Challenge (2026)**

A **Quantum-Annealing-inspired Simulated Annealing (QSA)** placer that positions
chip macros on a floorplan to minimise the competition's proxy cost:

```
Proxy Cost = 1.0 × Wirelength  +  0.5 × Density  +  0.5 × Congestion      (lower is better)
```

<p align="center">
  <img src="assets/qsa_ibm01.gif" alt="QSA optimising the ibm01 macro layout" width="560"><br>
  <em>QSA settling 246 macros on ibm01 - blue = movable, orange = just moved, darker = overlap.</em>
</p>

---

## Highlights

- **Zero overlaps on all 17 IBM benchmarks** - every placement is legal, no float-precision violations.
- **~38× faster HPWL** via a fully vectorised, batched net tensor - far more SA moves per second.
- **Optimises the real objective** - SA minimises the exact `WL + 0.5·D + 0.5·C` proxy cost, not just wirelength.
- **Single self-contained file** - [`qsa_contest.py`](submissions/our_submission/qsa_contest.py) runs under the official evaluator with no extra submission dependencies.

---

## The submission

```
submissions/our_submission/qsa_contest.py
```

Run it with the official evaluator:

```bash
uv run evaluate submissions/our_submission/qsa_contest.py -b ibm01   # single benchmark
uv run evaluate submissions/our_submission/qsa_contest.py --all      # all 17 IBM benchmarks
```

---

## How QSA works

The placer starts from a good layout and refines it with simulated annealing on the
true proxy cost. The pieces that matter:

1. **Smart initialisation.** Starts from the benchmark's hand-crafted `macro_positions`
   layout, legalised with a tetris-style spiral search. Falls back to shelf-packing only
   if legalisation fails - so SA begins from a far better point than a greedy start.

2. **Vectorised HPWL.** A padded net tensor lets half-perimeter wirelength be computed
   fully batched (~38× faster than a Python net-loop), which buys many more moves inside
   the time budget.

3. **Proxy-cost objective.** SA directly minimises `WL + 0.5·Density + 0.5·Congestion`
   - the exact metric the evaluator scores - instead of a wirelength-only proxy.

4. **Move mix.** 40% pairwise swaps + 60% random perturbations, with
   connectivity-weighted selection (macros on more nets move more often).

5. **Classical reheating.** Up to 4 reheats on a stall: restore the best-known placement
   and raise temperature to escape local optima.

6. **Time-based density resync.** The full proxy cost is recomputed every *N* wall-clock
   seconds (calibrated to 3× the measured evaluation latency), so a slow benchmark like
   ibm17 (~166 s/call) never stalls the run.

7. **`plc` auto-loading.** If the evaluator doesn't pass a `PlacementCost` object, the
   placer loads it from disk automatically - full proxy-cost tracking in every run mode.

---

## Results

Full 55-minute runs, all 17 IBM benchmarks - **17/17 valid, zero overlaps.**

| | Score |
|---|---|
| Our average proxy | **1.7446** |
| SA baseline | 2.1251 |
| RePlAce baseline | 1.4578 |
| vs SA baseline | **+17.9% better** |
| vs RePlAce baseline | -19.7% |
| Valid placements (0 overlaps) | **17 / 17** |

<details>
<summary>Per-benchmark breakdown</summary>

| Benchmark | Proxy | WL | Density | Congestion | vs SA | Overlaps |
|-----------|-------|------|---------|------------|-------|----------|
| ibm01 | 1.2967 | 0.065 | 1.054 | 1.409 | +1.5%  | 0 |
| ibm02 | 1.8409 | 0.078 | 1.060 | 2.466 | +3.5%  | 0 |
| ibm03 | 1.7131 | 0.090 | 1.028 | 2.218 | +1.6%  | 0 |
| ibm04 | 1.6252 | 0.073 | 1.077 | 2.027 | -8.1%  | 0 |
| ibm06 | 1.9216 | 0.064 | 1.021 | 2.694 | +23.3% | 0 |
| ibm07 | 1.7698 | 0.067 | 1.127 | 2.278 | +12.5% | 0 |
| ibm08 | 2.1493 | 0.080 | 1.159 | 2.980 | -11.7% | 0 |
| ibm09 | 1.4169 | 0.059 | 1.125 | 1.590 | -2.1%  | 0 |
| ibm10 | 1.8698 | 0.076 | 1.013 | 2.574 | +11.4% | 0 |
| ibm11 | 1.4808 | 0.058 | 1.111 | 1.734 | +13.5% | 0 |
| ibm12 | 2.1812 | 0.077 | 1.038 | 3.170 | +22.8% | 0 |
| ibm13 | 1.6712 | 0.055 | 1.123 | 2.110 | +12.7% | 0 |
| ibm14 | 1.7050 | 0.051 | 1.110 | 2.198 | +25.1% | 0 |
| ibm15 | 1.6761 | 0.058 | 1.032 | 2.205 | +27.1% | 0 |
| ibm16 | 1.7028 | 0.049 | 1.016 | 2.292 | +23.8% | 0 |
| ibm17 | 1.8081 | 0.053 | 1.025 | 2.485 | +50.8% | 0 |
| ibm18 | 1.8296 | 0.053 | 1.107 | 2.447 | +34.1% | 0 |
| **AVG** | **1.7446** | | | | **+17.9%** | **0** |

Full log and methodology: [`submissions/our_submission/results.md`](submissions/our_submission/results.md).

</details>

---

## Repository layout

```
submissions/our_submission/
    qsa_contest.py     ← the submission (self-contained)
    parallel_eval.py   ← run all 17 benchmarks in parallel
    quick_eval.py      ← quick iteration at a reduced time budget
    make_gif.py        ← render the optimisation GIF (see below)
    results.md         ← full evaluation results

macro_place/           ← evaluation framework (from the public challenge repo)
assets/qsa_ibm01.gif   ← optimisation animation
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/KrishVenky/HRT_Partcl_Hackathon.git
cd HRT_Partcl_Hackathon
uv sync
```

> To run the official evaluator end-to-end you also need the TILOS MacroPlacement
> evaluator and the IBM ICCAD04 benchmarks from the public Partcl × HRT challenge
> framework, placed under `external/MacroPlacement/` and `benchmarks/`.

---

## Running

**Single benchmark (official evaluator):**
```bash
uv run evaluate submissions/our_submission/qsa_contest.py -b ibm01
```

**All 17 benchmarks in parallel:**
```bash
uv run python submissions/our_submission/parallel_eval.py --workers 4
```

**Quick sanity check (60 s per benchmark):**
```bash
uv run python submissions/our_submission/parallel_eval.py \
    --placer submissions/our_submission/qsa_contest.py \
    --workers 2 --time 60 -b ibm01 ibm10 ibm18
```

**Render the optimisation GIF:**
```bash
uv run python submissions/our_submission/make_gif.py --benchmark ibm01 --frames 120 --fps 15
# → assets/qsa_ibm01.gif
```

---

## Team

**Team AxeCap** - Krishna Venkatesh, Karthikeya Machiraju, Krishna Sujith, Adithya Shetty.

Built for the Partcl × HRT Macro Placement Challenge. Thanks to William Salcedo and the
Partcl team for a genuinely rigorous competition, and to the HRT team for the swag.

## License

Apache License 2.0 - see [LICENSE.md](LICENSE.md).
