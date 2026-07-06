"""
Quick evaluation harness for fast iteration during development.

Run the placer at reduced time budget across a chosen subset of
benchmarks and print a comparison table vs. SA / RePlAce baselines.

This is NOT the submission scoring path — the evaluator harness
`uv run evaluate <placer>.py --all` is. This script is purely for
you to iterate on code changes without waiting 15+ minutes each
time to see if you broke something.

Usage (from repo root):
    # 60s per benchmark, 3 representative benchmarks
    uv run python submissions/our_submission/quick_eval.py

    # Override: 30s each, all benchmarks
    uv run python submissions/our_submission/quick_eval.py --time 30 --all

    # Run one specific benchmark with visualisation
    uv run python submissions/our_submission/quick_eval.py --benchmark ibm10 --vis

    # Ablate features
    uv run python submissions/our_submission/quick_eval.py --no-qubo
    uv run python submissions/our_submission/quick_eval.py --no-softcell

Benchmarks are picked to span the hard/easy spectrum:
    ibm01  — smallest (246 macros)       — fast sanity check
    ibm10  — medium   (387 macros)       — representative
    ibm18  — largest  (537 macros)       — worst-case stress
"""

import argparse
import sys
import time
from pathlib import Path

# Make the submission folder importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

# from hybrid_placer import HybridConfig, HybridPlacer
from qsa_tweaked import QSAConfig as HybridConfig, QSAPlacer as HybridPlacer

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement


# SA / RePlAce reference baselines from the README leaderboard.
# These are the numbers your placer has to beat to rank well.
BASELINES = {
    "ibm01": (1.3166, 0.9976),
    "ibm02": (1.9072, 1.8370),
    "ibm03": (1.7401, 1.3222),
    "ibm04": (1.5037, 1.3024),
    "ibm06": (2.5057, 1.6187),
    "ibm07": (2.0229, 1.4633),
    "ibm08": (1.9239, 1.4285),
    "ibm09": (1.3875, 1.1194),
    "ibm10": (2.1108, 1.5009),
    "ibm11": (1.7111, 1.1774),
    "ibm12": (2.8261, 1.7261),
    "ibm13": (1.9141, 1.3355),
    "ibm14": (2.2750, 1.5436),
    "ibm15": (2.3000, 1.5159),
    "ibm16": (2.2337, 1.4780),
    "ibm17": (3.6726, 1.6446),
    "ibm18": (2.7755, 1.7722),
}

# Representative subset for quick iteration
QUICK_SUBSET = ["ibm01", "ibm10", "ibm18"]
ALL_BENCHMARKS = list(BASELINES.keys())

# Relative to repo root (where `uv run` is invoked)
BENCHMARK_DIR = "external/MacroPlacement/Testcases/ICCAD04"


def run_benchmark(name: str, time_limit: float, cfg_overrides: dict,
                  visualize: bool = False) -> dict:
    """Run the hybrid placer on one benchmark; return a result dict."""
    print(f"\n=== {name} (budget {time_limit:.0f}s) ===")
    t0 = time.time()
    benchmark, plc = load_benchmark_from_dir(f"{BENCHMARK_DIR}/{name}")
    # Stash plc on the benchmark so the placer can access it for
    # soft-macro optimisation. Matches the convention the loader
    # presumably uses internally.
    benchmark._plc = plc

    cfg = HybridConfig(time_limit=time_limit, verbose=True, **cfg_overrides)
    placer = HybridPlacer(cfg)
    placement = placer.place(benchmark)

    # Validate
    is_valid, violations = validate_placement(placement, benchmark)
    if not is_valid:
        print(f"  ⚠  VALIDATION FAILED: {violations}")

    # Score
    costs = compute_proxy_cost(placement, benchmark, plc)
    sa_ref, replace_ref = BASELINES[name]

    proxy = costs["proxy_cost"]
    overlaps = costs["overlap_count"]

    vs_sa = (sa_ref - proxy) / sa_ref * 100.0
    vs_re = (replace_ref - proxy) / replace_ref * 100.0

    print(
        f"  result: proxy={proxy:.4f}  WL={costs['wirelength_cost']:.4f}  "
        f"D={costs['density_cost']:.4f}  C={costs['congestion_cost']:.4f}  "
        f"overlaps={overlaps}  ({time.time() - t0:.1f}s)"
    )
    print(
        f"          vs SA baseline {sa_ref:.4f}: "
        f"{'+' if vs_sa >= 0 else ''}{vs_sa:.1f}%  |  "
        f"vs RePlAce {replace_ref:.4f}: "
        f"{'+' if vs_re >= 0 else ''}{vs_re:.1f}%"
    )

    if visualize:
        out_path = f"/tmp/placement_{name}.png"
        visualize_placement(placement, benchmark, save_path=out_path)
        print(f"  visualised → {out_path}")

    return {
        "name": name,
        "proxy": proxy,
        "wl": costs["wirelength_cost"],
        "density": costs["density_cost"],
        "congestion": costs["congestion_cost"],
        "overlaps": overlaps,
        "vs_sa_pct": vs_sa,
        "vs_re_pct": vs_re,
        "valid": is_valid,
        "time_s": time.time() - t0,
    }


def print_summary(results: list) -> None:
    """Print a compact comparison table across all runs."""
    if not results:
        return
    print("\n" + "=" * 78)
    print(f"{'benchmark':<10} {'proxy':>8} {'SA':>8} {'RePlAce':>8} "
          f"{'vs SA%':>8} {'vs RE%':>8} {'ovlp':>5} {'time':>7}")
    print("-" * 78)
    tot_proxy = tot_sa = tot_re = 0.0
    for r in results:
        sa_ref, re_ref = BASELINES[r["name"]]
        tot_proxy += r["proxy"]
        tot_sa += sa_ref
        tot_re += re_ref
        print(
            f"{r['name']:<10} {r['proxy']:>8.4f} {sa_ref:>8.4f} {re_ref:>8.4f} "
            f"{r['vs_sa_pct']:>7.1f}% {r['vs_re_pct']:>7.1f}% "
            f"{r['overlaps']:>5d} {r['time_s']:>6.1f}s"
        )
    print("-" * 78)
    n = len(results)
    print(
        f"{'AVG':<10} {tot_proxy/n:>8.4f} {tot_sa/n:>8.4f} {tot_re/n:>8.4f} "
        f"{(tot_sa - tot_proxy) / tot_sa * 100:>7.1f}% "
        f"{(tot_re - tot_proxy) / tot_re * 100:>7.1f}%"
    )
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Quick hybrid-placer eval")
    parser.add_argument("--time", type=float, default=60.0,
                        help="seconds per benchmark (default 60)")
    parser.add_argument("--all", action="store_true",
                        help="run all 17 IBM benchmarks")
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help="run a specific benchmark (ibm01, ibm02, ...)")
    parser.add_argument("--vis", action="store_true",
                        help="save placement visualisation to /tmp/")
    parser.add_argument("--no-qubo", action="store_true",
                        help="disable QUBO/SB escape (pure SA)")
    parser.add_argument("--no-softcell", action="store_true",
                        help="disable soft-macro reoptimisation")
    parser.add_argument("--no-density", action="store_true",
                        help="disable density in SA cost")
    parser.add_argument("--force-softcell", action="store_true",
                        help="force soft-macro opt even in dev mode "
                             "(usually takes MANY minutes per call)")
    args = parser.parse_args()

    if args.benchmark:
        names = [args.benchmark]
    elif args.all:
        names = ALL_BENCHMARKS
    else:
        names = QUICK_SUBSET

    overrides = {}
    if args.no_qubo:
        overrides["use_qubo_escape"] = False
    if args.no_softcell:
        overrides["use_soft_macro_opt"] = False
    if args.no_density:
        overrides["use_density_in_sa"] = False

    # Warning for the thing that bit us on the first dev run:
    # soft-cell optimisation takes 5-15 minutes per call on CPU.
    if args.time < 300 and not args.no_softcell and not args.force_softcell:
        print(
            "  [quick_eval] NOTE: soft-cell opt will be auto-disabled "
            "because --time < 300s. Use --force-softcell to override "
            "(not recommended unless you have hours to wait)."
        )

    print(f"Running on {len(names)} benchmark(s): {', '.join(names)}")
    if overrides:
        print(f"Config overrides: {overrides}")
    else:
        print("Config: default (all features on)")

    results = []
    for name in names:
        try:
            r = run_benchmark(name, args.time, overrides, visualize=args.vis)
            results.append(r)
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print_summary(results)


if __name__ == "__main__":
    main()