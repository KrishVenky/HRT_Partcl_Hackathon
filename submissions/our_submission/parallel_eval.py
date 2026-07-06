"""
Parallel evaluation harness.

Runs a placer across all 17 IBM benchmarks in parallel, using multiple
processes. Each process evaluates one benchmark at a time. Results are
collected and printed as a summary table when all finish.

Usage (from repo root):
    # Use 2 parallel workers (for 10-core laptop)
    uv run python submissions/our_submission/parallel_eval.py --workers 2

    # Use 4 workers (for 16-core workstation)
    uv run python submissions/our_submission/parallel_eval.py --workers 4

    # Run a subset
    uv run python submissions/our_submission/parallel_eval.py --workers 2 -b ibm01 ibm10 ibm18

    # Custom time limit (seconds per benchmark)
    uv run python submissions/our_submission/parallel_eval.py --workers 2 --time 300
"""

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
from pathlib import Path

# Make repo root importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

SA_BASELINES = {
    "ibm01":1.3166,"ibm02":1.9072,"ibm03":1.7401,"ibm04":1.5037,
    "ibm06":2.5057,"ibm07":2.0229,"ibm08":1.9239,"ibm09":1.3875,
    "ibm10":2.1108,"ibm11":1.7111,"ibm12":2.8261,"ibm13":1.9141,
    "ibm14":2.2750,"ibm15":2.3000,"ibm16":2.2337,"ibm17":3.6726,"ibm18":2.7755,
}
REPLACE_BASELINES = {
    "ibm01":0.9976,"ibm02":1.8370,"ibm03":1.3222,"ibm04":1.3024,
    "ibm06":1.6187,"ibm07":1.4633,"ibm08":1.4285,"ibm09":1.1194,
    "ibm10":1.5009,"ibm11":1.1774,"ibm12":1.7261,"ibm13":1.3355,
    "ibm14":1.5436,"ibm15":1.5159,"ibm16":1.4780,"ibm17":1.6446,"ibm18":1.7722,
}

BENCHMARK_DIR = "external/MacroPlacement/Testcases/ICCAD04"


def _load_placer(placer_path: str, time_limit: float):
    """Load placer from .py file and override time_limit if possible."""
    path = Path(placer_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            # Try to instantiate with time_limit override
            try:
                cfg_class = None
                for name, obj in vars(mod).items():
                    if isinstance(obj, type) and "Config" in name:
                        cfg_class = obj
                        break
                if cfg_class:
                    cfg = cfg_class(time_limit=time_limit, use_soft_macro_opt=False)
                    return attr(cfg)
            except Exception:
                pass
            return attr()

    raise RuntimeError(f"No placer class found in {path}")


def _eval_one(args):
    """Worker function — runs one benchmark, returns result dict."""
    placer_path, bench_name, time_limit = args

    # Re-add repo root to path (needed in subprocess)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    try:
        placer = _load_placer(placer_path, time_limit)
        benchmark, plc = load_benchmark_from_dir(f"{BENCHMARK_DIR}/{bench_name}")
        benchmark._plc = plc

        t0 = time.time()
        placement = placer.place(benchmark, plc=plc) if _accepts_plc(placer) else placer.place(benchmark)
        runtime = time.time() - t0

        is_valid, _ = validate_placement(placement, benchmark)
        costs = compute_proxy_cost(placement, benchmark, plc)

        proxy = costs["proxy_cost"]
        overlaps = costs["overlap_count"]
        sa_ref = SA_BASELINES.get(bench_name, 0)
        re_ref = REPLACE_BASELINES.get(bench_name, 0)

        print(
            f"  [DONE] {bench_name:<6} proxy={proxy:.4f}  "
            f"vs SA {'+' if proxy < sa_ref else ''}{(sa_ref-proxy)/sa_ref*100:.1f}%  "
            f"vs RePlAce {'+' if proxy < re_ref else ''}{(re_ref-proxy)/re_ref*100:.1f}%  "
            f"overlaps={overlaps}  valid={is_valid}  [{runtime:.0f}s]",
            flush=True
        )

        return {
            "name": bench_name, "proxy": proxy,
            "wl": costs["wirelength_cost"],
            "density": costs["density_cost"],
            "congestion": costs["congestion_cost"],
            "overlaps": overlaps, "valid": is_valid, "runtime": runtime,
        }

    except Exception as e:
        import traceback
        print(f"  [ERROR] {bench_name}: {e}", flush=True)
        traceback.print_exc()
        return {"name": bench_name, "proxy": float("inf"), "overlaps": -1,
                "valid": False, "runtime": 0, "error": str(e)}


def _accepts_plc(placer):
    """Check if placer.place() accepts a plc keyword argument."""
    import inspect
    try:
        sig = inspect.signature(placer.place)
        return "plc" in sig.parameters
    except Exception:
        return False


def _print_summary(results):
    sa_vals = [SA_BASELINES[r["name"]] for r in results if r["name"] in SA_BASELINES]
    re_vals = [REPLACE_BASELINES[r["name"]] for r in results if r["name"] in REPLACE_BASELINES]
    proxy_vals = [r["proxy"] for r in results if r["proxy"] != float("inf")]

    print("\n" + "=" * 80)
    print(f"{'benchmark':<10} {'proxy':>8} {'SA':>8} {'RePlAce':>8} "
          f"{'vs SA%':>8} {'vs RE%':>8} {'ovlp':>5} {'valid':>6} {'time':>7}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["name"]):
        sa = SA_BASELINES.get(r["name"], 0)
        re = REPLACE_BASELINES.get(r["name"], 0)
        vs_sa = (sa - r["proxy"]) / sa * 100 if sa else 0
        vs_re = (re - r["proxy"]) / re * 100 if re else 0
        print(
            f"{r['name']:<10} {r['proxy']:>8.4f} {sa:>8.4f} {re:>8.4f} "
            f"{vs_sa:>7.1f}% {vs_re:>7.1f}% "
            f"{r['overlaps']:>5} {'Y' if r['valid'] else 'N':>6} {r['runtime']:>6.0f}s"
        )

    print("-" * 80)
    if proxy_vals:
        avg_proxy = sum(proxy_vals) / len(proxy_vals)
        avg_sa = sum(sa_vals) / len(sa_vals) if sa_vals else 0
        avg_re = sum(re_vals) / len(re_vals) if re_vals else 0
        print(
            f"{'AVG':<10} {avg_proxy:>8.4f} {avg_sa:>8.4f} {avg_re:>8.4f} "
            f"{(avg_sa-avg_proxy)/avg_sa*100:>7.1f}% "
            f"{(avg_re-avg_proxy)/avg_re*100:>7.1f}%"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Parallel benchmark evaluator")
    parser.add_argument("--placer", default="submissions/our_submission/qsa_tweaked.py",
                        help="Path to placer .py file")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Number of parallel workers (default: 2)")
    parser.add_argument("--benchmarks", "-b", nargs="+", default=None,
                        help="Specific benchmarks to run (default: all 17)")
    parser.add_argument("--time", "-t", type=float, default=3300.0,
                        help="Time limit per benchmark in seconds (default: 3300 = 55min)")
    args = parser.parse_args()

    benchmarks = args.benchmarks or IBM_BENCHMARKS
    placer_path = str(Path(args.placer).resolve())

    print(f"Placer:     {args.placer}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Workers:    {args.workers}")
    print(f"Time/bench: {args.time:.0f}s ({args.time/60:.1f} min)")
    print(f"Est. total: ~{args.time * len(benchmarks) / args.workers / 3600:.1f} hours")
    print()

    work = [(placer_path, b, args.time) for b in benchmarks]

    t_start = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_eval_one, work)

    print(f"\nAll done in {(time.time()-t_start)/3600:.2f} hours")
    _print_summary(results)


if __name__ == "__main__":
    mp.freeze_support()  # needed on Windows
    main()