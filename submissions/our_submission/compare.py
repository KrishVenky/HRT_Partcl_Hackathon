"""
Placer comparison harness.

Runs any number of placer .py files across the same benchmarks and prints a
side-by-side table so you can objectively compare algorithms before you commit
to a submission.

Usage
-----
# Compare two placers on all 17 IBM benchmarks:
    python submissions/our_submission/compare.py \\
        submissions/examples/greedy_row_placer.py \\
        submissions/our_submission/sa_placer.py

# Restrict to a subset of benchmarks (faster for iteration):
    python submissions/our_submission/compare.py \\
        submissions/our_submission/sa_placer.py \\
        -b ibm01 ibm03 ibm09

# Use all IBM benchmarks (default):
    python submissions/our_submission/compare.py \\
        submissions/our_submission/sa_placer.py \\
        --all

# Include NG45 designs too:
    python submissions/our_submission/compare.py \\
        submissions/our_submission/sa_placer.py \\
        --ng45

Output
------
One row per (benchmark, placer) with proxy cost, component breakdown, and
overlap count.  A summary table at the end ranks placers by average proxy
cost across the selected benchmarks.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── make sure macro_place is importable when run directly ───────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

# ── benchmark lists (matches the official evaluate harness) ─────────────────

IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

NG45_BENCHMARKS = {
    "ariane133":   "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136":   "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile":"external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla":       "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}

# Published baselines for reference
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

TESTCASE_ROOT = Path("external/MacroPlacement/Testcases/ICCAD04")


# ── placer loader (identical logic to the official evaluate harness) ─────────

def _load_placer(path: Path):
    """
    Import a placer .py file and instantiate its placer class.

    The convention (from the official harness):
      - The first class defined in the file that has a `place` method is used.
      - It is instantiated with no arguments.
    """
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()

    raise RuntimeError(
        f"No placer class found in {path}.\n"
        "Expected a class with a  place(self, benchmark) -> Tensor  method."
    )


# ── single benchmark evaluation ──────────────────────────────────────────────

def _run_one(placer, name: str, ng45_dir: Optional[str] = None) -> dict:
    """
    Run one placer on one benchmark.  Returns a result dict.
    """
    if ng45_dir:
        netlist = f"{ng45_dir}/netlist.pb.txt"
        plc_file = f"{ng45_dir}/initial.plc"
        benchmark, plc = load_benchmark(netlist, plc_file)
    else:
        benchmark, plc = load_benchmark_from_dir(str(TESTCASE_ROOT / name))

    t0 = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - t0

    is_valid, _ = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)

    return {
        "name":       name,
        "proxy":      costs["proxy_cost"],
        "wl":         costs["wirelength_cost"],
        "density":    costs["density_cost"],
        "congestion": costs["congestion_cost"],
        "overlaps":   costs["overlap_count"],
        "runtime":    runtime,
        "valid":      is_valid,
    }


# ── pretty-printing ──────────────────────────────────────────────────────────

def _col(s, w, align="right"):
    """Right- or left-align a string in a fixed-width column."""
    return s.rjust(w) if align == "right" else s.ljust(w)


def _print_per_benchmark(bench_name: str, results_by_placer: dict, placer_names: list):
    """
    Print one benchmark row showing each placer's proxy cost and overlaps.
    Also shows SA and RePlAce baselines if available.
    """
    sa  = SA_BASELINES.get(bench_name)
    rep = REPLACE_BASELINES.get(bench_name)

    parts = [_col(bench_name, 13)]
    for pname in placer_names:
        r = results_by_placer.get(pname, {}).get(bench_name)
        if r is None:
            parts.append(_col("—", 10))
            parts.append(_col("—", 5))
        else:
            cost_str = f"{r['proxy']:.4f}" + ("*" if r["overlaps"] > 0 else " ")
            overlap_str = str(r["overlaps"])
            parts.append(_col(cost_str, 10))
            parts.append(_col(overlap_str, 5))

    if sa:
        parts.append(_col(f"{sa:.4f}", 10))
    if rep:
        parts.append(_col(f"{rep:.4f}", 10))

    print("  ".join(parts))


def _print_summary(
    benchmarks: list,
    results_by_placer: dict,
    placer_names: list,
    placer_runtimes: dict,
):
    """
    Print a ranked summary table and per-benchmark breakdown.
    """
    # ── column headers ───────────────────────────────────────────────────────
    has_sa  = any(b in SA_BASELINES  for b in benchmarks)
    has_rep = any(b in REPLACE_BASELINES for b in benchmarks)

    name_width = 13
    col_w = 10  # proxy cost column
    flag_w = 5  # overlaps column

    header = _col("Benchmark", name_width)
    for pname in placer_names:
        short = pname[:8]  # truncate long names
        header += "  " + _col(short, col_w) + "  " + _col("OL", flag_w)
    if has_sa:
        header += "  " + _col("SA", col_w)
    if has_rep:
        header += "  " + _col("RePlAce", col_w)

    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    # ── per-benchmark rows ───────────────────────────────────────────────────
    for b in benchmarks:
        _print_per_benchmark(b, results_by_placer, placer_names)

    # ── averages ─────────────────────────────────────────────────────────────
    print(sep)
    avg_row = _col("AVG", name_width)
    placer_avgs = {}
    for pname in placer_names:
        costs = [
            results_by_placer[pname][b]["proxy"]
            for b in benchmarks
            if b in results_by_placer.get(pname, {})
            and results_by_placer[pname][b]["overlaps"] == 0
        ]
        total_overlaps = sum(
            results_by_placer[pname].get(b, {}).get("overlaps", 0)
            for b in benchmarks
        )
        if costs:
            avg = sum(costs) / len(costs)
            placer_avgs[pname] = avg
            avg_str = f"{avg:.4f}"
        else:
            avg = float("inf")
            placer_avgs[pname] = avg
            avg_str = "  N/A "
        avg_row += "  " + _col(avg_str, col_w) + "  " + _col(str(total_overlaps), flag_w)
    if has_sa:
        sa_vals = [SA_BASELINES[b] for b in benchmarks if b in SA_BASELINES]
        avg_sa = sum(sa_vals) / len(sa_vals) if sa_vals else 0
        avg_row += "  " + _col(f"{avg_sa:.4f}", col_w)
    if has_rep:
        rep_vals = [REPLACE_BASELINES[b] for b in benchmarks if b in REPLACE_BASELINES]
        avg_rep = sum(rep_vals) / len(rep_vals) if rep_vals else 0
        avg_row += "  " + _col(f"{avg_rep:.4f}", col_w)
    print(avg_row)
    print(sep)

    # ── ranked summary ───────────────────────────────────────────────────────
    print()
    print("Ranked summary (lower proxy = better):")
    print()

    all_entries = sorted(placer_avgs.items(), key=lambda x: x[1])

    # Also include baselines in the ranking
    if has_sa and has_rep:
        baseline_entries = [
            ("RePlAce (baseline)", avg_rep),
            ("SA (baseline)",      avg_sa),
        ]
    else:
        baseline_entries = []

    combined = sorted(all_entries + baseline_entries, key=lambda x: x[1])

    rank = 1
    for name, avg in combined:
        is_baseline = "(baseline)" in name
        marker = "  [baseline]" if is_baseline else ""
        rt_info = ""
        if not is_baseline and name in placer_runtimes:
            total_rt = placer_runtimes[name]
            rt_info = f"  ({total_rt:.1f}s total)"
        print(f"  {'—' if is_baseline else rank:>3}.  {avg:.4f}  {name}{marker}{rt_info}")
        if not is_baseline:
            rank += 1

    print()
    print("* = has overlaps (INVALID — would be disqualified)")
    print()


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="compare",
        description="Compare multiple macro placement algorithms side by side.",
    )
    parser.add_argument(
        "placers",
        nargs="+",
        help="Paths to placer .py files (e.g. submissions/our_submission/sa_placer.py).",
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        default=None,
        metavar="BENCH",
        help="Specific IBM benchmarks to run (e.g. -b ibm01 ibm03 ibm09). "
             "Default: ibm01 only (use --all for all 17).",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all 17 IBM benchmarks.",
    )
    parser.add_argument(
        "--ng45",
        action="store_true",
        help="Also run on NG45 designs (ariane133, ariane136, mempool_tile, nvdla).",
    )
    args = parser.parse_args()

    # ── resolve benchmark list ───────────────────────────────────────────────
    if args.benchmarks:
        benchmarks = args.benchmarks
    elif args.all:
        benchmarks = IBM_BENCHMARKS
    else:
        benchmarks = ["ibm01"]  # quick single-benchmark default

    if args.ng45:
        benchmarks = benchmarks + list(NG45_BENCHMARKS.keys())

    # ── check testcase root exists ───────────────────────────────────────────
    ibm_needed = [b for b in benchmarks if b in IBM_BENCHMARKS]
    if ibm_needed and not TESTCASE_ROOT.exists():
        print(f"Error: {TESTCASE_ROOT} not found.")
        print("Run: git submodule update --init external/MacroPlacement")
        sys.exit(1)

    # ── load all placers ─────────────────────────────────────────────────────
    placers = {}
    placer_names = []
    for p in args.placers:
        path = Path(p)
        print(f"Loading placer: {path.name} ...", end=" ", flush=True)
        try:
            placer = _load_placer(path)
            name = type(placer).__name__
            placers[name] = placer
            placer_names.append(name)
            print(f"OK ({name})")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)

    # ── run all combinations ─────────────────────────────────────────────────
    # results_by_placer[placer_name][bench_name] = result dict
    results_by_placer: dict = {n: {} for n in placer_names}
    placer_runtimes: dict = {n: 0.0 for n in placer_names}

    print()
    print("=" * 70)
    print(f"Running {len(placer_names)} placer(s) × {len(benchmarks)} benchmark(s)")
    print("=" * 70)

    for bench in benchmarks:
        ng45_dir = NG45_BENCHMARKS.get(bench) if bench in NG45_BENCHMARKS else None

        for pname in placer_names:
            print(f"  {pname:30s}  {bench:12s} ...", end=" ", flush=True)

            try:
                result = _run_one(placers[pname], bench, ng45_dir=ng45_dir)
                results_by_placer[pname][bench] = result
                placer_runtimes[pname] += result["runtime"]

                status = "OK" if result["overlaps"] == 0 else f"INVALID ({result['overlaps']} overlaps)"
                print(
                    f"proxy={result['proxy']:.4f}  "
                    f"(wl={result['wl']:.3f} den={result['density']:.3f} "
                    f"cong={result['congestion']:.3f})  "
                    f"{status}  [{result['runtime']:.1f}s]"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    # ── print summary ─────────────────────────────────────────────────────────
    _print_summary(benchmarks, results_by_placer, placer_names, placer_runtimes)


if __name__ == "__main__":
    main()
