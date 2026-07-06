"""Quick 60s sanity check for karthik_qsa placer on ibm01.

Soft-macro opt disabled at short budgets — at 60s it gets called once
and eats the entire remaining budget on Python force-directed solving.
"""
import sys; sys.path.insert(0, '.')
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
# from submissions.our_submission.karthik_qsa import QSAPlacer, QSAConfig
from submissions.our_submission.qsa_tweaked import QSAPlacer, QSAConfig
import time

bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
bench._plc = plc

t0 = time.time()
placer = QSAPlacer(QSAConfig(
    time_limit=60.0,
    verbose=True,
    use_soft_macro_opt=False,   # critical: too slow for short budgets
    use_sb_escape=False,
))
placement = placer.place(bench, plc=plc)
print(f"\nRuntime: {time.time()-t0:.1f}s")

valid, violations = validate_placement(placement, bench)
costs = compute_proxy_cost(placement, bench, plc)
print(f"Valid: {valid} | Overlaps: {costs['overlap_count']} | Proxy: {costs['proxy_cost']:.4f}")
if not valid:
    print(f"Violations: {violations}")
print(f"Your SA-v1 on ibm01 (full 55min): 1.2967")
print(f"SA baseline (published): 1.3166")
print(f"RePlAce baseline (published): 0.9976")