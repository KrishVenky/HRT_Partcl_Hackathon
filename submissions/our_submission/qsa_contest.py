"""
QSA contest wrapper.

This keeps the proven `qsa_tweaked.py` defaults intact and only fixes one
integration gap: the official evaluator calls `place(benchmark)` without
passing the `PlacementCost` object (`plc`). The base QSA implementation can
use `plc` for full proxy-cost tracking and soft-macro re-optimisation, so we
recover it here from the benchmark name when needed.

Usage:
    uv run evaluate submissions/our_submission/qsa_contest.py -b ibm01
    uv run evaluate submissions/our_submission/qsa_contest.py --all
"""

import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark
from submissions.our_submission.qsa_tweaked import (
    QSAConfig,
    QSAPlacer as BaseQSAPlacer,
)


def _load_plc_for_benchmark(name: str) -> Optional[PlacementCost]:
    """Best-effort plc loader for evaluator runs that only pass Benchmark."""
    ibm_root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm_root.exists():
        _, plc = load_benchmark_from_dir(str(ibm_root))
        return plc

    ng45 = {
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "mempool_tile": "mempool_tile",
        "nvdla": "nvdla",
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "mempool_tile_ng45": "mempool_tile",
        "nvdla_ng45": "nvdla",
    }
    design = ng45.get(name)
    if design is None:
        return None

    ng45_root = (
        Path("external/MacroPlacement/Flows/NanGate45")
        / design
        / "netlist"
        / "output_CT_Grouping"
    )
    netlist = ng45_root / "netlist.pb.txt"
    plc_file = ng45_root / "initial.plc"
    if netlist.exists() and plc_file.exists():
        _, plc = load_benchmark(str(netlist), str(plc_file), name=design)
        return plc

    return None


class QSAPlacer(BaseQSAPlacer):
    """
    Contest-ready QSA placer.

    Same defaults and tuning as `qsa_tweaked.QSAPlacer`; only auto-recovers
    `plc` so official evaluator runs can use the full objective path.

    Also adds a conservative runtime guard around soft-macro optimisation:
    if there is clearly not enough time left to profit from a softcell solve,
    we skip that call and keep the remaining budget for SA moves.
    """

    def __init__(self, config: Optional[QSAConfig] = None, **overrides):
        super().__init__(config=config, **overrides)
        self._run_started_at: Optional[float] = None
        self._softcell_runtime_estimate: Optional[float] = None

    def place(self, benchmark: Benchmark, plc=None):
        self._run_started_at = time.time()
        self._softcell_runtime_estimate = None
        if plc is None:
            plc = getattr(benchmark, "_plc", None)
        if plc is None:
            plc = _load_plc_for_benchmark(benchmark.name)
            if plc is not None:
                benchmark._plc = plc
        return super().place(benchmark, plc=plc)

    def _optimize_softcells(self, plc, benchmark, placement):
        if self._run_started_at is not None:
            elapsed = time.time() - self._run_started_at
            remaining = self.cfg.time_limit - elapsed
            estimate = self._softcell_runtime_estimate
            if estimate is None:
                estimate = 90.0
            guard = max(60.0, estimate * 1.25)
            if remaining <= guard:
                if self.cfg.verbose:
                    print(
                        f"  [{benchmark.name}] skip softcell "
                        f"(remaining={remaining:.0f}s guard={guard:.0f}s)"
                    )
                return

        t0 = time.time()
        super()._optimize_softcells(plc, benchmark, placement)
        runtime = time.time() - t0
        if runtime > 0.0:
            if self._softcell_runtime_estimate is None:
                self._softcell_runtime_estimate = runtime
            else:
                self._softcell_runtime_estimate = (
                    0.5 * self._softcell_runtime_estimate + 0.5 * runtime
                )


__all__ = ["QSAConfig", "QSAPlacer"]
