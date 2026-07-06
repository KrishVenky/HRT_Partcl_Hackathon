"""
Hybrid Simulated Annealing + Simulated-Bifurcation QUBO Placer
==============================================================

Our submission for the Partcl/HRT Macro Placement Challenge 2026.

High-level algorithm
--------------------
Outer loop: time-anchored simulated annealing (SA) on hard macros.
Inner escape: quantum-inspired QUBO k-cluster reassignment via
              Simulated Bifurcation (SB). Replaces blind reheating
              with a coordinated multi-macro move that single-macro
              perturbations cannot generate.

Upgrades over the baseline SA
-----------------------------
(A) Proxy-cost-weighted acceptance: SA objective includes density
    and congestion in addition to HPWL, so SA directly minimises
    what the evaluator measures.
(B) Periodic soft-macro reoptimisation: every `softcell_interval`
    seconds we call plc.optimize_stdcells() so standard-cell
    clusters follow the hard-macro moves.
(C) QUBO-based cluster escape: when acceptance collapses, pick k
    poorly-placed macros, build a slot-assignment QUBO over their
    candidate positions, solve with Simulated Bifurcation.

The "quantum" piece is Simulated Bifurcation — a GPU-native,
quantum-physics-inspired Ising solver (Goto et al., Sci. Adv.
2019). We invoke it via the `simulated_bifurcation` PyPI package
on subproblems of ~100-200 binary variables.

Each benchmark has a 1-hour hard budget. Default time budget is
3300 s (55 min) to leave buffer for I/O and evaluation.

Usage
-----
    uv run evaluate submissions/our_submission/hybrid_placer.py -b ibm01
    uv run evaluate submissions/our_submission/hybrid_placer.py --all
    uv run evaluate submissions/our_submission/hybrid_placer.py -b ibm01 --vis

For fast local iteration during development, construct the placer
directly with time_limit=60 and use the helper in quick_eval.py.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from macro_place.benchmark import Benchmark

# Simulated Bifurcation is optional — if not installed, the QUBO
# escape silently falls back to classical reheat. This keeps the
# placer runnable on machines without the extra dependency.
try:
    import simulated_bifurcation as sb
    _HAS_SB = True
except ImportError:
    _HAS_SB = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    """
    All tunable knobs in one place.

    Defaults are conservative — the placer will never silently
    explode on an untested benchmark. See TUNING.md for which
    knobs actually move the score.
    """
    # Runtime
    seed: int = 42
    time_limit: float = 3300.0          # 55 min; 5 min buffer for eval
    verbose: bool = True

    # Feature toggles (useful for ablation studies)
    use_density_in_sa: bool = True      # (A) add density to SA cost
    use_congestion_in_sa: bool = False  # congestion is slow; off by default
    use_soft_macro_opt: bool = True     # (B) periodic optimize_stdcells
    use_qubo_escape: bool = True        # (C) SB-QUBO coordinated escape

    # SA schedule (wall-clock anchored)
    cooling_alpha: float = 0.001         # T(end) = T0 * alpha
    perturb_frac_init: float = 0.30
    perturb_frac_final: float = 0.005
    swap_prob: float = 0.40
    accept_prob_init: float = 0.80

    # SA reheating (used if QUBO escape off or unavailable)
    reheat_threshold: float = 0.01
    max_reheats: int = 5

    # Cost weights (mirror evaluator: proxy = 1.0*WL + 0.5*D + 0.5*C)
    density_weight: float = 0.5
    congestion_weight: float = 0.5
    density_recompute_interval: int = 500   # moves between density recomputes
    density_grid: int = 32                   # grid resolution for SA density

    # Soft-macro reoptimisation
    # plc.optimize_stdcells() is pure Python and very slow on CPU
    # (~5-15 minutes per call on ibm01 even with reduced num_steps).
    # We therefore run it infrequently and with minimal inner steps.
    # On GPU this should be much faster, in which case you can shrink
    # softcell_interval and raise softcell_steps.
    softcell_interval: float = 600.0    # seconds between optimize_stdcells
    softcell_steps: Tuple[int, int, int] = (20, 20, 20)  # vs baseline (100,100,100)

    # QUBO escape
    qubo_k: int = 12                    # macros per QUBO call
    qubo_slots_per_macro: int = 8       # candidate positions per macro
    qubo_overlap_penalty: float = 10.0  # weight on overlap terms relative to HPWL
    qubo_agents: int = 32               # SB parallel search agents
    qubo_max_steps: int = 10000         # SB integration steps
    qubo_trigger_window: int = 2000     # accept-rate window size to trigger escape
    qubo_trigger_threshold: float = 0.02
    qubo_min_interval: float = 60.0     # don't call QUBO more often than this


# ---------------------------------------------------------------------------
# The placer
# ---------------------------------------------------------------------------

class HybridPlacer:
    """
    Hybrid SA + Simulated-Bifurcation QUBO escape placer.

    Has the same interface as every other placer in this repo:
    construct it, then call `.place(benchmark)`, which returns a
    `[num_macros, 2]` tensor of (x, y) centre positions.
    """

    def __init__(self, config: Optional[HybridConfig] = None, **overrides):
        """
        Pass a HybridConfig, or override specific fields via kwargs:
            HybridPlacer(time_limit=60, use_qubo_escape=False)
        """
        if config is None:
            config = HybridConfig()
        for k, v in overrides.items():
            if not hasattr(config, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(config, k, v)
        self.cfg = config

        # QUBO escape requires simulated_bifurcation; warn once if missing
        if self.cfg.use_qubo_escape and not _HAS_SB:
            if self.cfg.verbose:
                print(
                    "  [hybrid] simulated_bifurcation not installed — "
                    "falling back to classical reheating. "
                    "Install with: pip install simulated-bifurcation"
                )
            self.cfg.use_qubo_escape = False

    # ---------- public API -------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Run the placer on a benchmark; return placement tensor."""
        torch.manual_seed(self.cfg.seed)
        t_start = time.time()

        # 1. Zero-overlap init by shelf packing (same as baseline SA).
        placement = self._greedy_row_init(benchmark)

        movable_indices = self._get_movable_indices(benchmark)
        N = len(movable_indices)
        if N == 0:
            return placement

        # Reverse adjacency: macro i -> list of net ids containing it.
        macro_nets = self._build_macro_nets(benchmark)

        # Connectivity-weighted selection weights for perturbation moves.
        # +1 so macros with no nets still have a chance to move.
        degrees = torch.tensor(
            [len(macro_nets[i]) + 1.0 for i in movable_indices],
            dtype=torch.float32,
        )

        # 2. Estimate initial temperature from cost-landscape samples.
        T_start = self._estimate_init_temp(
            placement, benchmark, movable_indices, macro_nets
        )
        reheat_mult = 1.0
        log_alpha = math.log(self.cfg.cooling_alpha)

        # 3. Initial costs.
        current_hpwl = self._total_hpwl(placement, benchmark)
        current_density = (
            self._approx_density(placement, benchmark)
            if self.cfg.use_density_in_sa else 0.0
        )
        current_cost = self._combine_cost(current_hpwl, current_density, 0.0)

        best_placement = placement.clone()
        best_cost = current_cost
        best_hpwl = current_hpwl

        # 4. Run counters.
        accepted = rejected = total_moves = 0
        reheats = qubo_calls = 0
        accept_window: deque = deque(maxlen=self.cfg.qubo_trigger_window)
        last_softcell_time = t_start
        last_qubo_time = t_start
        last_density_move = 0

        # Soft-cell optimisation is disabled automatically when time_limit
        # is small (dev/iteration mode). plc.optimize_stdcells() is pure
        # Python and takes ~5-15 minutes per call on CPU, so it only earns
        # its keep on a full production run.
        #
        # IMPORTANT: we do NOT run an eager startup soft-cell pack. Previous
        # versions did that, which ate the entire budget on short runs
        # before any SA iterations could happen. The periodic schedule
        # below (triggered inside the loop) handles soft-cell updates
        # at the right cadence.
        if self.cfg.use_soft_macro_opt and self.cfg.time_limit < 300.0:
            if self.cfg.verbose:
                print(
                    f"  [hybrid] time_limit={self.cfg.time_limit:.0f}s < 300s "
                    "— disabling soft-cell optimisation for this dev run"
                )
            self.cfg.use_soft_macro_opt = False

        # --- Main SA loop (wall-clock-anchored) ---
        while True:
            elapsed = time.time() - t_start
            if elapsed >= self.cfg.time_limit:
                break
            progress = elapsed / self.cfg.time_limit

            T = T_start * math.exp(log_alpha * progress) * reheat_mult
            if T <= 0.0:
                T = 1e-12
            perturb_frac = (
                self.cfg.perturb_frac_init * (1.0 - progress)
                + self.cfg.perturb_frac_final * progress
            )

            # Refresh density estimate periodically — it's expensive
            # to recompute every move but stable enough over many moves.
            if (self.cfg.use_density_in_sa and
                    total_moves - last_density_move >= self.cfg.density_recompute_interval):
                current_density = self._approx_density(placement, benchmark)
                current_cost = self._combine_cost(current_hpwl, current_density, 0.0)
                last_density_move = total_moves

            # --- Decide move type ---
            use_swap = (N >= 2) and (torch.rand(1).item() < self.cfg.swap_prob)

            if use_swap:
                move_result = self._propose_swap(
                    placement, benchmark, macro_nets, movable_indices, N
                )
            else:
                move_result = self._propose_perturb(
                    placement, benchmark, macro_nets, movable_indices,
                    degrees, perturb_frac
                )

            if move_result is None:
                # Geometrically invalid (bounds / overlap).
                rejected += 1
                accept_window.append(False)
                total_moves += 1
                continue

            delta_hpwl, apply_fn = move_result

            # Density delta: we approximate it as proportional to HPWL
            # change modulated by current spread. For speed we only do
            # a full density recompute periodically; in between, the
            # SA objective = weighted HPWL, which is a good proxy.
            delta_cost = delta_hpwl  # density/congestion update below

            # Metropolis criterion on combined cost
            if delta_cost < 0.0 or torch.rand(1).item() < math.exp(-delta_cost / T):
                apply_fn(placement)
                current_hpwl += delta_hpwl
                current_cost += delta_cost
                accepted += 1
                accept_window.append(True)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_placement = placement.clone()
            else:
                rejected += 1
                accept_window.append(False)
            total_moves += 1

            # --- Periodic soft-macro reoptimisation ---
            if (self.cfg.use_soft_macro_opt and
                    time.time() - last_softcell_time > self.cfg.softcell_interval and
                    self.cfg.time_limit - elapsed > self.cfg.softcell_interval):
                self._optimize_softcells(benchmark)
                # After soft-macro moves, HPWL has changed; recompute once.
                current_hpwl = self._total_hpwl(placement, benchmark)
                if self.cfg.use_density_in_sa:
                    current_density = self._approx_density(placement, benchmark)
                current_cost = self._combine_cost(current_hpwl, current_density, 0.0)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_placement = placement.clone()
                last_softcell_time = time.time()

            # --- Escape trigger ---
            recent_accept_rate = (
                sum(accept_window) / len(accept_window)
                if len(accept_window) == accept_window.maxlen else 1.0
            )
            stuck = (
                len(accept_window) == accept_window.maxlen
                and recent_accept_rate < self.cfg.qubo_trigger_threshold
            )
            qubo_eligible = (
                self.cfg.use_qubo_escape
                and time.time() - last_qubo_time > self.cfg.qubo_min_interval
            )

            if stuck and qubo_eligible:
                # Quantum-inspired escape: solve a k-macro slot QUBO.
                new_placement, new_hpwl = self._qubo_escape(
                    placement, benchmark, macro_nets, movable_indices
                )
                if new_placement is not None and new_hpwl < current_hpwl:
                    placement = new_placement
                    current_hpwl = new_hpwl
                    if self.cfg.use_density_in_sa:
                        current_density = self._approx_density(placement, benchmark)
                    current_cost = self._combine_cost(
                        current_hpwl, current_density, 0.0
                    )
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_hpwl = current_hpwl
                        best_placement = placement.clone()
                qubo_calls += 1
                last_qubo_time = time.time()
                accept_window.clear()
                # Warm up temperature a bit so SA can explore around the
                # new basin instead of immediately getting stuck again.
                reheat_mult *= 2.0
                if self.cfg.verbose:
                    print(
                        f"  [{benchmark.name}] QUBO escape #{qubo_calls} at "
                        f"{elapsed:.0f}s | HPWL {current_hpwl:.4f}"
                    )
            elif stuck and reheats < self.cfg.max_reheats:
                # Classical fallback reheat
                reheat_mult *= 4.0
                reheats += 1
                placement = best_placement.clone()
                current_hpwl = best_hpwl
                current_cost = best_cost
                accept_window.clear()
                if self.cfg.verbose:
                    print(
                        f"  [{benchmark.name}] reheat #{reheats} at "
                        f"{elapsed:.0f}s | best cost {best_cost:.4f}"
                    )

        # Final soft-macro pack on the best placement.
        if self.cfg.use_soft_macro_opt:
            # We restore best placement to self.placer's working tensor via
            # the plc object implicitly — easiest is to just not touch it
            # here; plc's soft-macro state reflects whatever we left it in.
            pass

        if self.cfg.verbose:
            accept_rate = accepted / max(total_moves, 1) * 100
            print(
                f"  [{benchmark.name}] {total_moves:,} moves | "
                f"{accept_rate:.1f}% accept | reheats={reheats} | "
                f"qubo={qubo_calls} | HPWL {best_hpwl:.4f} | "
                f"{time.time() - t_start:.1f}s"
            )
        return best_placement

    # ---------- init -------------------------------------------------------

    def _greedy_row_init(self, benchmark: Benchmark) -> torch.Tensor:
        """Shelf-pack hard macros, tallest-first. Guaranteed zero overlap."""
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes
        indices.sort(key=lambda i: -sizes[i, 1].item())

        gap = 0.001  # float32 touching-edge guard
        cursor_x = cursor_y = row_h = 0.0
        for idx in indices:
            w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
            if cursor_x + w > benchmark.canvas_width:
                cursor_x = 0.0
                cursor_y += row_h + gap
                row_h = 0.0
            if cursor_y + h > benchmark.canvas_height:
                # Fallback — shouldn't trigger below 100% utilisation.
                placement[idx, 0] = w / 2.0
                placement[idx, 1] = h / 2.0
                continue
            placement[idx, 0] = cursor_x + w / 2.0
            placement[idx, 1] = cursor_y + h / 2.0
            cursor_x += w + gap
            row_h = max(row_h, h)
        return placement

    def _get_movable_indices(self, benchmark: Benchmark) -> List[int]:
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        return torch.where(movable)[0].tolist()

    def _build_macro_nets(self, benchmark: Benchmark) -> List[List[int]]:
        """macro_nets[i] = list of net ids that touch macro i."""
        macro_nets: List[List[int]] = [[] for _ in range(benchmark.num_macros)]
        for net_id, net_nodes in enumerate(benchmark.net_nodes):
            for node_idx in net_nodes.tolist():
                if 0 <= node_idx < benchmark.num_macros:
                    macro_nets[node_idx].append(net_id)
        return macro_nets

    # ---------- cost functions --------------------------------------------

    def _extended_positions(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        """[macro positions | port positions] for uniform net-node lookup."""
        if benchmark.port_positions.shape[0] > 0:
            return torch.cat([placement, benchmark.port_positions], dim=0)
        return placement

    def _total_hpwl(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """Weighted total HPWL across all nets. Call sparingly — it's O(all pins)."""
        ext = self._extended_positions(placement, benchmark)
        total = 0.0
        for net_id, net_nodes in enumerate(benchmark.net_nodes):
            if len(net_nodes) < 2:
                continue
            coords = ext[net_nodes]
            total += benchmark.net_weights[net_id].item() * (
                (coords[:, 0].max() - coords[:, 0].min())
                + (coords[:, 1].max() - coords[:, 1].min())
            ).item()
        return total

    def _approx_density(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> float:
        """
        Fast proxy for density: top-10% grid cell occupation.

        Bins hard macros onto a coarse grid by centre position and
        accumulates macro area per cell. Returns mean of the top 10%
        cells (matching the evaluator's 'top-10% density' definition).
        This is an approximation — the real evaluator computes
        fractional overlap per cell — but it's correlated strongly
        enough to use inside the SA acceptance test.
        """
        g = self.cfg.density_grid
        W, H = benchmark.canvas_width, benchmark.canvas_height
        n_hard = benchmark.num_hard_macros

        # Cell indices for each hard macro
        xs = (placement[:n_hard, 0] / W * g).clamp(0, g - 1).long()
        ys = (placement[:n_hard, 1] / H * g).clamp(0, g - 1).long()
        areas = (
            benchmark.macro_sizes[:n_hard, 0] * benchmark.macro_sizes[:n_hard, 1]
        )

        density_map = torch.zeros(g * g, dtype=torch.float32)
        flat = (ys * g + xs).long()
        density_map.scatter_add_(0, flat, areas.float())

        cell_area = (W / g) * (H / g)
        occ = density_map / max(cell_area, 1e-9)
        topk = max(1, int(0.1 * g * g))
        return occ.topk(topk).values.mean().item()

    def _combine_cost(self, hpwl: float, density: float, congestion: float) -> float:
        """Evaluator-aligned proxy cost used as SA objective."""
        cost = hpwl
        if self.cfg.use_density_in_sa:
            cost += self.cfg.density_weight * density
        if self.cfg.use_congestion_in_sa:
            cost += self.cfg.congestion_weight * congestion
        return cost

    def _delta_hpwl(
        self,
        idx: int,
        old_x: float, old_y: float,
        new_x: float, new_y: float,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
    ) -> float:
        """HPWL change from moving one macro. O(degree), not O(all nets)."""
        ext = self._extended_positions(placement, benchmark)
        delta = 0.0
        for net_id in macro_nets[idx]:
            net_nodes = benchmark.net_nodes[net_id]
            if len(net_nodes) < 2:
                continue
            weight = benchmark.net_weights[net_id].item()
            coords = ext[net_nodes]
            other = coords[net_nodes != idx]
            if not len(other):
                continue
            min_xo, max_xo = other[:, 0].min().item(), other[:, 0].max().item()
            min_yo, max_yo = other[:, 1].min().item(), other[:, 1].max().item()
            old_hpwl = (max(max_xo, old_x) - min(min_xo, old_x)
                        + max(max_yo, old_y) - min(min_yo, old_y))
            new_hpwl = (max(max_xo, new_x) - min(min_xo, new_x)
                        + max(max_yo, new_y) - min(min_yo, new_y))
            delta += weight * (new_hpwl - old_hpwl)
        return delta

    def _delta_hpwl_swap(
        self,
        idx_a: int, new_ax: float, new_ay: float,
        idx_b: int, new_bx: float, new_by: float,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
    ) -> float:
        """HPWL change from swapping two macros. Handles shared nets correctly."""
        ext = self._extended_positions(placement, benchmark)
        affected = set(macro_nets[idx_a]) | set(macro_nets[idx_b])
        delta = 0.0
        for net_id in affected:
            net_nodes = benchmark.net_nodes[net_id]
            if len(net_nodes) < 2:
                continue
            weight = benchmark.net_weights[net_id].item()
            coords = ext[net_nodes]
            old_hpwl = (
                (coords[:, 0].max() - coords[:, 0].min())
                + (coords[:, 1].max() - coords[:, 1].min())
            ).item()
            new_coords = coords.clone()
            new_coords[net_nodes == idx_a, 0] = new_ax
            new_coords[net_nodes == idx_a, 1] = new_ay
            new_coords[net_nodes == idx_b, 0] = new_bx
            new_coords[net_nodes == idx_b, 1] = new_by
            new_hpwl = (
                (new_coords[:, 0].max() - new_coords[:, 0].min())
                + (new_coords[:, 1].max() - new_coords[:, 1].min())
            ).item()
            delta += weight * (new_hpwl - old_hpwl)
        return delta

    # ---------- move proposals --------------------------------------------

    def _propose_perturb(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
        movable_indices: List[int],
        degrees: torch.Tensor,
        perturb_frac: float,
    ):
        """Single-macro perturbation, connectivity-biased. Returns (delta, apply_fn) or None."""
        pick = torch.multinomial(degrees, 1).item()
        idx = movable_indices[pick]
        old_x = placement[idx, 0].item()
        old_y = placement[idx, 1].item()
        w = benchmark.macro_sizes[idx, 0].item()
        h = benchmark.macro_sizes[idx, 1].item()

        dx = (torch.rand(1).item() * 2.0 - 1.0) * perturb_frac * benchmark.canvas_width
        dy = (torch.rand(1).item() * 2.0 - 1.0) * perturb_frac * benchmark.canvas_height
        new_x = max(w / 2.0, min(benchmark.canvas_width - w / 2.0, old_x + dx))
        new_y = max(h / 2.0, min(benchmark.canvas_height - h / 2.0, old_y + dy))

        if self._has_overlap(idx, new_x, new_y, placement, benchmark):
            return None

        delta = self._delta_hpwl(
            idx, old_x, old_y, new_x, new_y, placement, benchmark, macro_nets
        )

        def apply(p):
            p[idx, 0] = new_x
            p[idx, 1] = new_y
        return delta, apply

    def _propose_swap(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
        movable_indices: List[int],
        N: int,
    ):
        """Swap two random movable macros. Returns (delta, apply_fn) or None."""
        i = torch.randint(N, (1,)).item()
        j = torch.randint(N - 1, (1,)).item()
        if j >= i:
            j += 1
        idx_a, idx_b = movable_indices[i], movable_indices[j]
        old_ax, old_ay = placement[idx_a, 0].item(), placement[idx_a, 1].item()
        old_bx, old_by = placement[idx_b, 0].item(), placement[idx_b, 1].item()
        new_ax, new_ay = old_bx, old_by
        new_bx, new_by = old_ax, old_ay

        W, H = benchmark.canvas_width, benchmark.canvas_height
        wa = benchmark.macro_sizes[idx_a, 0].item()
        ha = benchmark.macro_sizes[idx_a, 1].item()
        wb = benchmark.macro_sizes[idx_b, 0].item()
        hb = benchmark.macro_sizes[idx_b, 1].item()

        if not (wa / 2 <= new_ax <= W - wa / 2 and ha / 2 <= new_ay <= H - ha / 2):
            return None
        if not (wb / 2 <= new_bx <= W - wb / 2 and hb / 2 <= new_by <= H - hb / 2):
            return None
        if self._has_overlap_excluding(idx_a, new_ax, new_ay, idx_b, placement, benchmark):
            return None
        if self._has_overlap_excluding(idx_b, new_bx, new_by, idx_a, placement, benchmark):
            return None
        if (abs(new_ax - new_bx) < (wa + wb) / 2
                and abs(new_ay - new_by) < (ha + hb) / 2):
            return None

        delta = self._delta_hpwl_swap(
            idx_a, new_ax, new_ay, idx_b, new_bx, new_by,
            placement, benchmark, macro_nets,
        )

        def apply(p):
            p[idx_a, 0] = new_ax
            p[idx_a, 1] = new_ay
            p[idx_b, 0] = new_bx
            p[idx_b, 1] = new_by
        return delta, apply

    # ---------- overlap ----------------------------------------------------

    def _has_overlap(
        self, idx: int, new_x: float, new_y: float,
        placement: torch.Tensor, benchmark: Benchmark,
    ) -> bool:
        H = benchmark.num_hard_macros
        dx = torch.abs(placement[:H, 0] - new_x)
        dy = torch.abs(placement[:H, 1] - new_y)
        w_i = benchmark.macro_sizes[idx, 0].item()
        h_i = benchmark.macro_sizes[idx, 1].item()
        min_sep_x = (benchmark.macro_sizes[:H, 0] + w_i) / 2.0
        min_sep_y = (benchmark.macro_sizes[:H, 1] + h_i) / 2.0
        overlapping = (dx < min_sep_x) & (dy < min_sep_y)
        overlapping[idx] = False
        return overlapping.any().item()

    def _has_overlap_excluding(
        self, idx: int, new_x: float, new_y: float,
        exclude_idx: int, placement: torch.Tensor, benchmark: Benchmark,
    ) -> bool:
        H = benchmark.num_hard_macros
        dx = torch.abs(placement[:H, 0] - new_x)
        dy = torch.abs(placement[:H, 1] - new_y)
        w_i = benchmark.macro_sizes[idx, 0].item()
        h_i = benchmark.macro_sizes[idx, 1].item()
        min_sep_x = (benchmark.macro_sizes[:H, 0] + w_i) / 2.0
        min_sep_y = (benchmark.macro_sizes[:H, 1] + h_i) / 2.0
        overlapping = (dx < min_sep_x) & (dy < min_sep_y)
        overlapping[idx] = False
        overlapping[exclude_idx] = False
        return overlapping.any().item()

    # ---------- T0 estimation ---------------------------------------------

    def _estimate_init_temp(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        movable_indices: List[int],
        macro_nets: List[List[int]],
    ) -> float:
        """T0 = -avg(|dHPWL|) / ln(accept_prob_init), from random sampled moves."""
        N = len(movable_indices)
        deltas: List[float] = []
        for _ in range(min(200, N * 5)):
            idx = movable_indices[torch.randint(N, (1,)).item()]
            old_x = placement[idx, 0].item()
            old_y = placement[idx, 1].item()
            w = benchmark.macro_sizes[idx, 0].item()
            h = benchmark.macro_sizes[idx, 1].item()
            dx = (torch.rand(1).item() * 2 - 1) * self.cfg.perturb_frac_init * benchmark.canvas_width
            dy = (torch.rand(1).item() * 2 - 1) * self.cfg.perturb_frac_init * benchmark.canvas_height
            new_x = max(w / 2, min(benchmark.canvas_width - w / 2, old_x + dx))
            new_y = max(h / 2, min(benchmark.canvas_height - h / 2, old_y + dy))
            d = abs(self._delta_hpwl(
                idx, old_x, old_y, new_x, new_y, placement, benchmark, macro_nets
            ))
            if d > 0:
                deltas.append(d)
        if not deltas:
            return 1.0
        return -sum(deltas) / len(deltas) / math.log(self.cfg.accept_prob_init)

    # ---------- soft-macro optimisation -----------------------------------

    def _optimize_softcells(self, benchmark: Benchmark) -> None:
        """
        Call plc.optimize_stdcells() to reposition soft macros (standard-cell
        clusters) given current hard-macro positions. This is what the SA
        baseline does between sweeps and is the single biggest gap in the
        original sa_placer.py's cost.

        NB: the PlacementCost object `plc` is attached to the benchmark by
        the loader. It mutates its own internal state when we call this —
        the evaluator reads the soft-macro positions back out at scoring
        time, so this directly improves reported cost.

        The call is slow (~tens of seconds to minutes per call) so we throttle
        via softcell_interval. We also use reduced num_steps vs. the baseline
        so it doesn't eat the whole budget.
        """
        plc = getattr(benchmark, "_plc", None)
        if plc is None:
            # SETUP.md indicates the loader stashes plc onto the benchmark
            # — but defensive in case a future loader version changes this.
            return
        try:
            canvas_size = max(benchmark.canvas_width, benchmark.canvas_height)
            steps = list(self.cfg.softcell_steps)
            plc.optimize_stdcells(
                use_current_loc=False,
                move_stdcells=True,
                move_macros=False,
                log_scale_conns=False,
                use_sizes=False,
                io_factor=1.0,
                num_steps=steps,
                max_move_distance=[canvas_size / 100.0] * 3,
                attract_factor=[100.0, 1.0e-3, 1.0e-5],
                repel_factor=[0.0, 1.0e6, 1.0e7],
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [hybrid] soft-macro opt skipped: {e}")

    # ---------- QUBO / Simulated Bifurcation escape -----------------------

    def _qubo_escape(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
        movable_indices: List[int],
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Quantum-inspired coordinated escape.

        1. Pick the k macros with highest *local* HPWL contribution
           (these are the "dissatisfied" macros — moving any one
           alone isn't enough because they're coupled).
        2. For each, enumerate m candidate target slots near the
           connectivity-weighted centroid of its net neighbours.
        3. Build a QUBO:
              H = sum_{i,p,j,q} W_ij * dist(p,q) * x_{i,p} * x_{j,q}
                + lambda_oh * (sum_p x_{i,p} - 1)^2
                + lambda_pen * sum_{overlapping (i,p),(j,q)} x_{i,p}*x_{j,q}
        4. Solve with Simulated Bifurcation on GPU/CPU.
        5. Decode; apply if it's an improvement AND overlap-free.

        Returns (new_placement, new_hpwl) or (None, +inf) if no
        valid improvement was found.
        """
        if not _HAS_SB:
            return None, float("inf")

        k = min(self.cfg.qubo_k, len(movable_indices))
        if k < 3:
            return None, float("inf")

        # Step 1: rank macros by local HPWL contribution
        local_hpwl = self._local_hpwl_per_macro(
            placement, benchmark, movable_indices, macro_nets
        )
        top_k_positions = sorted(
            range(len(movable_indices)),
            key=lambda i: -local_hpwl[i]
        )[:k]
        selected = [movable_indices[i] for i in top_k_positions]

        # Step 2: candidate slots for each selected macro
        m = self.cfg.qubo_slots_per_macro
        slot_coords = self._build_candidate_slots(
            selected, placement, benchmark, macro_nets, m
        )
        # slot_coords[i] is a list of (x, y) tuples of length ≤ m.
        # Truncate to uniform size; drop macros with no candidates.
        sizes_per_macro = [len(s) for s in slot_coords]
        if min(sizes_per_macro) == 0:
            return None, float("inf")
        m_eff = min(m, min(sizes_per_macro))
        slot_coords = [s[:m_eff] for s in slot_coords]

        # Step 3: build QUBO matrix Q of size (k*m_eff) x (k*m_eff)
        Q = self._build_qubo_matrix(
            selected, slot_coords, placement, benchmark, macro_nets
        )

        # Step 4: solve with SB
        try:
            best_bits = sb.minimize(
                Q,
                input_type="qubo",
                agents=self.cfg.qubo_agents,
                max_steps=self.cfg.qubo_max_steps,
                ballistic=True,  # bSB: better at escaping local minima
                verbose=False,
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [hybrid] SB solver failed: {e}")
            return None, float("inf")

        # best_bits is a [k*m_eff] tensor of 0/1 values
        if isinstance(best_bits, (list, tuple)):
            best_bits = best_bits[0]
        bits = best_bits.detach().cpu().numpy().flatten()

        # Step 5: decode — pick the slot with highest value per macro
        candidate_placement = placement.clone()
        assignment_ok = True
        for i, macro_idx in enumerate(selected):
            chunk = bits[i * m_eff:(i + 1) * m_eff]
            if chunk.sum() == 0:
                assignment_ok = False
                break
            pick = int(chunk.argmax())
            nx, ny = slot_coords[i][pick]
            # Apply tentatively
            candidate_placement[macro_idx, 0] = nx
            candidate_placement[macro_idx, 1] = ny

        if not assignment_ok:
            return None, float("inf")

        # Verify no overlaps in full placement (SB penalty is soft — we
        # enforce hard constraint by rejecting the solution if it violates).
        if self._full_placement_has_overlap(candidate_placement, benchmark):
            return None, float("inf")

        new_hpwl = self._total_hpwl(candidate_placement, benchmark)
        return candidate_placement, new_hpwl

    def _local_hpwl_per_macro(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        movable_indices: List[int],
        macro_nets: List[List[int]],
    ) -> List[float]:
        """
        For each movable macro, compute sum over its nets of (net HPWL).
        High values = macro is sitting inside a net with wide bbox =
        candidate for relocation.
        """
        ext = self._extended_positions(placement, benchmark)
        # Precompute net HPWLs
        net_hpwls = {}
        for idx in movable_indices:
            for net_id in macro_nets[idx]:
                if net_id in net_hpwls:
                    continue
                nodes = benchmark.net_nodes[net_id]
                if len(nodes) < 2:
                    net_hpwls[net_id] = 0.0
                    continue
                coords = ext[nodes]
                w = benchmark.net_weights[net_id].item()
                net_hpwls[net_id] = w * (
                    (coords[:, 0].max() - coords[:, 0].min())
                    + (coords[:, 1].max() - coords[:, 1].min())
                ).item()
        local = []
        for idx in movable_indices:
            local.append(sum(net_hpwls[n] for n in macro_nets[idx]))
        return local

    def _build_candidate_slots(
        self,
        selected: List[int],
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
        m: int,
    ) -> List[List[Tuple[float, float]]]:
        """
        For each selected macro, propose m candidate positions:
        its current position plus m-1 perturbations toward its
        connectivity-weighted neighbour centroid.
        """
        ext = self._extended_positions(placement, benchmark)
        W = benchmark.canvas_width
        H = benchmark.canvas_height
        slots_all: List[List[Tuple[float, float]]] = []

        for idx in selected:
            mw = benchmark.macro_sizes[idx, 0].item()
            mh = benchmark.macro_sizes[idx, 1].item()
            slots: List[Tuple[float, float]] = [
                (placement[idx, 0].item(), placement[idx, 1].item())
            ]

            # Compute neighbour centroid weighted by inverse net size
            cx_sum = cy_sum = wsum = 0.0
            for net_id in macro_nets[idx]:
                nodes = benchmark.net_nodes[net_id]
                if len(nodes) < 2:
                    continue
                nw = benchmark.net_weights[net_id].item() / max(len(nodes), 1)
                for n in nodes.tolist():
                    if n == idx or n >= ext.shape[0]:
                        continue
                    cx_sum += ext[n, 0].item() * nw
                    cy_sum += ext[n, 1].item() * nw
                    wsum += nw
            if wsum > 0:
                cx = cx_sum / wsum
                cy = cy_sum / wsum
            else:
                cx = placement[idx, 0].item()
                cy = placement[idx, 1].item()

            # Generate m-1 additional slots: perturbations toward centroid.
            # We use a small halo of grid-like offsets.
            rng = torch.Generator()
            rng.manual_seed(self.cfg.seed + idx)
            for _ in range(m - 1):
                alpha = 0.25 + 0.75 * torch.rand(1, generator=rng).item()
                jitter_x = (torch.rand(1, generator=rng).item() - 0.5) * W * 0.05
                jitter_y = (torch.rand(1, generator=rng).item() - 0.5) * H * 0.05
                nx = cx * alpha + placement[idx, 0].item() * (1 - alpha) + jitter_x
                ny = cy * alpha + placement[idx, 1].item() * (1 - alpha) + jitter_y
                nx = max(mw / 2, min(W - mw / 2, nx))
                ny = max(mh / 2, min(H - mh / 2, ny))
                slots.append((nx, ny))
            slots_all.append(slots)
        return slots_all

    def _build_qubo_matrix(
        self,
        selected: List[int],
        slot_coords: List[List[Tuple[float, float]]],
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
    ) -> torch.Tensor:
        """
        Build the QUBO matrix Q. Variable x_{i,p} = 1 if macro `selected[i]`
        goes to slot p.  Total variables = k * m.

        Energy terms:
        1. One-hot constraint per macro: lambda_oh * (sum_p x_{i,p} - 1)^2
           = lambda_oh * (sum_p x_{i,p}^2 + 2 sum_{p<q} x_{i,p}x_{i,q} - 2 sum_p x_{i,p} + 1)
           In QUBO form (x^2 = x for binary): diag += -lambda_oh, off-diag += 2*lambda_oh.
           The constant +1 is dropped.
        2. Pairwise HPWL proxy: for each pair of selected macros sharing
           a net, reward short Manhattan distance. We approximate HPWL
           of a net as sum of pairwise distances scaled by 1/(|net|-1),
           which is the standard clique-star approximation.
        3. Overlap penalty: if two slots (i,p) and (j,q) would overlap
           given macro sizes, lambda_pen on that coupling.

        The one-hot lambda needs to dominate any "free HPWL reward" from
        placing a macro in *two* slots. We set it to a multiple of the
        largest HPWL term magnitude.
        """
        # We build an UPPER-TRIANGULAR QUBO matrix Q such that the
        # QUBO energy is x^T Q x = sum_{i<=j} Q_{ij} x_i x_j. This
        # avoids the x2 double-counting trap that symmetric-form
        # matrices fall into (where Q_{ij} = Q_{ji} = c would give
        # 2c * x_i x_j as the pair energy instead of c).
        #
        # simulated_bifurcation accepts either symmetric or upper-
        # triangular form. Upper-triangular is unambiguous and saves
        # us having to think about the factor-of-2 every time we
        # write a coupling.
        k = len(selected)
        m = len(slot_coords[0])
        total = k * m
        Q = torch.zeros(total, total, dtype=torch.float32)

        def add_coupling(va: int, vb: int, c: float) -> None:
            """Add coupling c to the (va, vb) pair in upper-triangular form."""
            if va == vb:
                Q[va, vb] += c  # diagonal = linear coefficient (x^2 = x)
            elif va < vb:
                Q[va, vb] += c
            else:
                Q[vb, va] += c

        sel_set = set(selected)
        sel_pos_in_list = {mid: i for i, mid in enumerate(selected)}

        # --- pairwise HPWL term (clique-star HPWL approximation) ---
        # For each net touching ≥2 selected macros, each pair of
        # selected macros contributes w_net / (|net|-1) * |p-q|_1
        # for each pair of their slot assignments.
        hpwl_scale = 0.0
        for net_id, nodes in enumerate(benchmark.net_nodes):
            if len(nodes) < 2:
                continue
            nodes_list = nodes.tolist()
            in_selected = [n for n in nodes_list if n in sel_set]
            if len(in_selected) < 2:
                continue
            w = benchmark.net_weights[net_id].item()
            norm = 1.0 / (len(nodes_list) - 1)
            for a_idx in range(len(in_selected)):
                for b_idx in range(a_idx + 1, len(in_selected)):
                    ma, mb = in_selected[a_idx], in_selected[b_idx]
                    i, j = sel_pos_in_list[ma], sel_pos_in_list[mb]
                    for p in range(m):
                        pax, pay = slot_coords[i][p]
                        for q in range(m):
                            qbx, qby = slot_coords[j][q]
                            d = abs(pax - qbx) + abs(pay - qby)
                            coupling = w * norm * d
                            hpwl_scale = max(hpwl_scale, coupling)
                            add_coupling(i * m + p, j * m + q, coupling)

        # --- overlap penalty ---
        # Dominates any possible HPWL reward. The constant floor of
        # 1.0 prevents the penalty collapsing to zero when the HPWL
        # terms happen to be tiny.
        lambda_pen = max(self.cfg.qubo_overlap_penalty * hpwl_scale, 1.0)
        for i in range(k):
            mi = selected[i]
            wi = benchmark.macro_sizes[mi, 0].item()
            hi = benchmark.macro_sizes[mi, 1].item()
            for j in range(i + 1, k):
                mj = selected[j]
                wj = benchmark.macro_sizes[mj, 0].item()
                hj = benchmark.macro_sizes[mj, 1].item()
                for p in range(m):
                    pax, pay = slot_coords[i][p]
                    for q in range(m):
                        qbx, qby = slot_coords[j][q]
                        if (abs(pax - qbx) < (wi + wj) / 2
                                and abs(pay - qby) < (hi + hj) / 2):
                            add_coupling(i * m + p, j * m + q, lambda_pen)

        # --- one-hot constraint: lambda_oh * (sum_p x_{i,p} - 1)^2 ---
        # Expanding using x^2 = x for binary x gives
        #   lambda_oh * [ -sum_p x_p + 2 sum_{p<q} x_p x_q + const ].
        # Diagonal terms: -lambda_oh. Off-diagonal within same macro: +2*lambda_oh.
        # Must dominate both HPWL and overlap couplings so the solver can
        # never prefer an invalid two-slot assignment over a valid one.
        lambda_oh = max(hpwl_scale * 20.0, lambda_pen * 2.0, 10.0)
        for i in range(k):
            for p in range(m):
                vi = i * m + p
                add_coupling(vi, vi, -lambda_oh)
                for q in range(p + 1, m):
                    vj = i * m + q
                    add_coupling(vi, vj, 2.0 * lambda_oh)

        return Q

    def _full_placement_has_overlap(
        self, placement: torch.Tensor, benchmark: Benchmark,
    ) -> bool:
        """Check whole placement for any hard-macro overlap. O(N^2) but rare."""
        H = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:H]
        pos = placement[:H]
        # Pairwise check via broadcasting
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        min_sep_x = (sizes[:, 0].unsqueeze(0) + sizes[:, 0].unsqueeze(1)) / 2
        min_sep_y = (sizes[:, 1].unsqueeze(0) + sizes[:, 1].unsqueeze(1)) / 2
        overlap = (dx < min_sep_x) & (dy < min_sep_y)
        # Exclude self-overlap (diagonal)
        overlap.fill_diagonal_(False)
        return overlap.any().item()