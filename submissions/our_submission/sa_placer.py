"""
SA Placer — Our Submission for the Partcl/HRT Macro Placement Challenge
=======================================================================

Algorithm
---------
Simulated Annealing with two move types and connectivity-biased selection.

Initialisation
  Greedy shelf-pack (tallest-first rows) → guaranteed zero overlaps.

SA loop (time-based cooling schedule + periodic reheating)
  Move selection — 60 % perturbation, 40 % swap:

  Perturbation move (single macro)
    Pick a macro with probability ∝ its connectivity degree.
    High-degree macros affect more nets, so moving them has higher leverage.
    Propose a random (dx, dy) clamped to canvas bounds.
    Vectorised O(N) overlap check → reject if any hard-macro overlap.
    Incremental O(degree) delta-HPWL.
    Metropolis accept / reject.

  Swap move (two macros)
    Pick two random movable macros and exchange their centre positions.
    Directly fixes cases where macro A belongs where B is and vice versa.
    Overlap-checked without modifying placement (exclude-index variant).
    Delta-HPWL computed over the union of both macros' nets in one pass,
    handling nets that contain BOTH macros correctly.

Temperature schedule
  Time-based exponential decay: T(t) = T0 * cooling_alpha ^ (t / time_limit)
  With cooling_alpha=0.001, T drops to 0.1 % of T0 by end of budget.
  This is robust across benchmarks regardless of move rate — the schedule is
  wall-clock-anchored, not step-count-anchored.

Reheating
  Track acceptance rate over the last 500 moves.
  If it drops below 1 %, multiply T (via reheat_mult) by 4 × (up to 5 ×).
  This lets SA escape deep local minima late in the schedule.

Cost
  Primary SA objective: weighted HPWL (weight = benchmark.net_weights).
  Density and congestion not tracked during SA — their improvement comes
  indirectly from better macro spreading driven by WL optimisation.
  Final proxy cost (WL + density + congestion) is evaluated by the harness.

Runtime
  Each benchmark is allowed up to 1 hour (3600 s).  We use 3300 s to leave
  a safety buffer for loading, validation, and proxy cost computation.

Usage (from repo root)
----------------------
    uv run evaluate submissions/our_submission/sa_placer.py -b ibm01
    uv run evaluate submissions/our_submission/sa_placer.py --all
    uv run evaluate submissions/our_submission/sa_placer.py --all --vis
"""

import math
import time
from collections import deque
from typing import List, Optional, Tuple

import torch

from macro_place.benchmark import Benchmark


class SAPlacer:
    """
    Simulated Annealing macro placer with swap moves and connectivity bias.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    time_limit : float
        Wall-clock seconds per benchmark.  Rule: 1 hour (3600 s) per benchmark.
        We use 3300 s to leave a 5-minute buffer for loading and evaluation.
    cooling_alpha : float
        Time-based temperature end ratio: T(end) = T0 * cooling_alpha.
        0.001 means temperature drops to 0.1% of T0 by the end of time_limit.
        (Replaces the old step-based cooling_rate — more robust across designs.)
    perturb_frac_init : float
        Initial perturbation size as fraction of canvas (0.30 = 30 %).
    perturb_frac_final : float
        Final perturbation fraction at end of time budget (0.005 = 0.5 %).
    swap_prob : float
        Fraction of moves that are swap moves (rest are perturbations).
    accept_prob_init : float
        Target initial Metropolis acceptance rate used to set T₀.
    reheat_threshold : float
        Reheat when recent acceptance drops below this fraction (0.01 = 1 %).
    max_reheats : int
        Maximum number of reheats allowed per benchmark.
    verbose : bool
        Print per-benchmark statistics.
    """

    def __init__(
        self,
        seed: int = 42,
        time_limit: float = 3300.0,
        cooling_alpha: float = 0.001,
        perturb_frac_init: float = 0.30,
        perturb_frac_final: float = 0.005,
        swap_prob: float = 0.40,
        accept_prob_init: float = 0.80,
        reheat_threshold: float = 0.01,
        max_reheats: int = 5,
        verbose: bool = True,
    ):
        self.seed = seed
        self.time_limit = time_limit
        self.cooling_alpha = cooling_alpha
        self.perturb_frac_init = perturb_frac_init
        self.perturb_frac_final = perturb_frac_final
        self.swap_prob = swap_prob
        self.accept_prob_init = accept_prob_init
        self.reheat_threshold = reheat_threshold
        self.max_reheats = max_reheats
        self.verbose = verbose

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Run SA on *benchmark*, return a [num_macros, 2] placement tensor.

        Fixed macros stay at their original positions.
        Zero hard-macro overlaps guaranteed on return.
        """
        torch.manual_seed(self.seed)
        t_start = time.time()

        # ── 1. Zero-overlap initialisation ──────────────────────────────────
        placement = self._greedy_row_init(benchmark)

        # ── 2. Index structures ──────────────────────────────────────────────
        movable_indices = self._get_movable_indices(benchmark)
        N = len(movable_indices)
        if N == 0:
            return placement

        # macro_nets[i] = net IDs containing macro i  (reverse adjacency)
        macro_nets = self._build_macro_nets(benchmark)

        # Degree of each movable macro — used for connectivity-biased selection.
        # We add 1.0 so isolated macros (degree 0) still have a chance to move.
        degrees = torch.tensor(
            [len(macro_nets[i]) + 1.0 for i in movable_indices],
            dtype=torch.float32,
        )

        # ── 3. Initial temperature ───────────────────────────────────────────
        T_start = self._estimate_init_temp(placement, benchmark, movable_indices, macro_nets)
        # Time-based temperature: T(t) = T_start * cooling_alpha ^ progress
        # Reheating adds a separate multiplier on top of the schedule.
        reheat_mult = 1.0

        # ── 4. SA loop (flat — no inner temperature-step loop) ───────────────
        # Each iteration is one move.  Temperature is anchored to wall-clock
        # time, so the schedule is independent of move speed / benchmark size.
        current_hpwl = self._total_hpwl(placement, benchmark)
        best_placement = placement.clone()
        best_hpwl = current_hpwl

        accepted = rejected = total_moves = 0
        reheat_count = 0
        # Rolling window to track recent acceptance rate (for reheating).
        # Larger window (500) gives a more stable signal over long runs.
        accept_window: deque = deque(maxlen=500)

        log_alpha = math.log(self.cooling_alpha)   # precompute once

        while True:
            elapsed = time.time() - t_start
            if elapsed >= self.time_limit:
                break

            progress = elapsed / self.time_limit

            # Time-based temperature (wall-clock-anchored exponential decay)
            T = T_start * math.exp(log_alpha * progress) * reheat_mult
            if T <= 0.0:
                T = 1e-12

            # Linearly anneal perturbation size: large early → tiny late
            perturb_frac = (
                self.perturb_frac_init * (1.0 - progress)
                + self.perturb_frac_final * progress
            )

            # ── Decide move type ─────────────────────────────────────────────
            use_swap = (N >= 2) and (torch.rand(1).item() < self.swap_prob)

            if use_swap:
                result = self._swap_move(
                    placement, benchmark, macro_nets, movable_indices, N
                )
                if result is None:
                    # Swap was geometrically invalid (bounds / overlap)
                    accept_window.append(False)
                    rejected += 1
                    total_moves += 1
                    continue

                delta, idx_a, new_ax, new_ay, idx_b, new_bx, new_by = result

                if delta < 0.0 or torch.rand(1).item() < math.exp(-delta / T):
                    placement[idx_a, 0] = new_ax
                    placement[idx_a, 1] = new_ay
                    placement[idx_b, 0] = new_bx
                    placement[idx_b, 1] = new_by
                    current_hpwl += delta
                    accepted += 1
                    accept_window.append(True)
                    if current_hpwl < best_hpwl:
                        best_hpwl = current_hpwl
                        best_placement = placement.clone()
                else:
                    rejected += 1
                    accept_window.append(False)

            else:
                # ── Perturbation move ────────────────────────────────────────
                # Connectivity-biased selection: macros in more nets are
                # sampled proportionally more often because moving them has
                # higher expected HPWL impact.
                pick = torch.multinomial(degrees, 1).item()
                idx = movable_indices[pick]
                old_x = placement[idx, 0].item()
                old_y = placement[idx, 1].item()
                w = benchmark.macro_sizes[idx, 0].item()
                h = benchmark.macro_sizes[idx, 1].item()

                dx = (torch.rand(1).item() * 2.0 - 1.0) * perturb_frac * benchmark.canvas_width
                dy = (torch.rand(1).item() * 2.0 - 1.0) * perturb_frac * benchmark.canvas_height
                new_x = max(w / 2.0, min(benchmark.canvas_width  - w / 2.0, old_x + dx))
                new_y = max(h / 2.0, min(benchmark.canvas_height - h / 2.0, old_y + dy))

                if self._has_overlap(idx, new_x, new_y, placement, benchmark):
                    rejected += 1
                    accept_window.append(False)
                    total_moves += 1
                    continue

                delta = self._delta_hpwl(
                    idx, old_x, old_y, new_x, new_y,
                    placement, benchmark, macro_nets,
                )

                if delta < 0.0 or torch.rand(1).item() < math.exp(-delta / T):
                    placement[idx, 0] = new_x
                    placement[idx, 1] = new_y
                    current_hpwl += delta
                    accepted += 1
                    accept_window.append(True)
                    if current_hpwl < best_hpwl:
                        best_hpwl = current_hpwl
                        best_placement = placement.clone()
                else:
                    rejected += 1
                    accept_window.append(False)

            total_moves += 1

            # ── Reheating ────────────────────────────────────────────────────
            # When acceptance drops to nearly zero, SA is stuck in a local basin.
            # Raise T (via reheat_mult) and restart from best known placement.
            if (
                reheat_count < self.max_reheats
                and len(accept_window) == accept_window.maxlen
                and sum(accept_window) / accept_window.maxlen < self.reheat_threshold
            ):
                reheat_mult *= 4.0
                reheat_count += 1
                placement = best_placement.clone()
                current_hpwl = best_hpwl
                accept_window.clear()
                if self.verbose:
                    print(
                        f"  [{benchmark.name}] reheat #{reheat_count} at "
                        f"{elapsed:.0f}s | T={T:.4g} → {T * 4:.4g} | "
                        f"best HPWL={best_hpwl:.4f}"
                    )

        if self.verbose:
            print(
                f"  [{benchmark.name}] {total_moves:,} moves | "
                f"{accepted / max(total_moves,1)*100:.1f}% accept | "
                f"{reheat_count} reheats | HPWL {best_hpwl:.4f} | "
                f"{time.time()-t_start:.1f}s"
            )

        return best_placement

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _greedy_row_init(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Shelf-pack movable hard macros into rows, tallest-first.

        Sorts macros by descending height so each row's height is determined
        by its first (tallest) member.  Fills left-to-right; starts a new row
        when the next macro doesn't fit.  Mathematically cannot produce overlaps.

        Fixed macros and soft macros stay at their benchmark initial positions.
        """
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes

        indices.sort(key=lambda i: -sizes[i, 1].item())   # tallest first

        gap = 0.001          # prevents float32 touching-edge false overlaps
        cursor_x = cursor_y = row_h = 0.0

        for idx in indices:
            w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
            if cursor_x + w > benchmark.canvas_width:
                cursor_x = 0.0
                cursor_y += row_h + gap
                row_h = 0.0
            if cursor_y + h > benchmark.canvas_height:   # fallback
                placement[idx, 0] = w / 2.0
                placement[idx, 1] = h / 2.0
                continue
            placement[idx, 0] = cursor_x + w / 2.0
            placement[idx, 1] = cursor_y + h / 2.0
            cursor_x += w + gap
            row_h = max(row_h, h)

        return placement

    # -----------------------------------------------------------------------
    # Index structures
    # -----------------------------------------------------------------------

    def _get_movable_indices(self, benchmark: Benchmark) -> List[int]:
        """Hard macro tensor indices that SA is allowed to move."""
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        return torch.where(movable)[0].tolist()

    def _build_macro_nets(self, benchmark: Benchmark) -> List[List[int]]:
        """
        Reverse adjacency: macro_nets[i] = list of net IDs containing macro i.

        Inverts benchmark.net_nodes so we can find affected nets for a given
        macro in O(1) instead of scanning all nets.

        Nodes with index ≥ num_macros are I/O ports (fixed).  We skip them
        here — they appear in net bboxes automatically via port_positions.
        """
        macro_nets: List[List[int]] = [[] for _ in range(benchmark.num_macros)]
        for net_id, net_nodes in enumerate(benchmark.net_nodes):
            for node_idx in net_nodes.tolist():
                if 0 <= node_idx < benchmark.num_macros:
                    macro_nets[node_idx].append(net_id)
        return macro_nets

    # -----------------------------------------------------------------------
    # Cost functions
    # -----------------------------------------------------------------------

    def _extended_positions(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        """
        [macro positions | port positions] as one lookup tensor.

        net_nodes indices ≥ num_macros refer to I/O ports.  Appending
        benchmark.port_positions lets ext[net_nodes] work uniformly.
        """
        if benchmark.port_positions.shape[0] > 0:
            return torch.cat([placement, benchmark.port_positions], dim=0)
        return placement

    def _total_hpwl(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        """
        Weighted total HPWL across all nets.  Called once at SA start.

        HPWL(net) = (max_x − min_x) + (max_y − min_y) over all pins,
        including I/O ports at the canvas boundary.
        """
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

    def _delta_hpwl(
        self,
        idx: int,
        old_x: float, old_y: float,
        new_x: float, new_y: float,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
    ) -> float:
        """
        HPWL change from moving one macro.  O(degree), not O(all nets).

        For each net touching macro idx:
          1. Get all OTHER nodes' coords (includes ports via extended positions).
          2. Old HPWL = extend their bbox with (old_x, old_y).
          3. New HPWL = extend their bbox with (new_x, new_y).
          4. Accumulate weight × (new − old).
        """
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
        """
        HPWL change from swapping two macros simultaneously.

        Visits the UNION of both macros' nets in one pass.  Nets containing
        BOTH macros are handled correctly: we replace both positions at once
        before computing the new bbox, rather than treating the swap as two
        independent single-macro moves (which would double-count shared nets).
        """
        ext = self._extended_positions(placement, benchmark)
        affected = set(macro_nets[idx_a]) | set(macro_nets[idx_b])
        delta = 0.0

        for net_id in affected:
            net_nodes = benchmark.net_nodes[net_id]
            if len(net_nodes) < 2:
                continue
            weight = benchmark.net_weights[net_id].item()
            coords = ext[net_nodes]   # [K, 2] — current positions

            # Old HPWL with current positions
            old_hpwl = (
                (coords[:, 0].max() - coords[:, 0].min())
                + (coords[:, 1].max() - coords[:, 1].min())
            ).item()

            # Build new coords: replace idx_a and idx_b positions in one shot.
            # coords is already a gathered view; clone it to avoid modifying ext.
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

    # -----------------------------------------------------------------------
    # Move generators
    # -----------------------------------------------------------------------

    def _swap_move(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_nets: List[List[int]],
        movable_indices: List[int],
        N: int,
    ) -> Optional[Tuple[float, int, float, float, int, float, float]]:
        """
        Propose swapping the centre positions of two randomly chosen macros.

        Returns
        -------
        None if the swap is geometrically invalid (out of bounds or overlapping).
        Otherwise: (delta_hpwl, idx_a, new_ax, new_ay, idx_b, new_bx, new_by).

        Overlap check strategy (no temporary placement modification)
        ------------------------------------------------------------
        After swapping, macro A occupies B's old centre and vice versa.
        We check:
          1. Canvas bounds for both macros at their new centres.
          2. A's new position vs all other hard macros, EXCLUDING B
             (B has moved away from that position too).
          3. B's new position vs all other hard macros, EXCLUDING A.
          4. Whether A and B overlap EACH OTHER at their new positions
             (possible when they have very different sizes).
        All checks use the ORIGINAL placement, so no rollback is needed.
        """
        # Pick two distinct movable macros uniformly at random
        i = torch.randint(N, (1,)).item()
        j = torch.randint(N - 1, (1,)).item()
        if j >= i:
            j += 1
        idx_a = movable_indices[i]
        idx_b = movable_indices[j]

        old_ax, old_ay = placement[idx_a, 0].item(), placement[idx_a, 1].item()
        old_bx, old_by = placement[idx_b, 0].item(), placement[idx_b, 1].item()

        # After swap: A goes to B's old centre, B goes to A's old centre
        new_ax, new_ay = old_bx, old_by
        new_bx, new_by = old_ax, old_ay

        W, H = benchmark.canvas_width, benchmark.canvas_height
        wa = benchmark.macro_sizes[idx_a, 0].item()
        ha = benchmark.macro_sizes[idx_a, 1].item()
        wb = benchmark.macro_sizes[idx_b, 0].item()
        hb = benchmark.macro_sizes[idx_b, 1].item()

        # ── 1. Canvas bounds ─────────────────────────────────────────────────
        if not (wa / 2 <= new_ax <= W - wa / 2 and ha / 2 <= new_ay <= H - ha / 2):
            return None
        if not (wb / 2 <= new_bx <= W - wb / 2 and hb / 2 <= new_by <= H - hb / 2):
            return None

        # ── 2. A's new position vs everyone except B ─────────────────────────
        if self._has_overlap_excluding(idx_a, new_ax, new_ay, idx_b, placement, benchmark):
            return None

        # ── 3. B's new position vs everyone except A ─────────────────────────
        if self._has_overlap_excluding(idx_b, new_bx, new_by, idx_a, placement, benchmark):
            return None

        # ── 4. A and B overlap each other at new positions ───────────────────
        # (Same-size macros are safe — their new positions were each other's
        # old positions, which were already valid.  Different sizes need checking.)
        if abs(new_ax - new_bx) < (wa + wb) / 2 and abs(new_ay - new_by) < (ha + hb) / 2:
            return None

        delta = self._delta_hpwl_swap(
            idx_a, new_ax, new_ay,
            idx_b, new_bx, new_by,
            placement, benchmark, macro_nets,
        )
        return delta, idx_a, new_ax, new_ay, idx_b, new_bx, new_by

    # -----------------------------------------------------------------------
    # Overlap detection
    # -----------------------------------------------------------------------

    def _has_overlap(
        self,
        idx: int,
        new_x: float, new_y: float,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> bool:
        """
        Vectorised hard-macro overlap check.  One tensor op, no Python loop.

        Rectangles overlap iff  |Δcx| < (w_i + w_j)/2  AND  |Δcy| < (h_i + h_j)/2.
        """
        H = benchmark.num_hard_macros
        dx = torch.abs(placement[:H, 0] - new_x)
        dy = torch.abs(placement[:H, 1] - new_y)
        w_i, h_i = benchmark.macro_sizes[idx, 0].item(), benchmark.macro_sizes[idx, 1].item()
        min_sep_x = (benchmark.macro_sizes[:H, 0] + w_i) / 2.0
        min_sep_y = (benchmark.macro_sizes[:H, 1] + h_i) / 2.0
        overlapping = (dx < min_sep_x) & (dy < min_sep_y)
        overlapping[idx] = False
        return overlapping.any().item()

    def _has_overlap_excluding(
        self,
        idx: int,
        new_x: float, new_y: float,
        exclude_idx: int,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> bool:
        """
        Same as _has_overlap but also excludes a second macro from the check.

        Used in swap moves: when checking macro A's new position, we exclude
        macro B because B is simultaneously moving away from that area.
        """
        H = benchmark.num_hard_macros
        dx = torch.abs(placement[:H, 0] - new_x)
        dy = torch.abs(placement[:H, 1] - new_y)
        w_i, h_i = benchmark.macro_sizes[idx, 0].item(), benchmark.macro_sizes[idx, 1].item()
        min_sep_x = (benchmark.macro_sizes[:H, 0] + w_i) / 2.0
        min_sep_y = (benchmark.macro_sizes[:H, 1] + h_i) / 2.0
        overlapping = (dx < min_sep_x) & (dy < min_sep_y)
        overlapping[idx] = False
        overlapping[exclude_idx] = False
        return overlapping.any().item()

    # -----------------------------------------------------------------------
    # Temperature estimation
    # -----------------------------------------------------------------------

    def _estimate_init_temp(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        movable_indices: List[int],
        macro_nets: List[List[int]],
    ) -> float:
        """
        Set T₀ so average bad perturbation has accept_prob_init chance.

        Samples 200 random moves (ignoring overlaps — cost landscape only).
        avg_|ΔHPWL| → T₀ = −avg / ln(accept_prob_init).
        """
        N = len(movable_indices)
        deltas: List[float] = []

        for _ in range(min(200, N * 5)):
            idx = movable_indices[torch.randint(N, (1,)).item()]
            old_x = placement[idx, 0].item()
            old_y = placement[idx, 1].item()
            w, h = benchmark.macro_sizes[idx, 0].item(), benchmark.macro_sizes[idx, 1].item()
            dx = (torch.rand(1).item() * 2.0 - 1.0) * self.perturb_frac_init * benchmark.canvas_width
            dy = (torch.rand(1).item() * 2.0 - 1.0) * self.perturb_frac_init * benchmark.canvas_height
            new_x = max(w / 2.0, min(benchmark.canvas_width  - w / 2.0, old_x + dx))
            new_y = max(h / 2.0, min(benchmark.canvas_height - h / 2.0, old_y + dy))
            d = abs(self._delta_hpwl(idx, old_x, old_y, new_x, new_y,
                                     placement, benchmark, macro_nets))
            if d > 0.0:
                deltas.append(d)

        if not deltas:
            return 1.0
        return -sum(deltas) / len(deltas) / math.log(self.accept_prob_init)
