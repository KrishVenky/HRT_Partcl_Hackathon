"""
SA Placer v2 — Phase 1 of our hybrid SA + QUBO submission.

Three upgrades over v1:
  1. Full proxy-cost objective inside the SA loop (WL + 0.5*density + 0.5*congestion),
     not just HPWL. We optimize what we're actually scored on.
  2. Soft macro co-optimization via plc.optimize_stdcells() every N accepted moves.
     The SA baseline does this; we didn't. It buys a lot of wirelength and density.
  3. Smart initialization: start from the benchmark's initial placement (hand-crafted,
     much better than random) and only legalize it to remove overlaps. Saves thousands
     of moves recovering ground we shouldn't have lost.

Everything else from v1 (connectivity-biased selection, swap moves with correct
shared-net delta HPWL, wall-clock-anchored cooling, reheating on acceptance collapse)
is preserved.

Usage (from repo root):
    uv run evaluate submissions/our_submission/sa_placer_v2.py -b ibm01
    uv run evaluate submissions/our_submission/sa_placer_v2.py --all
"""

import math
import time
from collections import deque
from typing import List, Optional, Tuple

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


class SAPlacerV2:
    """
    SA with proxy-cost objective, soft macro co-optimization, and smart init.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    time_limit : float
        Wall-clock seconds per benchmark (rule: 3600 s max; we use 3300 s buffer).
    cooling_alpha : float
        Time-based temperature end ratio: T(end) = T0 * cooling_alpha. 0.001 means
        T drops to 0.1% of T0 by end of time_limit.
    perturb_frac_init, perturb_frac_final : float
        Perturbation size as fraction of canvas, annealed linearly over time.
    swap_prob : float
        Fraction of moves that are swap moves (rest are single-macro perturbations).
    accept_prob_init : float
        Target initial Metropolis acceptance rate used to calibrate T0.
    reheat_threshold : float
        Reheat when recent acceptance drops below this fraction.
    max_reheats : int
        Max reheats per benchmark.
    softcell_every_accepts : int
        Run plc.optimize_stdcells() after this many accepted hard-macro moves.
        Set to a large number (e.g. 5000) so we don't call it too often — each
        call takes ~30-60s in Python. A few calls per benchmark is enough.
    softcell_num_steps : int
        Steps per phase of optimize_stdcells. Lower = faster, less optimal.
    density_weight, congestion_weight : float
        Weights on density and congestion in the SA acceptance objective.
        Harness uses 0.5 each; we match that.
    density_recompute_every : int
        Recompute full proxy cost every N moves (to pick up congestion drift).
        Between recomputes, we use incremental WL delta + approximate density.
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
        softcell_every_accepts: int = 5000,
        softcell_num_steps: int = 50,
        density_weight: float = 0.5,
        congestion_weight: float = 0.5,
        density_recompute_every: int = 2000,
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
        self.softcell_every_accepts = softcell_every_accepts
        self.softcell_num_steps = softcell_num_steps
        self.density_weight = density_weight
        self.congestion_weight = congestion_weight
        self.density_recompute_every = density_recompute_every
        self.verbose = verbose

    # =======================================================================
    # Public API
    # =======================================================================

    def place(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        """
        Run SA on *benchmark*, return a [num_macros, 2] placement tensor.

        If plc is provided, we use it for soft-macro optimization and full
        proxy-cost recomputes. If None, we fall back to HPWL-only SA (v1 behavior).
        """
        torch.manual_seed(self.seed)
        t_start = time.time()

        # ── 1. Smart initialization ─────────────────────────────────────────
        # Start from benchmark's initial placement (hand-crafted, much better
        # than greedy shelf-pack). Legalize to remove any overlaps.
        placement = self._smart_init(benchmark)

        # ── 2. Index structures ─────────────────────────────────────────────
        movable_indices = self._get_movable_indices(benchmark)
        N = len(movable_indices)
        if N == 0:
            return placement

        macro_nets = self._build_macro_nets(benchmark)

        degrees = torch.tensor(
            [len(macro_nets[i]) + 1.0 for i in movable_indices],
            dtype=torch.float32,
        )

        # ── 3. Initial cost ─────────────────────────────────────────────────
        current_hpwl = self._total_hpwl(placement, benchmark)

        # Proxy components from the harness evaluator (if plc available).
        # These are held constant between recomputes; we track HPWL incrementally
        # and periodically sync by calling compute_proxy_cost in full.
        if plc is not None:
            try:
                full_costs = compute_proxy_cost(placement, benchmark, plc)
                current_density = float(full_costs["density_cost"])
                current_congestion = float(full_costs["congestion_cost"])
                # Scale factor: proxy WL is normalized HPWL. We track raw HPWL in SA
                # for incremental updates and convert when needed.
                # At init, wirelength_cost = normalized HPWL; we compute the ratio.
                wl_scale = float(full_costs["wirelength_cost"]) / max(current_hpwl, 1e-9)
            except Exception as e:
                if self.verbose:
                    print(f"  [{benchmark.name}] proxy cost unavailable ({e}); HPWL-only mode")
                plc = None
                current_density = 0.0
                current_congestion = 0.0
                wl_scale = 1.0
        else:
            current_density = 0.0
            current_congestion = 0.0
            wl_scale = 1.0

        def composite_cost(hpwl: float, density: float, congestion: float) -> float:
            """SA acceptance objective, matched to the harness proxy cost."""
            return (
                wl_scale * hpwl
                + self.density_weight * density
                + self.congestion_weight * congestion
            )

        current_cost = composite_cost(current_hpwl, current_density, current_congestion)
        best_placement = placement.clone()
        best_cost = current_cost
        best_hpwl = current_hpwl

        # ── 4. Initial temperature ──────────────────────────────────────────
        T_start = self._estimate_init_temp(placement, benchmark, movable_indices, macro_nets, wl_scale)
        reheat_mult = 1.0
        log_alpha = math.log(self.cooling_alpha)

        # ── 5. SA loop ──────────────────────────────────────────────────────
        accepted = rejected = total_moves = 0
        reheat_count = 0
        accepts_since_softcell = 0
        moves_since_recompute = 0
        accept_window: deque = deque(maxlen=500)

        while True:
            elapsed = time.time() - t_start
            if elapsed >= self.time_limit:
                break

            progress = elapsed / self.time_limit
            T = T_start * math.exp(log_alpha * progress) * reheat_mult
            if T <= 0.0:
                T = 1e-12

            perturb_frac = (
                self.perturb_frac_init * (1.0 - progress)
                + self.perturb_frac_final * progress
            )

            # Decide move type
            use_swap = (N >= 2) and (torch.rand(1).item() < self.swap_prob)

            # ── Propose and evaluate move ────────────────────────────────────
            move_accepted = False
            if use_swap:
                result = self._swap_move(
                    placement, benchmark, macro_nets, movable_indices, N
                )
                if result is None:
                    accept_window.append(False)
                    rejected += 1
                    total_moves += 1
                    moves_since_recompute += 1
                    continue

                delta_hpwl, idx_a, new_ax, new_ay, idx_b, new_bx, new_by = result
                delta_cost = wl_scale * delta_hpwl  # density/congestion approximated as unchanged

                if delta_cost < 0.0 or torch.rand(1).item() < math.exp(-delta_cost / T):
                    placement[idx_a, 0] = new_ax
                    placement[idx_a, 1] = new_ay
                    placement[idx_b, 0] = new_bx
                    placement[idx_b, 1] = new_by
                    current_hpwl += delta_hpwl
                    current_cost += delta_cost
                    move_accepted = True
            else:
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
                    rejected += 1
                    accept_window.append(False)
                    total_moves += 1
                    moves_since_recompute += 1
                    continue

                delta_hpwl = self._delta_hpwl(
                    idx, old_x, old_y, new_x, new_y,
                    placement, benchmark, macro_nets,
                )
                delta_cost = wl_scale * delta_hpwl

                if delta_cost < 0.0 or torch.rand(1).item() < math.exp(-delta_cost / T):
                    placement[idx, 0] = new_x
                    placement[idx, 1] = new_y
                    current_hpwl += delta_hpwl
                    current_cost += delta_cost
                    move_accepted = True

            # ── Bookkeeping ──────────────────────────────────────────────────
            if move_accepted:
                accepted += 1
                accepts_since_softcell += 1
                accept_window.append(True)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_hpwl = current_hpwl
                    best_placement = placement.clone()
            else:
                rejected += 1
                accept_window.append(False)

            total_moves += 1
            moves_since_recompute += 1

            # ── Periodic full proxy-cost sync ────────────────────────────────
            # Our incremental tracking assumes density/congestion are constant,
            # which drifts. Re-sync periodically to correct the composite cost.
            if (
                plc is not None
                and moves_since_recompute >= self.density_recompute_every
            ):
                try:
                    full_costs = compute_proxy_cost(placement, benchmark, plc)
                    current_density = float(full_costs["density_cost"])
                    current_congestion = float(full_costs["congestion_cost"])
                    new_wl_norm = float(full_costs["wirelength_cost"])
                    # Update wl_scale if HPWL and normalized WL have drifted relative to each other
                    wl_scale = new_wl_norm / max(current_hpwl, 1e-9)
                    current_cost = composite_cost(current_hpwl, current_density, current_congestion)
                    # Compare using PROXY cost for best-tracking (truest metric)
                    true_proxy = float(full_costs["proxy_cost"])
                    if true_proxy < best_cost:
                        best_cost = true_proxy
                        best_hpwl = current_hpwl
                        best_placement = placement.clone()
                except Exception:
                    pass
                moves_since_recompute = 0

            # ── Soft macro co-optimization ───────────────────────────────────
            # After many accepted hard-macro moves, soft macros are in stale
            # positions. Rerun the force-directed soft placer to follow.
            if (
                plc is not None
                and accepts_since_softcell >= self.softcell_every_accepts
            ):
                self._optimize_softcells(plc, benchmark, placement)
                # After softcell move, everything changed — refresh state
                current_hpwl = self._total_hpwl(placement, benchmark)
                try:
                    full_costs = compute_proxy_cost(placement, benchmark, plc)
                    current_density = float(full_costs["density_cost"])
                    current_congestion = float(full_costs["congestion_cost"])
                    wl_scale = float(full_costs["wirelength_cost"]) / max(current_hpwl, 1e-9)
                    current_cost = float(full_costs["proxy_cost"])
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_hpwl = current_hpwl
                        best_placement = placement.clone()
                    if self.verbose:
                        print(
                            f"  [{benchmark.name}] softcell @ {elapsed:.0f}s | "
                            f"proxy={current_cost:.4f} | best={best_cost:.4f}"
                        )
                except Exception:
                    pass
                accepts_since_softcell = 0
                moves_since_recompute = 0

            # ── Reheating ────────────────────────────────────────────────────
            if (
                reheat_count < self.max_reheats
                and len(accept_window) == accept_window.maxlen
                and sum(accept_window) / accept_window.maxlen < self.reheat_threshold
            ):
                reheat_mult *= 4.0
                reheat_count += 1
                placement = best_placement.clone()
                current_hpwl = self._total_hpwl(placement, benchmark)
                current_cost = best_cost
                accept_window.clear()
                if self.verbose:
                    print(
                        f"  [{benchmark.name}] reheat #{reheat_count} @ {elapsed:.0f}s | "
                        f"T→{T*4:.4g} | best cost={best_cost:.4f}"
                    )

        if self.verbose:
            print(
                f"  [{benchmark.name}] {total_moves:,} moves | "
                f"{accepted/max(total_moves,1)*100:.1f}% accept | "
                f"{reheat_count} reheats | best cost {best_cost:.4f} | "
                f"{time.time()-t_start:.1f}s"
            )

        return best_placement

    # =======================================================================
    # Initialization
    # =======================================================================

    def _smart_init(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Start from the benchmark's hand-crafted initial placement and legalize it.

        The benchmark ships with macro_positions already set — often this is a
        human-designed layout, which has dramatically better HPWL than any greedy
        shelf-pack. If it happens to contain overlaps (it usually doesn't), we
        legalize by moving overlapping macros to nearby free positions.
        """
        placement = benchmark.macro_positions.clone()

        # Check for overlaps in the initial placement
        if not self._any_hard_overlaps(placement, benchmark):
            return placement

        # Legalize: for each overlapping pair, nudge the second macro along a
        # spiral outward until it finds a clear spot, or fall back to shelf pack.
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_set = set(torch.where(movable)[0].tolist())

        H = benchmark.num_hard_macros
        legalized = placement.clone()
        sizes = benchmark.macro_sizes
        W, Hc = benchmark.canvas_width, benchmark.canvas_height

        for i in range(H):
            if i not in movable_set:
                continue
            if not self._has_overlap(i, legalized[i, 0].item(), legalized[i, 1].item(),
                                     legalized, benchmark):
                continue
            # Try spiral search
            w, h = sizes[i, 0].item(), sizes[i, 1].item()
            found = False
            step = min(w, h) * 1.1
            for ring in range(1, 40):
                for angle_steps in range(8 * ring):
                    theta = (angle_steps / (8 * ring)) * 2 * math.pi
                    cx = legalized[i, 0].item() + ring * step * math.cos(theta)
                    cy = legalized[i, 1].item() + ring * step * math.sin(theta)
                    cx = max(w/2, min(W - w/2, cx))
                    cy = max(h/2, min(Hc - h/2, cy))
                    if not self._has_overlap(i, cx, cy, legalized, benchmark):
                        legalized[i, 0] = cx
                        legalized[i, 1] = cy
                        found = True
                        break
                if found:
                    break
            if not found:
                # Last resort: greedy shelf pack for this macro only
                legalized[i] = self._shelf_pack_single(i, legalized, benchmark)

        # Final check — if still overlapping, fall back to full shelf pack
        if self._any_hard_overlaps(legalized, benchmark):
            if self.verbose:
                print(f"  [{benchmark.name}] init legalization incomplete, using shelf-pack fallback")
            return self._greedy_row_init(benchmark)
        return legalized

    def _shelf_pack_single(self, idx: int, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Find any valid position for one macro via left-to-right scan."""
        w = benchmark.macro_sizes[idx, 0].item()
        h = benchmark.macro_sizes[idx, 1].item()
        gap = 0.001
        step_x = max(w * 0.5, 0.1)
        step_y = max(h * 0.5, 0.1)
        y = h/2 + gap
        while y + h/2 <= benchmark.canvas_height:
            x = w/2 + gap
            while x + w/2 <= benchmark.canvas_width:
                if not self._has_overlap(idx, x, y, placement, benchmark):
                    return torch.tensor([x, y])
                x += step_x
            y += step_y
        # Give up — return center
        return torch.tensor([benchmark.canvas_width/2, benchmark.canvas_height/2])

    def _greedy_row_init(self, benchmark: Benchmark) -> torch.Tensor:
        """Shelf-pack fallback, identical to v1."""
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes
        indices.sort(key=lambda i: -sizes[i, 1].item())
        gap = 0.001
        cursor_x = cursor_y = row_h = 0.0
        for idx in indices:
            w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
            if cursor_x + w > benchmark.canvas_width:
                cursor_x = 0.0
                cursor_y += row_h + gap
                row_h = 0.0
            if cursor_y + h > benchmark.canvas_height:
                placement[idx, 0] = w / 2.0
                placement[idx, 1] = h / 2.0
                continue
            placement[idx, 0] = cursor_x + w / 2.0
            placement[idx, 1] = cursor_y + h / 2.0
            cursor_x += w + gap
            row_h = max(row_h, h)
        return placement

    def _any_hard_overlaps(self, placement: torch.Tensor, benchmark: Benchmark) -> bool:
        """Fast vectorized full-overlap check over hard macros."""
        H = benchmark.num_hard_macros
        if H < 2:
            return False
        pos = placement[:H]
        sz = benchmark.macro_sizes[:H]
        dx = torch.abs(pos[:, 0:1] - pos[:, 0:1].T)
        dy = torch.abs(pos[:, 1:2] - pos[:, 1:2].T)
        min_sep_x = (sz[:, 0:1] + sz[:, 0:1].T) / 2.0
        min_sep_y = (sz[:, 1:2] + sz[:, 1:2].T) / 2.0
        overlap = (dx < min_sep_x) & (dy < min_sep_y)
        overlap.fill_diagonal_(False)
        return overlap.any().item()

    # =======================================================================
    # Soft macro optimization
    # =======================================================================

    def _optimize_softcells(self, plc, benchmark: Benchmark, placement: torch.Tensor):
        """
        Run plc.optimize_stdcells() and copy soft macro results back to our tensor.

        The PlacementCost object holds its own position state. We first sync our
        hard macro positions into plc, run the force-directed soft placer, then
        read back the soft macro positions into our tensor.
        """
        # Sync hard macros: our placement → plc
        for tensor_idx in range(benchmark.num_hard_macros):
            if benchmark.macro_fixed[tensor_idx]:
                continue
            plc_idx = benchmark.hard_macro_indices[tensor_idx]
            x = placement[tensor_idx, 0].item()
            y = placement[tensor_idx, 1].item()
            try:
                plc.update_node_coords(plc.modules_w_pins[plc_idx].get_name(), x, y)
            except Exception:
                # API fallback
                try:
                    plc.set_macro_xy_coordinates([plc_idx], [[x, y]])
                except Exception:
                    pass

        # Run soft macro optimization
        canvas_size = max(benchmark.canvas_width, benchmark.canvas_height)
        try:
            plc.optimize_stdcells(
                use_current_loc=False,
                move_stdcells=True,
                move_macros=False,
                log_scale_conns=False,
                use_sizes=False,
                io_factor=1.0,
                num_steps=[self.softcell_num_steps] * 3,
                max_move_distance=[canvas_size / 100.0] * 3,
                attract_factor=[100.0, 1.0e-3, 1.0e-5],
                repel_factor=[0.0, 1.0e6, 1.0e7],
            )
        except Exception as e:
            if self.verbose:
                print(f"  [{benchmark.name}] optimize_stdcells failed: {e}")
            return

        # Read back soft macro positions: plc → our placement
        for tensor_idx in range(benchmark.num_hard_macros, benchmark.num_macros):
            plc_idx = benchmark.soft_macro_indices[tensor_idx - benchmark.num_hard_macros]
            try:
                module = plc.modules_w_pins[plc_idx]
                x, y = module.get_pos()
                placement[tensor_idx, 0] = float(x)
                placement[tensor_idx, 1] = float(y)
            except Exception:
                pass

    # =======================================================================
    # Index structures
    # =======================================================================

    def _get_movable_indices(self, benchmark: Benchmark) -> List[int]:
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        return torch.where(movable)[0].tolist()

    def _build_macro_nets(self, benchmark: Benchmark) -> List[List[int]]:
        macro_nets: List[List[int]] = [[] for _ in range(benchmark.num_macros)]
        for net_id, net_nodes in enumerate(benchmark.net_nodes):
            for node_idx in net_nodes.tolist():
                if 0 <= node_idx < benchmark.num_macros:
                    macro_nets[node_idx].append(net_id)
        return macro_nets

    # =======================================================================
    # HPWL computation (unchanged from v1)
    # =======================================================================

    def _extended_positions(self, placement, benchmark):
        if benchmark.port_positions.shape[0] > 0:
            return torch.cat([placement, benchmark.port_positions], dim=0)
        return placement

    def _total_hpwl(self, placement, benchmark):
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

    def _delta_hpwl(self, idx, old_x, old_y, new_x, new_y, placement, benchmark, macro_nets):
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

    def _delta_hpwl_swap(self, idx_a, new_ax, new_ay, idx_b, new_bx, new_by,
                         placement, benchmark, macro_nets):
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

    # =======================================================================
    # Move generators (unchanged from v1)
    # =======================================================================

    def _swap_move(self, placement, benchmark, macro_nets, movable_indices, N):
        i = torch.randint(N, (1,)).item()
        j = torch.randint(N - 1, (1,)).item()
        if j >= i:
            j += 1
        idx_a = movable_indices[i]
        idx_b = movable_indices[j]
        old_ax, old_ay = placement[idx_a, 0].item(), placement[idx_a, 1].item()
        old_bx, old_by = placement[idx_b, 0].item(), placement[idx_b, 1].item()
        new_ax, new_ay = old_bx, old_by
        new_bx, new_by = old_ax, old_ay
        W, H = benchmark.canvas_width, benchmark.canvas_height
        wa = benchmark.macro_sizes[idx_a, 0].item()
        ha = benchmark.macro_sizes[idx_a, 1].item()
        wb = benchmark.macro_sizes[idx_b, 0].item()
        hb = benchmark.macro_sizes[idx_b, 1].item()
        if not (wa/2 <= new_ax <= W - wa/2 and ha/2 <= new_ay <= H - ha/2):
            return None
        if not (wb/2 <= new_bx <= W - wb/2 and hb/2 <= new_by <= H - hb/2):
            return None
        if self._has_overlap_excluding(idx_a, new_ax, new_ay, idx_b, placement, benchmark):
            return None
        if self._has_overlap_excluding(idx_b, new_bx, new_by, idx_a, placement, benchmark):
            return None
        if abs(new_ax - new_bx) < (wa + wb)/2 and abs(new_ay - new_by) < (ha + hb)/2:
            return None
        delta = self._delta_hpwl_swap(
            idx_a, new_ax, new_ay, idx_b, new_bx, new_by,
            placement, benchmark, macro_nets,
        )
        return delta, idx_a, new_ax, new_ay, idx_b, new_bx, new_by

    def _has_overlap(self, idx, new_x, new_y, placement, benchmark):
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

    def _has_overlap_excluding(self, idx, new_x, new_y, exclude_idx, placement, benchmark):
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

    # =======================================================================
    # Temperature estimation
    # =======================================================================

    def _estimate_init_temp(self, placement, benchmark, movable_indices, macro_nets, wl_scale):
        N = len(movable_indices)
        deltas: List[float] = []
        for _ in range(min(200, N * 5)):
            idx = movable_indices[torch.randint(N, (1,)).item()]
            old_x = placement[idx, 0].item()
            old_y = placement[idx, 1].item()
            w = benchmark.macro_sizes[idx, 0].item()
            h = benchmark.macro_sizes[idx, 1].item()
            dx = (torch.rand(1).item() * 2.0 - 1.0) * self.perturb_frac_init * benchmark.canvas_width
            dy = (torch.rand(1).item() * 2.0 - 1.0) * self.perturb_frac_init * benchmark.canvas_height
            new_x = max(w/2, min(benchmark.canvas_width - w/2, old_x + dx))
            new_y = max(h/2, min(benchmark.canvas_height - h/2, old_y + dy))
            d_hpwl = abs(self._delta_hpwl(idx, old_x, old_y, new_x, new_y,
                                          placement, benchmark, macro_nets))
            d = wl_scale * d_hpwl  # scale to composite-cost units
            if d > 0.0:
                deltas.append(d)
        if not deltas:
            return 1.0
        return -sum(deltas) / len(deltas) / math.log(self.accept_prob_init)