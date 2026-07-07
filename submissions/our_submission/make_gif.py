"""
make_gif.py — Generate a QSA visualisation GIF for ibm01 (or any benchmark).

Usage:
    uv run python submissions/our_submission/make_gif.py
    uv run python submissions/our_submission/make_gif.py --benchmark ibm01 --frames 120 --fps 15

Outputs: assets/qsa_<benchmark>.gif
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default="ibm01")
parser.add_argument("--frames", type=int, default=100, help="Number of frames to capture")
parser.add_argument("--fps", type=int, default=12)
parser.add_argument("--time-limit", type=float, default=60.0, help="SA time budget in seconds")
args = parser.parse_args()

BENCH = args.benchmark
N_FRAMES = args.frames
FPS = args.fps
TIME_LIMIT = args.time_limit

# ---------------------------------------------------------------------------
# Load benchmark
# ---------------------------------------------------------------------------

from macro_place.loader import load_benchmark_from_dir

ibm_root = Path("external/MacroPlacement/Testcases/ICCAD04") / BENCH
_, plc = load_benchmark_from_dir(str(ibm_root))

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir as _lbfd

# Re-load just the Benchmark object
import importlib, sys
loader = importlib.import_module("macro_place.loader")
benchmark, plc = loader.load_benchmark_from_dir(str(ibm_root))

# ---------------------------------------------------------------------------
# Import QSA internals
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from qsa_contest import QSAPlacer, QSAConfig

cfg = QSAConfig(
    time_limit=TIME_LIMIT,
    verbose=False,
    seed=42,
)
placer = QSAPlacer(cfg)

# ---------------------------------------------------------------------------
# Monkey-patch the SA loop to collect frames
# ---------------------------------------------------------------------------

frames: list[dict] = []          # list of {placement, last_moved, step, T, cost, best}
capture_interval = max(1, int(TIME_LIMIT * 1000 / N_FRAMES))  # approx every N ms worth of moves

_orig_place = QSAPlacer.place


def _instrumented_place(self, bench, plc_obj=None):
    """Thin wrapper that snapshots placement every capture_interval moves."""
    self._run_started_at = time.time()
    self._softcell_runtime_estimate = None

    torch.manual_seed(self.cfg.seed)
    t_start = self._run_started_at
    cfg = self.cfg

    if plc_obj is None and hasattr(bench, "_plc"):
        plc_obj = bench._plc

    placement = self._smart_init(bench)
    movable_indices = self._get_movable_indices(bench)
    N = len(movable_indices)

    macro_nets = self._build_macro_nets(bench)
    degrees = torch.tensor(
        [len(macro_nets[i]) + 1.0 for i in movable_indices], dtype=torch.float32
    )
    self._build_net_tensor(bench)
    self._precompute_overlap_cache(bench)
    self._init_ext_buffer(placement, bench)

    current_hpwl = self._total_hpwl_fast(placement, bench)
    wl_scale = 1.0 / max(current_hpwl, 1.0)
    current_cost = wl_scale * current_hpwl
    best_cost = current_cost
    best_placement = placement.clone()
    best_hpwl = current_hpwl

    T_start = self._estimate_init_temp(placement, bench, movable_indices, macro_nets, wl_scale)
    log_alpha = math.log(cfg.cooling_alpha)
    reheat_mult = 1.0

    accepted = rejected = total_moves = 0
    reheats = 0
    accept_window = deque(maxlen=cfg.sb_trigger_window)
    last_moved_idx = -1

    next_capture = 0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= cfg.time_limit:
            break

        progress = elapsed / cfg.time_limit
        T = max(T_start * math.exp(log_alpha * progress) * reheat_mult, 1e-12)
        perturb_frac = (
            cfg.perturb_frac_init * (1 - progress)
            + cfg.perturb_frac_final * progress
        )

        roll = torch.rand(1).item()
        move_accepted = False
        moved_idx = -1

        if roll < cfg.swap_prob and N >= 2:
            res = self._swap_move(placement, bench, macro_nets, movable_indices, N)
            if res is None:
                rejected += 1; total_moves += 1
                accept_window.append(False); continue
            delta_hpwl, ia, nax, nay, ib, nbx, nby = res
            delta_cost = wl_scale * delta_hpwl
            if delta_cost < 0 or torch.rand(1).item() < math.exp(-delta_cost / T):
                placement[ia, 0] = nax; placement[ia, 1] = nay
                placement[ib, 0] = nbx; placement[ib, 1] = nby
                current_hpwl += delta_hpwl; current_cost += delta_cost
                move_accepted = True; moved_idx = ia
        else:
            pick = torch.multinomial(degrees, 1).item()
            idx = movable_indices[pick]
            old_x, old_y = placement[idx, 0].item(), placement[idx, 1].item()
            w, h = bench.macro_sizes[idx, 0].item(), bench.macro_sizes[idx, 1].item()
            dx = (torch.rand(1).item()*2 - 1) * perturb_frac * bench.canvas_width
            dy = (torch.rand(1).item()*2 - 1) * perturb_frac * bench.canvas_height
            new_x = max(w/2, min(bench.canvas_width  - w/2, old_x + dx))
            new_y = max(h/2, min(bench.canvas_height - h/2, old_y + dy))
            if self._has_overlap(idx, new_x, new_y, placement, bench):
                rejected += 1; total_moves += 1
                accept_window.append(False); continue
            delta_hpwl = self._delta_hpwl(idx, old_x, old_y, new_x, new_y, placement, bench, macro_nets)
            delta_cost = wl_scale * delta_hpwl
            if delta_cost < 0 or torch.rand(1).item() < math.exp(-delta_cost / T):
                placement[idx, 0] = new_x; placement[idx, 1] = new_y
                current_hpwl += delta_hpwl; current_cost += delta_cost
                move_accepted = True; moved_idx = idx

        if move_accepted:
            accepted += 1
            accept_window.append(True)
            self._sync_ext_buffer(placement)
            last_moved_idx = moved_idx
            if current_cost < best_cost:
                best_cost = current_cost
                best_hpwl = current_hpwl
                best_placement = placement.clone()
        else:
            rejected += 1
            accept_window.append(False)
        total_moves += 1

        # Reheat
        stuck = (
            len(accept_window) == accept_window.maxlen
            and sum(accept_window) / accept_window.maxlen < cfg.sb_trigger_threshold
        )
        if stuck and reheats < cfg.max_reheats:
            reheat_mult *= 4.0; reheats += 1
            placement = best_placement.clone()
            current_hpwl = best_hpwl; current_cost = best_cost
            accept_window.clear()

        # Capture frame
        if total_moves >= next_capture:
            frames.append({
                "placement": placement.clone(),
                "last_moved": last_moved_idx,
                "step": total_moves,
                "T": T,
                "cost": current_cost,
                "best": best_cost,
                "elapsed": elapsed,
            })
            next_capture += capture_interval
            pct = len(frames) / N_FRAMES * 100
            print(f"\r  Capturing frames: {len(frames)}/{N_FRAMES} ({pct:.0f}%) "
                  f"t={elapsed:.1f}s cost={current_cost:.4f}", end="", flush=True)

    print()
    return best_placement


QSAPlacer.place = _instrumented_place

print(f"Running QSA on {BENCH} for {TIME_LIMIT:.0f}s to capture {N_FRAMES} frames...")
result = placer.place(benchmark)
print(f"SA done. Captured {len(frames)} frames. Rendering GIF...")

# ---------------------------------------------------------------------------
# Render frames → GIF
# ---------------------------------------------------------------------------

try:
    from PIL import Image
except ImportError:
    print("PIL not found. Install with: uv pip install pillow")
    raise

W = benchmark.canvas_width
H = benchmark.canvas_height
fixed_mask = benchmark.macro_fixed
Hn = benchmark.num_hard_macros


def _rgba(hex_str, alpha):
    """Hex string -> (r, g, b, alpha) tuple in 0..1 for matplotlib."""
    hex_str = hex_str.lstrip("#")
    r, g, b = (int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


# Base full-strength colours + face alpha. These COMPOSITE (alpha-blend over
# white) to the reference displayed colours, so overlaps darken naturally:
#   blue  base #1F77B4 @ a=0.55  ->  single #84B4D5, double-overlap #4C92C2
#   orange base #FF7F0E @ a=0.72 ->  "just moved" pops orange
#   pink  fixed             @ a=0.85 ->  soft pink #EDAFB0
BLUE_BASE,   BLUE_A   = "#1F77B4", 0.55   # movable
ORANGE_BASE, ORANGE_A = "#FF7F0E", 0.72   # just moved
PINK_BASE,   PINK_A   = "#E08A8D", 0.85   # fixed

# Edges: full opacity, a couple shades darker than the fill so boxes pop.
BLUE_EDGE   = "#1A5F8F"
ORANGE_EDGE = "#E06A0A"
PINK_EDGE   = "#C0392B"

# Displayed single-box colours (for the legend swatches).
BLUE_DISP   = "#84B4D5"
ORANGE_DISP = "#FF983E"
PINK_DISP   = "#EDAFB0"

pil_frames: list[Image.Image] = []

for fi, frame in enumerate(frames):
    pl = frame["placement"]
    last_moved = frame["last_moved"]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xlabel("X (μm)", fontsize=9)
    ax.set_ylabel("Y (μm)", fontsize=9)
    ax.set_title(f"{BENCH} — QSA Placement", fontsize=11, fontweight="bold")

    # Draw movable boxes first, then just-moved/fixed on top so highlights pop.
    moved_or_fixed = []
    for i in range(Hn):
        x, y = pl[i, 0].item(), pl[i, 1].item()
        w, h = benchmark.macro_sizes[i, 0].item(), benchmark.macro_sizes[i, 1].item()
        is_fixed = fixed_mask[i].item()
        is_moved = (i == last_moved)

        if is_fixed:
            face, edge, lw, z = _rgba(PINK_BASE, PINK_A), PINK_EDGE, 0.8, 3
        elif is_moved:
            face, edge, lw, z = _rgba(ORANGE_BASE, ORANGE_A), ORANGE_EDGE, 1.0, 4
        else:
            face, edge, lw, z = _rgba(BLUE_BASE, BLUE_A), BLUE_EDGE, 0.5, 2

        # facecolor carries its own alpha (composites for overlaps); edge stays
        # fully opaque so borders read crisply. Do NOT pass alpha= kwarg.
        rect = mpatches.Rectangle(
            (x - w/2, y - h/2), w, h,
            facecolor=face, edgecolor=edge,
            linewidth=lw, zorder=z,
        )
        if is_fixed or is_moved:
            moved_or_fixed.append(rect)
        else:
            ax.add_patch(rect)

    for rect in moved_or_fixed:
        ax.add_patch(rect)

    # Stats box
    stats = (
        f"Step: {frame['step']}\n"
        f"Temp: {frame['T']:.2e}\n"
        f"Cost: {frame['cost']:.4f}\n"
        f"Best: {frame['best']:.4f}"
    )
    ax.text(
        0.02, 0.98, stats,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
        fontfamily="monospace",
    )

    # Legend — solid swatches in the displayed single-box colours.
    legend_elements = [
        mpatches.Patch(facecolor=BLUE_DISP,   edgecolor=BLUE_EDGE,   label="Movable"),
        mpatches.Patch(facecolor=ORANGE_DISP, edgecolor=ORANGE_EDGE, label="Just moved"),
        mpatches.Patch(facecolor=PINK_DISP,   edgecolor=PINK_EDGE,   label="Fixed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.9)

    fig.tight_layout()

    # Render to PIL
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    import numpy as np
    img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    pil_frames.append(Image.fromarray(img_arr[..., :3]))
    plt.close(fig)

    if (fi + 1) % 10 == 0:
        print(f"  Rendered {fi+1}/{len(frames)} frames")

# ---------------------------------------------------------------------------
# Save GIF
# ---------------------------------------------------------------------------

out_path = Path("assets") / f"qsa_{BENCH}.gif"
out_path.parent.mkdir(exist_ok=True)

duration_ms = int(1000 / FPS)
pil_frames[0].save(
    out_path,
    save_all=True,
    append_images=pil_frames[1:],
    duration=duration_ms,
    loop=0,
    optimize=False,
)
print(f"\nSaved → {out_path}  ({len(pil_frames)} frames @ {FPS}fps)")
