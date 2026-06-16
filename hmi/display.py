"""HMI / visualization for the drift sweet-spot advisor.

Two views:
  * plot_comparison(...) -- static multi-panel comparison of driver behaviours
    (sideslip vs time, path, and the advice timeline).
  * animate_hud(...)     -- a live HUD echoing the dashboard concept: a steering
    target marker vs the current wheel, a throttle/brake target bar vs current, the
    text cues, a stability-margin bar, and a beta-r phase-portrait dot with the target
    drift equilibrium and the driven path.

The steering cue is shown as a TARGET the driver should move the wheel toward (it
retracts as the car returns to the sweet spot), never a non-retracting "turn more"
arrow -- this is what avoids the lift+countersteer pendulum.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

_COLORS = {"ignore": "tab:red", "rescue": "tab:orange", "assist": "tab:green"}


def plot_comparison(histories: dict, save: str | None = None):
    """histories: {mode: hist dict}.  Sideslip-vs-time, path, and advice timeline."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Drift sweet-spot advisor — too-much-throttle scenario", fontsize=13)

    ax = axes[0]
    for mode, h in histories.items():
        ax.plot(h["t"], h["beta"], color=_COLORS.get(mode, "k"), label=mode, lw=2)
    bstar = next(iter(histories.values()))["beta_star"]
    tgt = next((b for b in bstar if not math.isnan(b)), -30.0)
    ax.axhline(tgt, ls="--", color="gray", lw=1, label="target β*")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time [s]"); ax.set_ylabel("sideslip β [deg]")
    ax.set_title("Sideslip: ignore spins, rescue/assist hold")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)

    ax = axes[1]
    for mode, h in histories.items():
        ax.plot(h["X"], h["Y"], color=_COLORS.get(mode, "k"), label=mode, lw=2)
        ax.plot(h["X"][0], h["Y"][0], "o", color=_COLORS.get(mode, "k"), ms=5)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Path (global)"); ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    h = histories.get("rescue") or next(iter(histories.values()))
    ax.plot(h["t"], h["delta"], color="tab:blue", lw=2, label="δ current")
    ax.plot(h["t"], h["delta_target"], color="tab:blue", ls="--", lw=1.5, label="δ target")
    ax2 = ax.twinx()
    ax2.plot(h["t"], np.array(h["gas"]) * 100, color="tab:purple", lw=2, label="gas current %")
    ax2.plot(h["t"], np.array(h["gas_target"]) * 100, color="tab:purple", ls="--", lw=1.5,
             label="gas target %")
    ax.set_xlabel("time [s]"); ax.set_ylabel("steering δ [deg]", color="tab:blue")
    ax2.set_ylabel("throttle [%]", color="tab:purple")
    ax.set_title("Advice timeline (rescue): δ and throttle vs targets")
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)
        print(f"saved {save}")
    return fig


def animate_hud(h: dict, save: str | None = None, fps: int = 30, decim: int = 3):
    """Animated HUD for one run.  decim subsamples frames for speed."""
    from matplotlib.animation import FuncAnimation

    idx = list(range(0, len(h["t"]), decim))
    fig = plt.figure(figsize=(13, 6.5), facecolor="#0b0f14")
    fig.suptitle("DRIFT SWEET-SPOT ADVISOR", color="w", fontsize=14)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])

    ax_steer = fig.add_subplot(gs[0, 0]); _dark(ax_steer, "STEERING")
    ax_ped = fig.add_subplot(gs[0, 1]); _dark(ax_ped, "ACCELERATOR")
    ax_txt = fig.add_subplot(gs[0, 2]); _dark(ax_txt, "ADVICE"); ax_txt.axis("off")
    ax_pp = fig.add_subplot(gs[1, 0]); _dark(ax_pp, "β–r phase (sweet spot ★)")
    ax_path = fig.add_subplot(gs[1, 1:]); _dark(ax_path, "PATH")

    # static phase-portrait target + path
    bstar = next((b for b in h["beta_star"] if not math.isnan(b)), -30.0)
    rstar = next((v for v in h["r_star"] if not math.isnan(v)), 0.74)
    ax_pp.plot([rstar], [bstar], "*", color="gold", ms=18, zorder=5)
    ax_pp.axhline(bstar, color="gold", ls=":", lw=0.8, alpha=0.5)
    ax_pp.set_xlabel("yaw rate r [rad/s]", color="w")
    ax_pp.set_ylabel("β [deg]", color="w")
    ax_pp.set_xlim(-1.5, 1.5); ax_pp.set_ylim(-85, 30)
    pp_trail, = ax_pp.plot([], [], color="tab:cyan", lw=1, alpha=0.6)
    pp_dot, = ax_pp.plot([], [], "o", color="tab:cyan", ms=9)

    ax_path.plot(h["X"], h["Y"], color="#33424f", lw=1, alpha=0.6)
    ax_path.axis("equal")
    path_dot, = ax_path.plot([], [], "o", color="tab:cyan", ms=8)
    path_trail, = ax_path.plot([], [], color="tab:cyan", lw=2)
    ax_path.set_xlabel("X [m]", color="w"); ax_path.set_ylabel("Y [m]", color="w")

    # steering dial
    ax_steer.set_xlim(-1.4, 1.4); ax_steer.set_ylim(-1.4, 1.4); ax_steer.set_aspect("equal")
    ax_steer.axis("off")
    th = np.linspace(math.radians(210), math.radians(-30), 100)
    ax_steer.plot(np.cos(th), np.sin(th), color="#33424f", lw=3)
    cur_needle, = ax_steer.plot([], [], color="tab:cyan", lw=4)
    tgt_needle, = ax_steer.plot([], [], color="gold", lw=3, ls="--")
    steer_lbl = ax_steer.text(0, -1.25, "", color="w", ha="center", fontsize=11)

    # pedal bars
    ax_ped.set_xlim(0, 2); ax_ped.set_ylim(-1.05, 1.05); ax_ped.axis("off")
    ax_ped.axhline(0, color="#33424f", lw=1)
    cur_bar = ax_ped.bar(0.55, 0, width=0.5, color="tab:cyan")[0]
    tgt_line, = ax_ped.plot([], [], color="gold", lw=3)
    ax_ped.text(0.55, 1.0, "now", color="tab:cyan", ha="center", fontsize=9)
    ax_ped.text(1.45, 1.0, "target", color="gold", ha="center", fontsize=9)
    ped_lbl = ax_ped.text(1.0, -1.0, "", color="w", ha="center", fontsize=11)

    txt_label = ax_txt.text(0.5, 0.78, "", color="w", ha="center", fontsize=15, weight="bold",
                            transform=ax_txt.transAxes)
    txt_steer = ax_txt.text(0.5, 0.55, "", color="gold", ha="center", fontsize=13,
                            transform=ax_txt.transAxes)
    txt_ped = ax_txt.text(0.5, 0.40, "", color="gold", ha="center", fontsize=13,
                          transform=ax_txt.transAxes)
    txt_margin = ax_txt.text(0.5, 0.18, "", color="w", ha="center", fontsize=11,
                             transform=ax_txt.transAxes)

    sev_color = {"ok": "tab:green", "watch": "gold", "act": "tab:red", "—": "w"}

    def needle(angle_deg):
        # wheel angle -> needle within the dial arc (clamp display to +/-40 deg -> arc)
        frac = np.clip(angle_deg / 40.0, -1, 1)
        a = math.radians(90 - frac * 120)   # map to 210..-30 arc
        return [0, math.cos(a)], [0, math.sin(a)]

    def init():
        return ()

    def frame(j):
        i = idx[j]
        # steering dial
        xs, ys = needle(h["delta"][i]); cur_needle.set_data(xs, ys)
        dt_ = h["delta_target"][i]
        if not math.isnan(dt_):
            xt, yt = needle(dt_); tgt_needle.set_data(xt, yt)
        steer_lbl.set_text(h["steer_text"][i])
        # pedal
        g = h["gas"][i]; cur_bar.set_y(min(0, g)); cur_bar.set_height(abs(g))
        cur_bar.set_color("tab:cyan" if g >= 0 else "tab:red")
        gt = h["gas_target"][i]
        if not math.isnan(gt):
            tgt_line.set_data([1.2, 1.7], [gt, gt])
        ped_lbl.set_text(h["pedal_text"][i])
        # advice text
        lab = h["label"][i]; sev = h["severity"][i]
        txt_label.set_text(f"{lab.upper()}  ({sev})")
        txt_label.set_color(sev_color.get(sev, "w"))
        txt_steer.set_text("→ " + h["steer_text"][i])
        txt_ped.set_text("→ " + h["pedal_text"][i])
        m = h["margin"][i]; tau = h["tau"][i]
        txt_margin.set_text(f"stability margin {m*100:3.0f}%   |   time-to-loss {tau:.1f}s\n"
                            f"t={h['t'][i]:.2f}s  β={h['beta'][i]:.0f}°  V={h['V'][i]:.0f} m/s")
        # phase + path trails
        lo = max(0, i - 80)
        pp_trail.set_data(h["r"][lo:i + 1], h["beta"][lo:i + 1])
        pp_dot.set_data([h["r"][i]], [h["beta"][i]])
        path_trail.set_data(h["X"][:i + 1], h["Y"][:i + 1])
        path_dot.set_data([h["X"][i]], [h["Y"][i]])
        return ()

    anim = FuncAnimation(fig, frame, init_func=init, frames=len(idx),
                         interval=1000 / fps, blit=False)
    fig.tight_layout()
    if save:
        anim.save(save, fps=fps, dpi=90)
        print(f"saved {save}")
    return anim, fig


def _dark(ax, title):
    ax.set_facecolor("#0b0f14")
    ax.set_title(title, color="w", fontsize=10)
    ax.tick_params(colors="#88909a")
    for sp in ax.spines.values():
        sp.set_color("#33424f")
