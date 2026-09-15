"""Drift sweet-spot advisor -- entry point.

Examples
--------
  python main.py                       # run the scenario, print summary, show comparison plot
  python main.py --save out.png        # save the comparison figure (no window needed)
  python main.py --animate rescue      # live HUD for the 'rescue' run
  python main.py --animate ignore --save hud.mp4   # save the HUD animation (needs ffmpeg)
  python main.py --noise               # add sensor noise
  python main.py --validate            # sanity-check a nominal drift equilibrium, no plot
"""
from __future__ import annotations

import argparse
import math
import sys

from scenarios.too_much_throttle import simulate, summarize


def _validate() -> bool:
    """Solve a nominal drift equilibrium with default params and report feasibility
    and controllability, so setup issues surface without running the full scenario."""
    from config.params import VehicleParams
    from control.equilibria import solve_drift_equilibrium
    from control.stability import controllability, linearize

    p = VehicleParams()
    V, beta, mu_f, mu_r = 15.0, math.radians(25.0), 1.0, 1.0
    eq = solve_drift_equilibrium(V, beta, p, mu_f, mu_r)
    print(f"equilibrium: feasible={eq.feasible} delta={math.degrees(eq.delta):.1f} deg "
          f"Fxr={eq.Fxr:.0f} N reason='{eq.reason}'")
    if not eq.feasible:
        return False
    A, B = linearize(eq, p, mu_f, mu_r)
    rank, cond = controllability(A, B[:, [0]])
    print(f"controllability: rank={rank}/3 cond={cond:.1f}")
    return rank == 3


def main():
    ap = argparse.ArgumentParser(description="Drift sweet-spot advisor demo")
    ap.add_argument("--animate", choices=["ignore", "rescue", "assist"], default=None,
                    help="show the live HUD for one driver behaviour")
    ap.add_argument("--save", default=None, help="save figure/animation to this path")
    ap.add_argument("--noise", action="store_true", help="add sensor noise")
    ap.add_argument("--validate", action="store_true",
                    help="sanity-check a nominal drift equilibrium and exit (no plot)")
    args = ap.parse_args()

    if args.validate:
        sys.exit(0 if _validate() else 1)

    import matplotlib
    if args.save and not args.animate:
        matplotlib.use("Agg")    # headless save for the static figure
    import matplotlib.pyplot as plt

    from hmi.display import animate_hud, plot_comparison

    if args.animate:
        h = simulate(args.animate, noise=args.noise)
        print(summarize(args.animate, noise=args.noise))
        _anim, _fig = animate_hud(h, save=args.save)
        if not args.save:
            plt.show()
        return

    print("Too-much-throttle scenario:\n")
    histories = {}
    for mode in ("ignore", "rescue", "assist"):
        print("  " + summarize(mode, noise=args.noise))
        histories[mode] = simulate(mode, noise=args.noise)
    plot_comparison(histories, save=args.save)
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
