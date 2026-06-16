"""Drift sweet-spot advisor -- entry point.

Examples
--------
  python main.py                       # run the scenario, print summary, show comparison plot
  python main.py --save out.png        # save the comparison figure (no window needed)
  python main.py --animate rescue      # live HUD for the 'rescue' run
  python main.py --animate ignore --save hud.mp4   # save the HUD animation (needs ffmpeg)
  python main.py --noise               # add sensor noise
"""
from __future__ import annotations

import argparse

from scenarios.too_much_throttle import simulate, summarize


def main():
    ap = argparse.ArgumentParser(description="Drift sweet-spot advisor demo")
    ap.add_argument("--animate", choices=["ignore", "rescue", "assist"], default=None,
                    help="show the live HUD for one driver behaviour")
    ap.add_argument("--save", default=None, help="save figure/animation to this path")
    ap.add_argument("--noise", action="store_true", help="add sensor noise")
    args = ap.parse_args()

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
