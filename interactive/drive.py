"""Drive-it-yourself: a real-time, top-down drift sim you control, with the live advisor
HUD overlaid.

    python -m interactive.drive                 # play it (keyboard)
    python -m interactive.drive --record out.gif # headless: render the scripted "rescue"
                                                 # story to a GIF (for the README)

Controls
    LEFT / RIGHT ...... steer
    UP ................ throttle      DOWN ... brake
    SPACE ............. toggle AUTOPILOT (advisor drives) — feel the difference
    R ................. reset to the sweet spot
    ESC / Q ........... quit

The car starts in a steady left-hand drift.  Hold throttle and countersteer to keep it;
the HUD shows the advisor's steering + pedal TARGETS and a stability margin.  The score
counts time spent in the sweet spot.
"""
from __future__ import annotations

import argparse
import math
import os

from config.params import (
    DEFAULT_CONTROLLER,
    DEFAULT_LEARNING,
    DEFAULT_MONITOR,
    DEFAULT_RATES,
    DEFAULT_VEHICLE,
)
from control.equilibria import solve_drift_equilibrium
from realtime.loop import Advisor
from sim.sensors import SensorBus
from sim.vehicle_model import rk4_step, sideslip, speed

MU = 0.95
V0, BETA0 = 12.0, math.radians(-30.0)

# colors
BG = (11, 15, 20); FG = (220, 226, 232); DIM = (120, 130, 140)
CYAN = (0, 200, 220); GOLD = (240, 200, 40); RED = (230, 70, 70)
GREEN = (70, 200, 110); GREY = (50, 64, 78); TRACK = (28, 36, 44)
SEV = {"ok": GREEN, "watch": GOLD, "act": RED, "—": DIM}


class DriveSim:
    """Plant + advisor + driver inputs + score (rendering-agnostic)."""

    def __init__(self):
        self.p = DEFAULT_VEHICLE
        self.eq0 = solve_drift_equilibrium(V0, BETA0, self.p, MU, MU)
        self.advisor = Advisor(self.p, DEFAULT_CONTROLLER, DEFAULT_MONITOR,
                               DEFAULT_LEARNING, DEFAULT_RATES)
        self.bus = SensorBus()
        self.reset()

    def reset(self):
        e = self.eq0
        self.x = [e.vx, e.vy, e.r, 0.0, 0.0, 0.0]
        self.delta, self.Fxr = e.delta, e.Fxr
        self.autopilot = False
        self.score = 0.0
        self.t = 0.0
        self.trail = []
        self.spun = False

    def step(self, dt, steer_cmd, throttle_cmd):
        """steer_cmd in [-1,1] (rate), throttle_cmd in [-1,1] (target gas/brake)."""
        p, e = self.p, self.eq0
        # driver inputs evolve from commands (skip if autopilot)
        if not self.autopilot:
            self.delta += steer_cmd * math.radians(110.0) * dt
            self.delta = max(-DEFAULT_CONTROLLER.delta_max,
                             min(DEFAULT_CONTROLLER.delta_max, self.delta))
            tgt_F = (throttle_cmd * p.Fx_motor_max if throttle_cmd >= 0
                     else throttle_cmd * p.Fx_brake_max)
            self.Fxr += (tgt_F - self.Fxr) * min(1.0, 6.0 * dt)

        s = self.bus.read(self.x, self.delta, self.Fxr, MU, MU, p)
        tel = self.advisor.update(s, dt)

        if self.autopilot and tel.advice and tel.advice.feasible:
            self.delta, self.Fxr = tel.advice.delta_target, tel.advice.Fxr_target

        self.x = rk4_step(self.x, self.delta, self.Fxr, p, MU, MU, dt, Fxf=e.Fxf)
        self.t += dt
        b = sideslip(self.x[0], self.x[1])
        if abs(b - e.beta) < math.radians(10):
            self.score += dt
        self.trail.append((self.x[3], self.x[4]))
        if len(self.trail) > 600:
            self.trail.pop(0)
        if speed(self.x[0], self.x[1]) < 1.0 or abs(math.degrees(b)) > 80:
            self.spun = True
        return tel


# --------------------------------------------------------------------------- render
def _w2s(X, Y, cam, scale, w, h):
    return int(w / 2 + (X - cam[0]) * scale), int(h / 2 - (Y - cam[1]) * scale)


def draw(screen, pygame, sim: DriveSim, tel, font, bigfont):
    w, h = screen.get_size()
    screen.fill(BG)
    x = sim.x
    cam = (x[3], x[4]); scale = 7.0

    # path trail
    if len(sim.trail) > 1:
        pts = [_w2s(px, py, cam, scale, w, h) for px, py in sim.trail]
        pygame.draw.lines(screen, TRACK, False, pts, 6)

    # car body (oriented by yaw psi), velocity vector (psi+beta)
    psi = x[5]
    cx, cy = _w2s(x[3], x[4], cam, scale, w, h)
    L, W = 4.6 * scale, 2.0 * scale
    corners = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
    rot = []
    for px, py in corners:
        rx = px * math.cos(psi) - py * math.sin(psi)
        ry = px * math.sin(psi) + py * math.cos(psi)
        rot.append((cx + rx, cy - ry))
    rear_sat = tel.monitor and tel.monitor.U_r > 0.92
    pygame.draw.polygon(screen, RED if rear_sat else CYAN, rot, 0)
    pygame.draw.polygon(screen, FG, rot, 2)
    # velocity arrow
    b = sideslip(x[0], x[1]); V = speed(x[0], x[1])
    vang = psi + b
    vx2 = cx + math.cos(vang) * V * 1.4
    vy2 = cy - math.sin(vang) * V * 1.4
    pygame.draw.line(screen, GOLD, (cx, cy), (vx2, vy2), 3)

    # ---- HUD ----
    _hud(screen, pygame, sim, tel, font, bigfont, w, h)


def _bar(screen, pygame, x, y, wpx, hpx, frac, color, center=False):
    pygame.draw.rect(screen, GREY, (x, y, wpx, hpx), 1)
    if center:
        mid = x + wpx // 2
        ln = int((wpx // 2) * max(-1, min(1, frac)))
        pygame.draw.rect(screen, color, (mid if ln >= 0 else mid + ln, y + 1, abs(ln), hpx - 2))
    else:
        pygame.draw.rect(screen, color, (x, y, int(wpx * max(0, min(1, frac))), hpx))


def _hud(screen, pygame, sim, tel, font, bigfont, w, h):
    adv, mon = tel.advice, tel.monitor
    b = math.degrees(sideslip(sim.x[0], sim.x[1])); V = speed(sim.x[0], sim.x[1])
    panel = pygame.Surface((300, h)); panel.set_alpha(220); panel.fill((6, 9, 12))
    screen.blit(panel, (0, 0))

    def txt(s, y, c=FG, f=None):
        screen.blit((f or font).render(s, True, c), (16, y))

    txt("DRIFT ADVISOR", 14, FG, bigfont)
    sev = mon.severity if mon else "—"
    lab = mon.label if mon else "—"
    txt(f"{lab.upper()}  ({sev})", 50, SEV.get(sev, DIM), bigfont)

    # steering: current vs target
    txt("STEERING", 92, DIM)
    dmax = DEFAULT_CONTROLLER.delta_max
    _bar(screen, pygame, 16, 112, 260, 16, sim.delta / dmax, CYAN, center=True)
    if adv and adv.feasible:
        tgt = max(-1, min(1, adv.delta_target / dmax))
        mx = 16 + 130 + int(130 * tgt)
        pygame.draw.line(screen, GOLD, (mx, 108), (mx, 132), 3)
    txt(adv.steer_text if adv else "—", 134, GOLD)

    # throttle/brake: current vs target
    txt("ACCELERATOR / BRAKE", 168, DIM)
    g = adv.gas_current if adv else 0.0
    _bar(screen, pygame, 16, 188, 260, 16, g, GREEN if g >= 0 else RED, center=True)
    if adv and adv.feasible:
        gt = max(-1, min(1, adv.gas_target))
        mx = 16 + 130 + int(130 * gt)
        pygame.draw.line(screen, GOLD, (mx, 184), (mx, 208), 3)
    txt(adv.pedal_text if adv else "—", 210, GOLD)

    # margin
    txt("STABILITY MARGIN", 244, DIM)
    m = mon.margin if mon else 0.0
    _bar(screen, pygame, 16, 264, 260, 16, m, SEV.get(sev, DIM))
    txt(f"time-to-loss {mon.tau:.1f}s" if mon else "—", 286, FG)

    txt(f"beta {b:5.0f} deg   (target {math.degrees(BETA0):.0f})", 322, FG)
    txt(f"V    {V:5.1f} m/s", 344, FG)
    txt(f"score {sim.score:5.1f} s in sweet spot", 372, GOLD, bigfont)
    txt("AUTOPILOT ON" if sim.autopilot else "you are driving",
        404, GREEN if sim.autopilot else DIM)
    if sim.spun:
        txt("SPUN OUT — press R", 432, RED, bigfont)
    txt("arrows steer/throttle  SPACE autopilot  R reset", h - 28, DIM)


# --------------------------------------------------------------------------- modes
def _bootstrap_display(pygame):
    """Create the window + fonts (fast).  Shared by the desktop and browser loops."""
    screen = pygame.display.set_mode((1000, 680))
    pygame.display.set_caption("Drift Sweet-Spot Advisor — drive it")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    bigfont = pygame.font.SysFont("consolas", 20, bold=True)
    return screen, clock, font, bigfont


def _frame(pygame, screen, sim, dt, font, bigfont):
    """Process one frame (events, input, sim steps, render).  Returns running:bool."""
    running = True
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif ev.key == pygame.K_SPACE:
                sim.autopilot = not sim.autopilot
            elif ev.key == pygame.K_r:
                sim.reset()
    keys = pygame.key.get_pressed()
    steer = (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT])        # +left per ISO
    throttle = 1.0 if keys[pygame.K_UP] else (-1.0 if keys[pygame.K_DOWN] else 0.0)
    tel = None
    for _ in range(2):                                          # ~2 sim steps / frame
        tel = sim.step(dt, steer, throttle)
    draw(screen, pygame, sim, tel, font, bigfont)
    pygame.display.flip()
    return running


def play():
    """Desktop loop (synchronous)."""
    import pygame
    pygame.init()
    screen, clock, font, bigfont = _bootstrap_display(pygame)
    sim = DriveSim()
    dt = DEFAULT_RATES.dt_sim
    running = True
    while running:
        running = _frame(pygame, screen, sim, dt, font, bigfont)
        clock.tick(50)
    pygame.quit()


async def _present(pygame, screen, frames=3):
    """Flip + yield a few times so pygbag actually presents the current surface."""
    import asyncio
    for _ in range(frames):
        pygame.display.flip()
        await asyncio.sleep(0)


async def play_async():
    """Browser loop (pygbag/WASM): identical frame, but yields to the event loop.

    Wrapped so a startup error is painted onto the canvas (the browser console does not
    always surface Python tracebacks from the WASM runtime) instead of leaving a blank
    grey canvas.
    """
    import asyncio
    import traceback

    import pygame
    screen = None
    try:
        pygame.init()
        screen, clock, font, bigfont = _bootstrap_display(pygame)
        # present the splash over several frames BEFORE the one-time equilibrium solve in
        # DriveSim(): one flip may not present, and the solve briefly blocks the thread.
        screen.fill(BG)
        screen.blit(bigfont.render("Loading drift advisor...", True, FG), (40, 40))
        await _present(pygame, screen)
        sim = DriveSim()
        sim.autopilot = True        # start with the advisor holding the drift (the page can
        #                             sit through a multi-second load; a hands-off car would
        #                             coast out of the drift before the visitor interacts).
        #                             SPACE hands control to the visitor.
        dt = DEFAULT_RATES.dt_sim
        running = True
        while running:
            running = _frame(pygame, screen, sim, dt, font, bigfont)
            await asyncio.sleep(0)                              # hand control to the browser
            clock.tick(50)
        pygame.quit()
    except Exception:
        tb = traceback.format_exc()
        try:
            if screen is None:
                screen = pygame.display.set_mode((1000, 680))
            ef = pygame.font.SysFont("consolas", 13)
            screen.fill((12, 12, 16))
            screen.blit(ef.render("drive demo error:", True, (255, 110, 110)), (12, 10))
            for i, line in enumerate(tb.splitlines()[-34:]):
                screen.blit(ef.render(line[:160], True, (230, 180, 180)), (12, 32 + 15 * i))
            await _present(pygame, screen, 5)
        except Exception:
            pass
        raise


def record(path: str, seconds: float = 7.0):
    """Headless: render the scripted 'rescue' story (drift -> throttle mistake ->
    advisor cue -> recover) to a GIF using the SDL dummy driver + PIL."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from PIL import Image
    pygame.init()
    screen = pygame.Surface((1000, 680))
    font = pygame.font.SysFont("consolas", 16)
    bigfont = pygame.font.SysFont("consolas", 20, bold=True)
    sim = DriveSim()
    dt = DEFAULT_RATES.dt_sim
    frames = []
    n = int(seconds / dt)
    for k in range(n):
        t = k * dt
        # scripted driver: hold throttle + slight countersteer; brief over-throttle mistake
        if 1.0 <= t < 1.3:
            sim.delta, sim.Fxr = sim.eq0.delta, sim.eq0.Fxr + 3000      # the mistake
            tel = sim.bus.read(sim.x, sim.delta, sim.Fxr, MU, MU, sim.p)
            tel = sim.advisor.update(tel, dt)
            sim.x = rk4_step(sim.x, sim.delta, sim.Fxr, sim.p, MU, MU, dt, Fxf=sim.eq0.Fxf)
            sim.t += dt; sim.trail.append((sim.x[3], sim.x[4]))
        else:
            sim.autopilot = True                                        # then obey advisor
            tel = sim.step(dt, 0.0, 0.0)
        if k % 3 == 0:                                                  # ~33 fps
            draw(screen, pygame, sim, tel, font, bigfont)
            arr = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
            frames.append(Image.fromarray(arr))
    pygame.quit()
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=30, loop=0)
    print(f"saved {path} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", metavar="PATH", default=None,
                    help="headless: render the scripted rescue story to a GIF")
    ap.add_argument("--seconds", type=float, default=7.0)
    args = ap.parse_args()
    if args.record:
        record(args.record, args.seconds)
    else:
        play()


if __name__ == "__main__":
    main()
