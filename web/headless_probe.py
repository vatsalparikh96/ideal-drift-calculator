"""Headless reproduction of the WASM demo to capture what the browser actually does.

Drives the live (or a given) pygbag build in headless Edge, logging console output,
page errors, the embedded xterm/vtx terminal text (where pygbag prints Python
tracebacks), a center-pixel sample of each canvas, and a screenshot.  Not shipped — a
local debugging tool.

    python web/headless_probe.py [URL] [device_scale_factor] [wait_seconds]
"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://vatsalparikh96.github.io/ideal-drift-calculator/"
DSF = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
WAIT = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0


def run():
    logs: list[str] = []
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 860}, device_scale_factor=DSF)
        page = ctx.new_page()
        page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: errors.append(str(e)))
        print(f"== loading {URL}  dsf={DSF}  wait={WAIT}s ==")
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(int(WAIT * 1000))

        # canvas state
        canvases = page.evaluate(
            """() => [...document.querySelectorAll('canvas')].map(c => {
                let center = null;
                try {
                    const ctx = c.getContext('2d');
                    if (ctx) { const d = ctx.getImageData((c.width/2)|0,(c.height/2)|0,1,1).data;
                               center = [d[0],d[1],d[2],d[3]]; }
                    else center = 'no-2d-ctx';
                } catch(e) { center = 'err:'+e.message; }
                return {id:c.id, w:c.width, h:c.height,
                        cssW:c.style.width, cssH:c.style.height, center};
            })"""
        )
        # embedded terminal text (pygbag prints python output/tracebacks here)
        term = page.evaluate(
            """() => {
                const el = document.querySelector('.xterm-rows') ||
                           document.querySelector('#xtermjs') || document.body;
                return (el && el.innerText ? el.innerText : '').slice(-4000);
            }"""
        )
        page.screenshot(path="web/_probe.png", full_page=False)
        browser.close()

    print(f"\n==== {len(logs)} console messages (last 60) ====")
    for line in logs[-60:]:
        print(line[:200])
    print(f"\n==== {len(errors)} page errors ====")
    for e in errors:
        print(e[:400])
    print("\n==== canvases ====")
    for c in canvases:
        print(c)
    print("\n==== embedded terminal (tail) ====")
    print(term[-2500:])
    print("\n==== screenshot -> web/_probe.png ====")


if __name__ == "__main__":
    run()
