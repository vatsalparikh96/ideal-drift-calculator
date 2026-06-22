"""Compile the drive-it-yourself demo to WebAssembly with pygbag.

    pip install pygbag
    python web/build.py            # -> web/_app/build/web/  (open index.html via a server)

Assembles a *self-contained* app dir (only the runtime packages, so the WASM bundle
excludes media/, tests/, docs/, experiments/ and stays small), then runs pygbag on it.
CI (.github/workflows/pages.yml) runs this and deploys web/_app/build/web to GitHub Pages.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "web" / "_app"
# the minimal import closure of interactive.drive (all NumPy-only at runtime)
PACKAGES = ["config", "control", "sim", "realtime", "intent", "estimation",
            "learning", "interactive"]


def assemble() -> Path:
    if APP.exists():
        shutil.rmtree(APP)
    APP.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for pkg in PACKAGES:
        shutil.copytree(ROOT / pkg, APP / pkg, ignore=ignore)
    shutil.copy2(ROOT / "web" / "main.py", APP / "main.py")
    return APP


def _enable_autostart(out: Path) -> None:
    """Patch the generated loader so the demo runs without a click.

    pygbag hardcodes ``autorun:0`` + ``ume_block:1`` (wait for a user gesture to unlock
    audio before starting).  This app has no audio, so we start immediately instead of
    showing a blank canvas until the visitor happens to click.
    """
    idx = out / "index.html"
    html = idx.read_text(encoding="utf-8")
    patched = (html.replace("autorun : 0,", "autorun : 1,")
                   .replace("ume_block : 1,", "ume_block : 0,"))
    if patched != html:
        idx.write_text(patched, encoding="utf-8")
        print("patched index.html: autorun=1, ume_block=0 (auto-start, no click)")
    else:
        print("WARN: autorun/ume_block patterns not found in index.html (pygbag template changed?)")


def build() -> int:
    app = assemble()
    cmd = [sys.executable, "-m", "pygbag", "--build", str(app / "main.py")]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    out = app / "build" / "web"
    if result.returncode == 0 and out.exists():
        _enable_autostart(out)
        print(f"OK -> {out}")
    else:
        print(f"build failed (exit {result.returncode}); expected output: {out}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(build())
