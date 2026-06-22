"""Browser (pygbag / WebAssembly) entry point for the drive-it-yourself demo.

pygbag bundles the app directory and runs this file's ``asyncio.run`` in the page.
The advisor runs the identical NumPy-only control law as the desktop build (SciPy is
replaced by control._numerics in the WASM sandbox).

Build locally with:  python web/build.py
"""
# /// script
# dependencies = ["numpy"]
# ///
# (pygbag reads this PEP 723 block at startup and pre-loads NumPy from its CDN.)
import asyncio

# pygame MUST be imported at the entry top level so pygbag wires up the real pygame-ce.
# A lazy import inside the loop yields a stub without pygame.init -> blank canvas, no console error.
import pygame  # noqa: F401

from interactive.drive import play_async

asyncio.run(play_async())
