#!/usr/bin/env python
"""Capture every kernel of the given models and measure each unique kernel in isolation.

Thin wrapper kept for existing workflows; the same command is ``ecalc measure kernels``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecalc.measurement_commands import kernels_main  # noqa: E402

if __name__ == "__main__":
    kernels_main()
