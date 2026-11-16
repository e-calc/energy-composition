#!/usr/bin/env python
"""Measure the GPU's static power and store it.

Thin wrapper kept for existing workflows; the same command is ``ecalc measure static-power``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecalc.measurement_commands import static_power_main  # noqa: E402

if __name__ == "__main__":
    static_power_main()
