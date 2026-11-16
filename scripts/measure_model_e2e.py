#!/usr/bin/env python
"""Measure each whole model with the same Zeus protocol (the reference for a composition).

Thin wrapper kept for existing workflows; the same command is ``ecalc measure model``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecalc.measurement_commands import model_main  # noqa: E402

if __name__ == "__main__":
    model_main()
