"""``python -m ecalc`` is the ``ecalc`` command."""

import sys

from .command_line import main

if __name__ == "__main__":
    sys.exit(main())
