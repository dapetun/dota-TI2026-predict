"""Deprecated wrapper around scripts/download_details.py.

Prefer: ``python scripts/download_details.py``

Delegates to the atomic resume-friendly downloader (``*_matchlist.json``).
"""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

_TARGET = Path(__file__).with_name("download_details.py")


def main() -> None:
    """Print deprecation warning and run download_details CLI."""
    warnings.warn(
        "scripts/download_details_bg.py is deprecated; use scripts/download_details.py",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "WARNING: download_details_bg.py is deprecated. "
        "Use: python scripts/download_details.py",
        file=sys.stderr,
    )
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
