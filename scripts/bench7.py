from __future__ import annotations

"""Thin wrapper so the benchmark can be run directly from the repository checkout."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcbj_iic.cli import main           # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
