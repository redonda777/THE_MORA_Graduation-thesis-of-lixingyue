from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CONTROL_DIR = Path(__file__).resolve().parent


def main() -> int:
    command = [
        sys.executable,
        str(BASE_DIR / "gvnm" / "community_detection.py"),
        "--version-matrix",
        str(CONTROL_DIR / "version_similarity_matrix.csv"),
        "--tree-json",
        str(BASE_DIR / "mora_v4.1_0406.json"),
        "--similarity-json",
        str(CONTROL_DIR / "standard_edit_count_sentence_edges.json"),
        "--output-dir",
        str(CONTROL_DIR / "gvnm_output"),
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
