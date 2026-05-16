"""Shared repository paths for the Climate Frames workspace."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CURRENT_DATA_DIR = DATA_DIR / "current"
LEGACY_DATA_DIR = DATA_DIR / "legacy"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
OUTPUTS_DIR = REPO_ROOT / "outputs"
REQUIREMENTS_DIR = REPO_ROOT / "requirements"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

