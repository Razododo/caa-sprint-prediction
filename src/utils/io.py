"""
I/O helper functions for the CAA Sprint pipeline.
Handles config loading, parquet I/O, and result serialization.
"""
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    """Load project configuration from config.yaml."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    # Resolve all paths relative to project root
    for key, val in config.get("paths", {}).items():
        config["paths"][key] = str(PROJECT_ROOT / val)
    return config


def ensure_dirs(config: dict) -> None:
    """Create all output directories if they don't exist."""
    for key, path_str in config.get("paths", {}).items():
        Path(path_str).mkdir(parents=True, exist_ok=True)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file with logging."""
    path = Path(path)
    logger.info(f"Reading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"  → {len(df):,} rows × {len(df.columns)} columns")
    return df


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a dataframe to parquet with logging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Wrote {path.name}: {len(df):,} rows × {len(df.columns)} cols")


def save_results_meta(results: dict[str, Any], path: str | Path) -> None:
    """Save results metadata as JSON (for reproducibility tracking)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    path.write_text(json.dumps(meta, indent=2, default=str))
    logger.info(f"Saved metadata to {path.name}")
