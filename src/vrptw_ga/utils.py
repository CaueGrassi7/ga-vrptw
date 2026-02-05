from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Create and return a NumPy RNG with a fixed seed."""

    return np.random.default_rng(seed)


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, default=_to_jsonable))


def append_csv(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
