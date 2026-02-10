from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

from .pipeline_config import FULL_INSTANCES, SMALL_INSTANCES


def build_seed_list(value: str) -> List[int]:
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    count = int(value)
    return list(range(1, count + 1))


def resolve_instances(mode: str, base_dir: Path) -> Tuple[List[str], List[str]]:
    instances: List[str] = []
    missing: List[str] = []

    if mode == "small":
        for group, name in SMALL_INSTANCES.items():
            path = base_dir / f"{name}.txt"
            if path.exists():
                instances.append(str(path))
            else:
                missing.append(str(path))
        return instances, missing

    for group, names in FULL_INSTANCES.items():
        for name in names:
            path = base_dir / f"{name}.txt"
            if path.exists():
                instances.append(str(path))
            else:
                missing.append(str(path))

    return instances, missing


def safe_clean_dir(
    target_dir: Path, archive_old: bool, archive_root: Path, archive_label: str
) -> None:
    if not target_dir.exists():
        return
    if archive_old:
        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_dir = archive_root / f"{ts}_{archive_label}"
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_dir), str(archive_dir))
    else:
        shutil.rmtree(target_dir)


def ensure_within_results(path: Path, results_root: Path) -> None:
    resolved_root = results_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to operate outside results root: {path}") from exc
