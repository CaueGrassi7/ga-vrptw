from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


GROUP_PREFIX = {
    "C1": "C1",
    "C2": "C2",
    "R1": "R1",
    "R2": "R2",
    "RC1": "RC1",
    "RC2": "RC2",
}


def list_solomon_instances(data_dir: Path, groups: Iterable[str]) -> List[str]:
    instances: List[str] = []
    for group in groups:
        prefix = GROUP_PREFIX.get(group.upper())
        if not prefix:
            continue
        pattern = f"{prefix}*.txt"
        for path in sorted(data_dir.glob(pattern)):
            instances.append(str(path))
    return instances


def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]
