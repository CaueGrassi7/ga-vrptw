from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .distance import build_distance_matrix
from .model import Customer, Instance


@dataclass(frozen=True)
class ParsedHeader:
    name: str
    capacity: float
    vehicle_count: int | None


def _parse_header(lines: List[str]) -> ParsedHeader:
    name = lines[0].strip() if lines else "UNKNOWN"
    capacity: float | None = None
    vehicle_count: int | None = None

    cap_idx = None
    for i, line in enumerate(lines):
        if "CAPACITY" in line.upper():
            cap_idx = i
            break

    if cap_idx is not None:
        # Usually the next non-empty line has: vehicle_count capacity
        for j in range(cap_idx + 1, len(lines)):
            parts = lines[j].split()
            if not parts:
                continue
            if len(parts) >= 2 and parts[0].isdigit():
                vehicle_count = int(parts[0])
                capacity = float(parts[1])
                break
    if capacity is None:
        raise ValueError("Could not parse vehicle capacity from Solomon header.")

    return ParsedHeader(name=name, capacity=capacity, vehicle_count=vehicle_count)


def _find_customer_lines(lines: List[str]) -> List[str]:
    # Customer table usually starts after a line containing "CUST" or "CUSTOMER"
    start = None
    for i, line in enumerate(lines):
        if "CUST" in line.upper() and "NO" in line.upper():
            start = i + 1
            break
    if start is None:
        # fallback: find first line with >= 7 numeric fields
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 7 and all(_is_number(p) for p in parts[:7]):
                start = i
                break
    if start is None:
        raise ValueError("Could not locate customer table in Solomon instance.")

    cust_lines = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        if not all(_is_number(p) for p in parts[:7]):
            continue
        cust_lines.append(line)

    if not cust_lines:
        raise ValueError("No customer rows parsed from Solomon instance.")

    return cust_lines


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_solomon(path: str | Path) -> Instance:
    """Parse a Solomon VRPTW instance from a .txt file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instance not found: {path}")

    lines = [line.rstrip("\n") for line in path.read_text().splitlines()]
    header = _parse_header(lines)
    cust_lines = _find_customer_lines(lines)

    customers: Dict[int, Customer] = {}
    for line in cust_lines:
        parts = line.split()
        cid = int(parts[0])
        customer = Customer(
            id=cid,
            x=float(parts[1]),
            y=float(parts[2]),
            demand=float(parts[3]),
            ready_time=float(parts[4]),
            due_date=float(parts[5]),
            service_time=float(parts[6]),
        )
        customers[cid] = customer

    if 0 not in customers:
        raise ValueError("Depot with id 0 not found in instance.")

    distance_matrix, id_order, id_to_index = build_distance_matrix(customers)
    return Instance(
        name=header.name,
        capacity=header.capacity,
        depot_id=0,
        customers=customers,
        distance_matrix=distance_matrix,
        id_order=id_order,
        id_to_index=id_to_index,
    )
