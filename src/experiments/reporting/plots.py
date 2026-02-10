from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def plot_line_with_band(
    series: Dict[float, Dict[str, float]],
    out_path: Path,
    title: str,
    ylabel: str,
    mean_key: str,
    std_key: str,
) -> None:
    if not series:
        return
    gens = sorted(series.keys())
    means = [series[g].get(mean_key) for g in gens]
    stds = [series[g].get(std_key) for g in gens]
    if all(v is None for v in means):
        return

    plt.figure(figsize=(7, 4))
    plt.plot(gens, means, label="Mean")
    lower = [m - s if m is not None and s is not None else None for m, s in zip(means, stds)]
    upper = [m + s if m is not None and s is not None else None for m, s in zip(means, stds)]
    plt.fill_between(gens, lower, upper, alpha=0.2, label="Std")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bar_compare(
    groups: List[str],
    paper_values: List[float],
    our_values: List[float],
    out_path: Path,
    title: str,
    ylabel: str,
    paper_label: str,
    our_label: str,
) -> None:
    if not groups:
        return
    x = range(len(groups))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar([i - width / 2 for i in x], paper_values, width=width, label=paper_label)
    plt.bar([i + width / 2 for i in x], our_values, width=width, label=our_label)
    plt.xticks(list(x), groups)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_boxplots(
    values_by_group: Dict[str, List[float]],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    groups = [g for g, vals in values_by_group.items() if vals]
    if not groups:
        return
    data = [values_by_group[g] for g in groups]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data, labels=groups, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_multi_line(
    series_by_label: Dict[str, Dict[float, float]],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    if not series_by_label:
        return
    plt.figure(figsize=(8, 4))
    for label, series in series_by_label.items():
        if not series:
            continue
        gens = sorted(series.keys())
        values = [series[g] for g in gens]
        plt.plot(gens, values, label=label)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter_labels(
    points: Dict[str, tuple[float, float]],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not points:
        return
    plt.figure(figsize=(7, 4))
    for label, (x, y) in points.items():
        plt.scatter([x], [y], label=label)
        plt.text(x, y, label, fontsize=8, ha="left", va="bottom")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
