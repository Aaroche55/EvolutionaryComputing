"""Plot a run's fitness and clear time across generations.

    python -m evolution plot runs/seed2            -> runs/seed2/progress.png

Two panels share the generation axis (fitness and seconds have different scales, so they never share a
y-axis):
  * fitness: best and mean per generation, with the population's interquartile range shaded;
  * clear time: the fastest and the median clear time among individuals that clear the level, with the
    generation of the first clear marked.
Reads history.csv and population_fitness.csv; works while a run is still training.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .rundir import RunDir

# Reference data-viz palette (light surface); see the dataviz skill's palette.md.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue: best
SERIES_2 = "#eb6834"  # orange: mean / median
BAND = "#cde2fb"  # blue 100: population interquartile range

Floats = npt.NDArray[np.float64]


@dataclass
class Curves:
    generation: npt.NDArray[np.int64]
    best: Floats
    mean: Floats
    q25: Floats | None  # population interquartile range, if population_fitness.csv exists
    q75: Floats | None
    fastest_clear: Floats  # NaN where no individual cleared
    median_clear: Floats
    first_clear: int | None


def load_curves(run_dir: RunDir) -> Curves:
    with run_dir.history_file.open(newline="") as f:
        history = list(csv.DictReader(f))
    gens = np.array([int(r["generation"]) for r in history])
    best = np.array([float(r["best"]) for r in history])
    mean = np.array([float(r["mean"]) for r in history])

    n = len(gens)
    q25 = q75 = None
    fastest = np.full(n, np.nan)
    median = np.full(n, np.nan)
    rows = run_dir.population_fitness()
    if rows:
        fitness: dict[int, list[float]] = {}
        clear_times: dict[int, list[float]] = {}
        for r in rows:
            g = int(r["generation"])
            fitness.setdefault(g, []).append(float(r["fitness"]))
            if r["cleared"] == "True":
                clear_times.setdefault(g, []).append(float(r["time"]))
        q25 = np.array([np.percentile(fitness[g], 25) if g in fitness else np.nan for g in gens])
        q75 = np.array([np.percentile(fitness[g], 75) if g in fitness else np.nan for g in gens])
        for i, g in enumerate(gens):
            if g in clear_times:
                fastest[i] = min(clear_times[g])
                median[i] = float(np.median(clear_times[g]))
    else:
        # Older runs without per-individual data: the best individual's time, where it cleared.
        best_time = np.array([float(r["best_time"]) for r in history])
        cleared = np.array([int(r["cleared"]) > 0 for r in history])
        fastest = np.where(cleared & (best < 1.0), best_time, np.nan)

    cleared_gens = gens[~np.isnan(fastest)]
    first = int(cleared_gens[0]) if cleared_gens.size else None
    return Curves(gens, best, mean, q25, q75, fastest, median, first)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)


def _end_label(ax, x, y, text: str) -> None:
    """Direct label at the right end of a line, in text ink (the line beside it carries the colour)."""
    if np.isnan(y):
        return
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)


def plot(run_dir: RunDir, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = load_curves(run_dir)
    fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor(SURFACE)
    name = run_dir.path.name
    fig.suptitle(f"GA progress: {name}", x=0.06, ha="left", fontsize=13, fontweight="bold", color=TEXT)

    # Fitness
    _style(ax_f)
    if c.q25 is not None and c.q75 is not None:
        ax_f.fill_between(c.generation, c.q25, c.q75, color=BAND, linewidth=0, label="Population middle 50%")
    ax_f.plot(c.generation, c.mean, color=SERIES_2, linewidth=2, label="Mean")
    ax_f.plot(c.generation, c.best, color=SERIES_1, linewidth=2, label="Best")
    ax_f.set_title("Fitness (lower is better)", loc="left", fontsize=11, color=TEXT)
    ax_f.set_ylabel("Fitness")
    ax_f.set_ylim(bottom=0)
    ax_f.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT)
    _end_label(ax_f, c.generation[-1], c.best[-1], f"{c.best[-1]:.3f}")

    # Clear time
    _style(ax_t)
    ax_t.set_title("Time to clear the level", loc="left", fontsize=11, color=TEXT)
    ax_t.set_xlabel("Generation")
    ax_t.set_ylabel("Seconds")
    if c.first_clear is None:
        ax_t.text(
            0.5,
            0.5,
            "No individual has cleared the level yet",
            transform=ax_t.transAxes,
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=10,
        )
        ax_t.set_yticks([])
    else:
        if not np.isnan(c.median_clear).all():
            ax_t.plot(c.generation, c.median_clear, color=SERIES_2, linewidth=2, label="Median of clearers")
        ax_t.plot(c.generation, c.fastest_clear, color=SERIES_1, linewidth=2, label="Fastest")
        ax_t.axvline(c.first_clear, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        # Bottom of the panel: right after the first clear, the lines sit near the top.
        ax_t.annotate(
            f"first clear: gen {c.first_clear}",
            (c.first_clear, 0),
            xycoords=("data", "axes fraction"),
            xytext=(5, 5),
            textcoords="offset points",
            va="bottom",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
        ax_t.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT)
        _end_label(ax_t, c.generation[-1], c.fastest_clear[-1], f"{c.fastest_clear[-1]:.2f}s")
    ax_t.set_xlim(c.generation[0], c.generation[-1])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out
