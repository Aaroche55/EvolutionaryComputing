"""Compare the game engines' throughput on large populations.

    uv run --extra gpu python benchmarks/engines.py 10000 100000 1000000

For each population size N it plays N games of level 1 (80 s limit, 5 decisions per second) with:
  numba  gallop.fast, one game per CPU core at a time (numba prange: all cores, no process overhead)
  mps    gallop.gpu on the Apple GPU (float32), compiled with torch.compile

on two populations:
  random  random genomes, as in generation 0: most fall within seconds
  late    mutants of the best genome of runs/seed6_pop2000 (~5 bit flips each), as late in training:
          most run the whole level, about 26 s of game time each
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from numba import njit, prange

from evolution.genome import decode, random_genome
from evolution.rundir import RunDir, genome_from_record
from gallop.fast import _DIRECTION, _JUMP, _level_arrays, _play
from gallop.gpu import simulate_batch

MAX_TIME = 80.0
TICKS_PER_ACTION = 24
MAX_TICKS = round(MAX_TIME * 120)


@njit(parallel=True, cache=True)
def _play_many(actions, grid, coins, geese, sx, sy, goal_x, tpa, max_ticks, direction_of, jump_of):
    n = actions.shape[0]
    cleared = np.zeros(n, dtype=np.bool_)
    times = np.zeros(n)
    for i in prange(n):
        _, t, counts = _play(
            actions[i], grid, coins, geese, sx, sy, goal_x, tpa, max_ticks, True, direction_of, jump_of
        )
        cleared[i] = counts[0] == 1
        times[i] = t
    return cleared, times


def numba_many(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid, coins, geese, sx, sy, goal_x = _level_arrays(1)
    a = actions.astype(np.int64)
    return _play_many(a, grid, coins, geese, sx, sy, goal_x, TICKS_PER_ACTION, MAX_TICKS, _DIRECTION, _JUMP)


def populations(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    rand = np.stack([np.array(decode(random_genome(rng, 400)), dtype=np.uint8) for _ in range(n)])
    best = genome_from_record(RunDir("runs/seed6_pop2000").best_record())
    late = np.tile(best, (n, 1))
    late ^= (rng.random(late.shape) < 5 / late.shape[1]).astype(np.uint8)
    late_actions = (late.reshape(n, -1, 6) @ (1 << np.arange(5, -1, -1))) % 6
    return {"random": rand, "late": late_actions.astype(np.uint8)}


def timed(f, *args):
    t = time.perf_counter()
    r = f(*args)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return r, time.perf_counter() - t


def main(sizes: list[int]) -> None:
    rng = np.random.default_rng(0)
    # Warm up: compile both engines before timing.
    warm = populations(64, rng)["random"]
    numba_many(warm)
    simulate_batch(warm, max_time=MAX_TIME, device="mps", compile=True)

    print(f"{'population':>10} {'N':>9} {'numba (12 cores)':>22} {'GPU (mps, compiled)':>24} {'GPU/numba':>10}")
    for n in sizes:
        for name, acts in populations(n, rng).items():
            (cl_n, _), t_n = timed(numba_many, acts)
            r, t_g = timed(lambda a: simulate_batch(a, max_time=MAX_TIME, device="mps", compile=True), acts)
            agree = (r.cleared == cl_n).mean()
            print(
                f"{name:>10} {n:>9} {t_n:8.2f}s {n / t_n:9.0f}/s {t_g:9.2f}s {n / t_g:9.0f}/s {t_n / t_g:9.2f}x"
                f"   clears: numba {cl_n.mean():.1%}, GPU {r.cleared.mean():.1%}, same outcome {agree:.1%}",
                flush=True,
            )


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [10_000, 100_000])
