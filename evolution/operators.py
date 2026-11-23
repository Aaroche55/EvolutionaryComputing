"""Variation and selection operators. All randomness comes from the Generator passed in."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .genome import Genome


def single_point_crossover(a: Genome, b: Genome, rng: np.random.Generator) -> tuple[Genome, Genome]:
    """Cut both parents at the same random bit position and swap the tails.

    The cut point is uniform in 1..L-1, so each child always takes at least one bit from each parent.
    """
    if a.shape != b.shape:
        raise ValueError("parents must have the same length")
    point = int(rng.integers(1, a.size))
    return np.concatenate((a[:point], b[point:])), np.concatenate((b[:point], a[point:]))


def bit_flip_mutation(genomes: npt.NDArray[np.uint8], rate: float, rng: np.random.Generator) -> None:
    """Flip each bit independently with probability `rate`, in place."""
    if rate > 0:
        genomes ^= (rng.random(genomes.shape) < rate).astype(np.uint8)


def negative_tournament_cull(
    fitness: npt.NDArray[np.float64], survivors: int, k: int, rng: np.random.Generator
) -> npt.NDArray[np.intp]:
    """Shrink a pool to `survivors` individuals by repeated negative tournaments (minimisation).

    Each tournament draws k distinct individuals still in the pool and removes the one with the
    highest (worst) fitness. Returns the indices of the survivors in ascending order.

    The current best can only lose a tournament when every contender ties with it, so the best fitness
    in the pool never gets worse.
    """
    n = fitness.size
    if not 0 < survivors <= n:
        raise ValueError(f"survivors must be in 1..{n}")
    if not 1 <= k <= survivors + 1:
        raise ValueError("tournament size must be between 1 and survivors + 1")
    alive = np.arange(n)
    while alive.size > survivors:
        contenders = rng.choice(alive.size, size=k, replace=False)
        loser = contenders[np.argmax(fitness[alive[contenders]])]
        alive = np.delete(alive, loser)
    return alive
