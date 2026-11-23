"""Bitstring genome and its decoding into game actions.

A genome is N_GENES genes of GENE_BITS bits each (6 x 1000 = 6000 bits). Gene i is read as an unsigned
integer (most significant bit first) and taken modulo the number of actions to give the action for
decision step i. At 5 decisions per second, 1000 genes cover 200 s of play.

Note that 2**6 = 64 is not a multiple of the 6 actions, so the modulo gives actions 0-3 a probability of
11/64 each and actions 4-5 (LEFT_JUMP, RIGHT_JUMP) 10/64 each under a uniformly random gene.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gallop import N_ACTIONS

GENE_BITS = 6
N_GENES = 1000
GENOME_BITS = GENE_BITS * N_GENES

Genome = npt.NDArray[np.uint8]  # 1-D array of 0/1 values

_PLACE_VALUES = 1 << np.arange(GENE_BITS - 1, -1, -1)  # [32, 16, 8, 4, 2, 1]


def random_genome(rng: np.random.Generator, n_genes: int = N_GENES) -> Genome:
    """Every bit is an independent fair coin flip."""
    return rng.integers(0, 2, size=n_genes * GENE_BITS, dtype=np.uint8)


def decode(genome: Genome) -> list[int]:
    """Genome -> list of action values (0..N_ACTIONS-1), one per gene."""
    if genome.ndim != 1 or genome.size % GENE_BITS:
        raise ValueError(f"genome must be a flat array whose length is a multiple of {GENE_BITS}")
    values = genome.reshape(-1, GENE_BITS).astype(np.int64) @ _PLACE_VALUES
    return (values % N_ACTIONS).tolist()


def encode(actions: list[int]) -> Genome:
    """Action values -> a genome that decodes back to them (each gene holds the action value itself)."""
    values = np.asarray(actions, dtype=np.int64)
    if values.ndim != 1 or (values < 0).any() or (values >= N_ACTIONS).any():
        raise ValueError(f"actions must be ints in 0..{N_ACTIONS - 1}")
    bits = (values[:, None] & _PLACE_VALUES) > 0
    return bits.astype(np.uint8).ravel()
