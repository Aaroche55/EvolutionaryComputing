"""Genetic algorithm that evolves bitstring action sequences to play Ballinasloe Gallop (gallop package)."""

from .fitness import Evaluation, evaluate, fitness
from .ga import GAConfig, GenerationStats, GeneticAlgorithm, RunResult
from .genome import GENE_BITS, GENOME_BITS, N_GENES, decode, encode, random_genome

__all__ = [
    "GENE_BITS",
    "GENOME_BITS",
    "N_GENES",
    "Evaluation",
    "GAConfig",
    "GenerationStats",
    "GeneticAlgorithm",
    "RunResult",
    "decode",
    "encode",
    "evaluate",
    "fitness",
    "random_genome",
]
