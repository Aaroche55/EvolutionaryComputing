"""The genetic algorithm.

Each generation:
  1. pick population_size / 2 parent pairs uniformly at random (two distinct parents per pair);
  2. single point crossover gives two children per pair;
  3. bit-flip mutation on the children;
  4. evaluate the children (parents keep their cached fitness: the game is deterministic);
  5. pool parents + children and cull back to population_size with negative tournaments.

Parents compete with their children for survival, so good parents carry forward and the best fitness
in the population never gets worse. A run is reproducible from its seed, whatever the number of workers.
"""

from __future__ import annotations

import signal
import statistics
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from gallop.constants import STEP

from .fitness import DEFAULT_MAX_TIME, DEFAULT_TICKS_PER_ACTION, FITNESS_FUNCTIONS, Evaluation, evaluate
from .genome import GENE_BITS, N_GENES, Genome, random_genome
from .operators import bit_flip_mutation, negative_tournament_cull, single_point_crossover


def _ignore_sigint() -> None:
    """Worker initialiser: Ctrl-C is handled by the main process, which then shuts the workers down."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 400
    generations: int = 100
    n_genes: int = N_GENES
    tournament_size: int = 3
    mutation_rate: float | None = None  # per bit; None means 1 / genome length
    max_time: float = DEFAULT_MAX_TIME
    ticks_per_action: int = DEFAULT_TICKS_PER_ACTION
    fitness: str = "sum"  # see fitness.py
    seed: int = 0
    workers: int = 1  # processes used to evaluate genomes; 1 evaluates in this process

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("population_size must be an even number >= 2")
        if not 1 <= self.tournament_size <= self.population_size + 1:
            raise ValueError("tournament_size must be between 1 and population_size + 1")
        if self.mutation_rate is not None and not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.fitness not in FITNESS_FUNCTIONS:
            raise ValueError(f"fitness must be one of {FITNESS_FUNCTIONS}")
        if self.max_time > self.n_genes * self.ticks_per_action * STEP + 1e-9:
            raise ValueError("max_time is longer than the genome can play (GenomeExhausted); lower it or add genes")
        if self.generations < 0 or self.n_genes < 1 or self.workers < 1:
            raise ValueError("generations must be >= 0, n_genes and workers >= 1")

    @property
    def genome_bits(self) -> int:
        return self.n_genes * GENE_BITS

    @property
    def bit_mutation_rate(self) -> float:
        return 1 / self.genome_bits if self.mutation_rate is None else self.mutation_rate


@dataclass(frozen=True)
class GenerationStats:
    generation: int  # 0 is the initial population
    evaluations: int  # games played so far
    best: float
    worst: float
    mean: float
    median: float
    std: float
    best_progress: float
    best_time: float
    cleared: int  # individuals in the population that clear the level


@dataclass
class Population:
    genomes: npt.NDArray[np.uint8]  # shape (population_size, genome_bits)
    evaluations: list[Evaluation]

    @property
    def fitness(self) -> npt.NDArray[np.float64]:
        return np.array([e.fitness for e in self.evaluations])

    def best_index(self) -> int:
        return int(np.argmin(self.fitness))


@dataclass
class Checkpoint:
    """The state after a generation: enough to continue the run exactly as if it had never stopped."""

    generation: int
    population: Population
    rng_state: dict[str, Any]  # numpy bit generator state
    evaluations: int
    history: list[GenerationStats]

    @property
    def stats(self) -> GenerationStats:
        return self.history[-1]

    def best(self) -> tuple[Genome, Evaluation]:
        b = self.population.best_index()
        return self.population.genomes[b], self.population.evaluations[b]


@dataclass
class RunResult:
    config: GAConfig
    best_genome: Genome
    best: Evaluation
    history: list[GenerationStats] = field(default_factory=list)


class GeneticAlgorithm:
    def __init__(self, config: GAConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.evaluations = 0
        self._map: Callable[[list[Genome]], list[Evaluation]] = self._map_serial
        self._evaluate = partial(
            evaluate,
            max_time=config.max_time,
            ticks_per_action=config.ticks_per_action,
            fitness_function=config.fitness,
        )

    # ---------- evaluation

    def _map_serial(self, genomes: list[Genome]) -> list[Evaluation]:
        return [self._evaluate(g) for g in genomes]

    @contextmanager
    def _evaluator(self) -> Generator[None]:
        if self.config.workers == 1:
            yield
            return
        pool = ProcessPoolExecutor(self.config.workers, initializer=_ignore_sigint)
        chunk = max(1, self.config.population_size // (self.config.workers * 4))
        self._map = lambda genomes: list(pool.map(self._evaluate, genomes, chunksize=chunk))
        try:
            yield
        except BaseException:
            # Ctrl-C (or any error): drop queued evaluations instead of finishing the generation.
            pool.shutdown(cancel_futures=True)
            raise
        else:
            pool.shutdown()
        finally:
            self._map = self._map_serial

    def evaluate_all(self, genomes: npt.NDArray[np.uint8]) -> list[Evaluation]:
        self.evaluations += len(genomes)
        return self._map(list(genomes))

    # ---------- generations

    def initialise(self) -> Population:
        cfg = self.config
        genomes = np.stack([random_genome(self.rng, cfg.n_genes) for _ in range(cfg.population_size)])
        return Population(genomes, self.evaluate_all(genomes))

    def make_children(self, pop: Population) -> npt.NDArray[np.uint8]:
        cfg = self.config
        children = []
        for _ in range(cfg.population_size // 2):
            i, j = self.rng.choice(cfg.population_size, size=2, replace=False)
            children.extend(single_point_crossover(pop.genomes[i], pop.genomes[j], self.rng))
        kids = np.stack(children)
        bit_flip_mutation(kids, cfg.bit_mutation_rate, self.rng)
        return kids

    def next_generation(self, pop: Population) -> Population:
        children = self.make_children(pop)
        genomes = np.concatenate((pop.genomes, children))
        evaluations = pop.evaluations + self.evaluate_all(children)
        fitness = np.array([e.fitness for e in evaluations])
        keep = negative_tournament_cull(fitness, self.config.population_size, self.config.tournament_size, self.rng)
        return Population(genomes[keep], [evaluations[k] for k in keep])

    def stats(self, generation: int, pop: Population) -> GenerationStats:
        f = pop.fitness.tolist()
        best = pop.evaluations[pop.best_index()].result
        return GenerationStats(
            generation=generation,
            evaluations=self.evaluations,
            best=min(f),
            worst=max(f),
            mean=statistics.fmean(f),
            median=statistics.median(f),
            std=statistics.pstdev(f),
            best_progress=best.progress,
            best_time=best.time,
            cleared=sum(e.result.cleared for e in pop.evaluations),
        )

    def checkpoint(self, generation: int, pop: Population, history: list[GenerationStats]) -> Checkpoint:
        return Checkpoint(generation, pop, dict(self.rng.bit_generator.state), self.evaluations, list(history))

    def run(
        self,
        on_generation: Callable[[Checkpoint], None] | None = None,
        resume: Checkpoint | None = None,
    ) -> RunResult:
        """Run until generation config.generations (0 is the initial population) and return the best individual.

        on_generation is called with a Checkpoint after every generation, including 0. Passing one of those
        checkpoints as `resume` continues from it and gives the same results as an uninterrupted run.
        """
        cfg = self.config
        with self._evaluator():
            if resume is None:
                pop = self.initialise()
                history = [self.stats(0, pop)]
                if on_generation:
                    on_generation(self.checkpoint(0, pop, history))
            else:
                if resume.population.genomes.shape != (cfg.population_size, cfg.genome_bits):
                    raise ValueError("checkpoint population does not match the config")
                pop, history = resume.population, list(resume.history)
                self.rng.bit_generator.state = resume.rng_state
                self.evaluations = resume.evaluations
            for gen in range(len(history), cfg.generations + 1):
                pop = self.next_generation(pop)
                history.append(self.stats(gen, pop))
                if on_generation:
                    on_generation(self.checkpoint(gen, pop, history))
        b = pop.best_index()
        return RunResult(cfg, pop.genomes[b].copy(), pop.evaluations[b], history)


def config_dict(config: GAConfig) -> dict[str, Any]:
    return {**asdict(config), "genome_bits": config.genome_bits, "bit_mutation_rate": config.bit_mutation_rate}


def config_from_dict(data: dict[str, Any]) -> GAConfig:
    """Inverse of config_dict (ignores the derived fields)."""
    names = {f.name for f in fields(GAConfig)}
    return GAConfig(**{k: v for k, v in data.items() if k in names})
