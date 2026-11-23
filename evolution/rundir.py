"""Saving a GA run to disk as it goes, so it can be resumed, inspected and replayed.

A run directory holds:
    config.json               the GAConfig (everything needed to reproduce the run)
    history.csv               per-generation statistics, one row per generation
    best_per_generation.jsonl best individual of every generation: fitness, game result, bitstring
    population_fitness.csv    fitness, progress, time and outcome of every individual in every generation
    best.json                 best individual so far (same fields)
    best_actions.json         its decoded actions, playable with `python -m gallop replay`
    checkpoint.npz            the latest whole population + RNG state, for --resume

Every file is rewritten after each generation, and checkpoint.npz is replaced atomically, so stopping a
run at any moment (even Ctrl-C mid-generation) loses at most the generation in progress.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from gallop import Result

from .fitness import Evaluation
from .ga import Checkpoint, GAConfig, GenerationStats, Population, config_dict, config_from_dict
from .genome import Genome, decode

POPULATION_COLUMNS = ["generation", "individual", "fitness", "progress", "time", "cleared", "death_cause"]


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        write(f)
    tmp.replace(path)


def _bits_to_str(genome: Genome) -> str:
    return "".join(map(str, genome.tolist()))


def _bits_from_str(bits: str) -> Genome:
    return np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")


def individual_record(generation: int, genome: Genome, ev: Evaluation) -> dict[str, Any]:
    return {"generation": generation, "fitness": ev.fitness, "result": asdict(ev.result), "bits": _bits_to_str(genome)}


def genome_from_record(record: dict[str, Any]) -> Genome:
    return _bits_from_str(record["bits"])


class RunDir:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.config_file = self.path / "config.json"
        self.history_file = self.path / "history.csv"
        self.bests_file = self.path / "best_per_generation.jsonl"
        self.population_file = self.path / "population_fitness.csv"
        self.best_file = self.path / "best.json"
        self.best_actions_file = self.path / "best_actions.json"
        self.checkpoint_file = self.path / "checkpoint.npz"

    # ---------- writing

    def start(self, config: GAConfig) -> None:
        """Begin a new run here. Refuses to overwrite an existing run."""
        if self.checkpoint_file.exists():
            raise FileExistsError(f"{self.path} already holds a run; resume it or choose another directory")
        self.path.mkdir(parents=True, exist_ok=True)
        self.bests_file.unlink(missing_ok=True)
        self.population_file.unlink(missing_ok=True)
        self.save_config(config)

    def save_config(self, config: GAConfig) -> None:
        self.config_file.write_text(json.dumps(config_dict(config), indent=2) + "\n")

    def record(self, cp: Checkpoint) -> None:
        """Save everything about the generation just finished. Call from GeneticAlgorithm.run(on_generation=...)."""
        genome, ev = cp.best()
        record = individual_record(cp.generation, genome, ev)
        with self.bests_file.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.best_file.write_text(json.dumps(record, indent=2) + "\n")
        self.best_actions_file.write_text(json.dumps(decode(genome)) + "\n")
        self._append_population(cp)
        with self.history_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(GenerationStats)])
            writer.writeheader()
            writer.writerows(asdict(s) for s in cp.history)
        # Written last: a checkpoint for generation N means every file above covers generation N too.
        self.save_checkpoint(cp)

    def _append_population(self, cp: Checkpoint) -> None:
        new = not self.population_file.exists()
        with self.population_file.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(POPULATION_COLUMNS)
            # Sorted best first, so `individual` is the rank within the generation.
            ranked = sorted(cp.population.evaluations, key=lambda e: e.fitness)
            for rank, e in enumerate(ranked):
                r = e.result
                writer.writerow([cp.generation, rank, e.fitness, r.progress, r.time, r.cleared, r.death_cause or ""])

    def save_checkpoint(self, cp: Checkpoint) -> None:
        pop = cp.population
        meta = {
            "generation": cp.generation,
            "rng_state": cp.rng_state,
            "evaluations": cp.evaluations,
            "history": [asdict(s) for s in cp.history],
            "population": [{"fitness": e.fitness, "result": asdict(e.result)} for e in pop.evaluations],
            "genome_bits": pop.genomes.shape[1],
        }
        _write_atomic(
            self.checkpoint_file,
            lambda f: np.savez_compressed(f, genomes=np.packbits(pop.genomes, axis=1), meta=np.array(json.dumps(meta))),
        )

    # ---------- reading

    def load_config(self) -> GAConfig:
        return config_from_dict(json.loads(self.config_file.read_text()))

    def load_checkpoint(self) -> Checkpoint:
        with np.load(self.checkpoint_file, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            genomes = np.unpackbits(data["genomes"], axis=1, count=meta["genome_bits"])
        evaluations = [Evaluation(p["fitness"], Result(**p["result"])) for p in meta["population"]]
        return Checkpoint(
            generation=meta["generation"],
            population=Population(genomes, evaluations),
            rng_state=meta["rng_state"],
            evaluations=meta["evaluations"],
            history=[GenerationStats(**s) for s in meta["history"]],
        )

    def bests(self) -> list[dict[str, Any]]:
        """Best individual of each generation, in order."""
        if not self.bests_file.exists():
            return []
        lines = self.bests_file.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def population_fitness(self) -> list[dict[str, str]]:
        """Rows of population_fitness.csv (values as strings, as csv reads them)."""
        if not self.population_file.exists():
            return []
        with self.population_file.open(newline="") as f:
            return list(csv.DictReader(f))

    def truncate_to(self, last_generation: int) -> None:
        """Drop per-generation records newer than the checkpoint.

        They exist if a run was stopped after recording a generation but before its checkpoint was saved;
        resuming will produce those generations again.
        """
        kept = [r for r in self.bests() if r["generation"] <= last_generation]
        self.bests_file.write_text("".join(json.dumps(r) + "\n" for r in kept))
        if self.population_file.exists():
            rows = [r for r in self.population_fitness() if int(r["generation"]) <= last_generation]
            with self.population_file.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=POPULATION_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

    def best_record(self, generation: int | None = None) -> dict[str, Any]:
        """Best of the given generation, or the best individual of the whole run so far."""
        bests = self.bests()
        if not bests:
            raise FileNotFoundError(f"no generations recorded in {self.path}")
        if generation is None:
            return min(reversed(bests), key=lambda r: r["fitness"])  # latest among equal bests
        for r in bests:
            if r["generation"] == generation:
                return r
        raise KeyError(f"generation {generation} not recorded (have 0..{bests[-1]['generation']})")
