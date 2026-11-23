import csv
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest

from evolution import GENE_BITS, GENOME_BITS, GAConfig, GeneticAlgorithm, decode, encode, evaluate, fitness
from evolution.__main__ import main as cli
from evolution.genome import random_genome
from evolution.operators import bit_flip_mutation, negative_tournament_cull, single_point_crossover
from evolution.rundir import RunDir, genome_from_record
from gallop import N_ACTIONS, Result


def bits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s], dtype=np.uint8)


def result(progress=0.0, time=0.0, cleared=False) -> Result:
    return Result(
        progress=progress,
        cleared=cleared,
        died=False,
        death_cause=None,
        timed_out=not cleared,
        time=time,
        ticks=0,
        steps=0,
        horseshoes=0,
        horseshoes_total=0,
        geese_stomped=0,
    )


# ---------- genome


def test_genome_is_6000_bits():
    assert GENOME_BITS == 6 * 1000
    g = random_genome(np.random.default_rng(0))
    assert g.shape == (GENOME_BITS,)
    assert set(np.unique(g)) <= {0, 1}


def test_random_genome_is_fair_and_seeded():
    g = random_genome(np.random.default_rng(1))
    assert 0.47 < g.mean() < 0.53
    assert np.array_equal(g, random_genome(np.random.default_rng(1)))
    assert not np.array_equal(g, random_genome(np.random.default_rng(2)))


@pytest.mark.parametrize(
    ("gene", "action"),
    [("000000", 0), ("000101", 5), ("000110", 0), ("000111", 1), ("111111", 63 % 6), ("100000", 32 % 6)],
)
def test_decode_gene_modulo_actions(gene, action):
    assert decode(bits(gene)) == [action]


def test_decode_reads_genes_in_order():
    assert decode(bits("000001000010000011")) == [1, 2, 3]


def test_decode_rejects_partial_gene():
    with pytest.raises(ValueError, match="multiple of 6"):
        decode(bits("0101"))


def test_encode_round_trips():
    actions = list(range(N_ACTIONS)) * 3
    g = encode(actions)
    assert g.size == len(actions) * GENE_BITS
    assert decode(g) == actions
    with pytest.raises(ValueError, match="actions must be ints"):
        encode([N_ACTIONS])


def test_decoded_actions_are_valid():
    actions = decode(random_genome(np.random.default_rng(3)))
    assert len(actions) == 1000
    assert all(0 <= a < N_ACTIONS for a in actions)


# ---------- fitness


def test_fitness_terms():
    assert fitness(result(progress=0.0, time=200.0), max_time=200) == 2.0  # stuck at the start
    assert fitness(result(progress=1.0, time=20.0, cleared=True), max_time=200) == pytest.approx(0.1)
    assert fitness(result(progress=0.5, time=50.0), max_time=200) == pytest.approx(0.75)


def test_fitness_time_term_capped():
    assert fitness(result(progress=0.5, time=500.0), max_time=200) == pytest.approx(1.5)


def test_evaluate_solution_clears_level(solution):
    genome = encode(solution + [0] * (1000 - len(solution)))
    ev = evaluate(genome)
    assert ev.result.cleared
    assert ev.fitness == pytest.approx(ev.result.time / 200)


def test_evaluate_full_genome_never_exhausts():
    # All NOOP: the player stands still until the 200 s limit, using all 1000 genes.
    ev = evaluate(encode([0] * 1000))
    assert ev.result.timed_out
    assert ev.result.steps == 1000
    assert ev.fitness == pytest.approx(2.0)


# ---------- operators


def test_single_point_crossover_swaps_tails():
    rng = np.random.default_rng(0)
    a, b = np.zeros(100, dtype=np.uint8), np.ones(100, dtype=np.uint8)
    for _ in range(50):
        c1, c2 = single_point_crossover(a, b, rng)
        point = int(np.argmax(c1))  # first 1 in c1 = cut point
        assert 1 <= point <= 99
        assert (c1[:point] == 0).all()
        assert (c1[point:] == 1).all()
        assert np.array_equal(c2, 1 - c1)  # children are complementary


def test_crossover_rejects_mismatched_parents():
    with pytest.raises(ValueError, match="same length"):
        single_point_crossover(np.zeros(6, np.uint8), np.zeros(12, np.uint8), np.random.default_rng(0))


def test_mutation_rates():
    rng = np.random.default_rng(0)
    g = np.zeros((10, 6000), dtype=np.uint8)
    bit_flip_mutation(g, 0.0, rng)
    assert g.sum() == 0
    bit_flip_mutation(g, 1.0, rng)
    assert (g == 1).all()
    g[:] = 0
    bit_flip_mutation(g, 0.01, rng)
    assert 0.008 < g.mean() < 0.012


def test_cull_keeps_requested_number_and_best():
    rng = np.random.default_rng(0)
    for _ in range(20):
        f = rng.random(800)
        keep = negative_tournament_cull(f, 400, 3, rng)
        assert keep.size == 400
        assert np.unique(keep).size == 400
        assert np.argmin(f) in keep


def test_cull_tournament_over_whole_pool_removes_the_worst():
    f = np.array([0.3, 0.9, 0.1, 0.5])
    keep = negative_tournament_cull(f, 3, 4, np.random.default_rng(0))  # one tournament containing everyone
    assert keep.tolist() == [0, 2, 3]


def test_cull_rejects_bad_sizes():
    f = np.zeros(10)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="survivors"):
        negative_tournament_cull(f, 11, 3, rng)
    with pytest.raises(ValueError, match="tournament size"):
        negative_tournament_cull(f, 4, 6, rng)


def test_cull_pressure_increases_with_k():
    rng = np.random.default_rng(0)
    f = rng.random(800)
    mean_k1 = f[negative_tournament_cull(f, 400, 1, rng)].mean()  # k=1: removal is random
    mean_k5 = f[negative_tournament_cull(f, 400, 5, rng)].mean()
    assert mean_k5 < mean_k1


# ---------- GA


def small_config(seed: int, workers: int = 1, generations: int = 3) -> GAConfig:
    """A GA small enough to run in a test: 10 individuals, 3 generations, 100 genes, 20 s games."""
    return GAConfig(population_size=10, generations=generations, n_genes=100, max_time=20.0, seed=seed, workers=workers)


def test_config_defaults():
    c = GAConfig()
    assert c.population_size == 400
    assert c.genome_bits == 6000
    assert c.bit_mutation_rate == pytest.approx(1 / 6000)
    assert c.max_time == 200
    with pytest.raises(ValueError, match="even"):
        GAConfig(population_size=401)


def test_ga_runs_and_best_never_worsens():
    run = GeneticAlgorithm(small_config(1)).run()
    assert len(run.history) == 3 + 1
    bests = [s.best for s in run.history]
    assert all(b2 <= b1 for b1, b2 in itertools.pairwise(bests))
    assert run.best.fitness == bests[-1]
    assert run.history[-1].evaluations == 10 + 3 * 10
    assert evaluate(run.best_genome, max_time=20.0).fitness == run.best.fitness


def test_ga_is_reproducible_from_seed():
    h1 = GeneticAlgorithm(small_config(7)).run().history
    h2 = GeneticAlgorithm(small_config(7)).run().history
    h3 = GeneticAlgorithm(small_config(8)).run().history
    assert h1 == h2
    assert h1 != h3


def test_parallel_matches_serial():
    serial = GeneticAlgorithm(small_config(3)).run()
    parallel = GeneticAlgorithm(small_config(3, workers=2)).run()
    assert serial.history == parallel.history
    assert np.array_equal(serial.best_genome, parallel.best_genome)


# ---------- checkpoints and run directories


def test_resume_matches_uninterrupted_run():
    full = GeneticAlgorithm(small_config(5, generations=4)).run()
    checkpoints = []
    GeneticAlgorithm(small_config(5, generations=2)).run(on_generation=checkpoints.append)
    assert [cp.generation for cp in checkpoints] == [0, 1, 2]
    resumed = GeneticAlgorithm(small_config(5, generations=4)).run(resume=checkpoints[-1])
    assert resumed.history == full.history
    assert np.array_equal(resumed.best_genome, full.best_genome)


def test_resume_rejects_mismatched_population():
    checkpoints = []
    GeneticAlgorithm(small_config(5, generations=0)).run(on_generation=checkpoints.append)
    bigger = GAConfig(population_size=12, n_genes=100, max_time=20.0, generations=2)
    with pytest.raises(ValueError, match="does not match"):
        GeneticAlgorithm(bigger).run(resume=checkpoints[0])


def test_run_dir_records_every_generation(tmp_path):
    rd = RunDir(tmp_path / "run")
    config = small_config(2)
    rd.start(config)
    run = GeneticAlgorithm(config).run(on_generation=rd.record)

    assert rd.load_config() == config
    bests = rd.bests()
    assert [b["generation"] for b in bests] == [0, 1, 2, 3]
    assert [b["fitness"] for b in bests] == [s.best for s in run.history]
    best = rd.best_record()
    assert best["fitness"] == run.best.fitness
    assert np.array_equal(genome_from_record(best), run.best_genome)
    assert json.loads(rd.best_actions_file.read_text()) == decode(run.best_genome)
    assert rd.best_record(1)["generation"] == 1
    with pytest.raises(KeyError, match="generation 9"):
        rd.best_record(9)
    with (tmp_path / "run" / "history.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 4


def test_run_dir_refuses_to_overwrite(tmp_path):
    rd = RunDir(tmp_path)
    rd.start(small_config(2, generations=0))
    GeneticAlgorithm(small_config(2, generations=0)).run(on_generation=rd.record)
    with pytest.raises(FileExistsError, match="resume"):
        rd.start(small_config(2))


def test_checkpoint_file_round_trip_and_resume(tmp_path):
    full = GeneticAlgorithm(small_config(9, generations=4)).run()
    rd = RunDir(tmp_path)
    rd.start(small_config(9, generations=2))
    GeneticAlgorithm(small_config(9, generations=2)).run(on_generation=rd.record)

    cp = rd.load_checkpoint()
    assert cp.generation == 2
    assert cp.population.genomes.shape == (10, 600)
    assert cp.population.genomes.dtype == np.uint8
    resumed = GeneticAlgorithm(small_config(9, generations=4)).run(on_generation=rd.record, resume=cp)
    assert resumed.history == full.history
    assert [b["generation"] for b in rd.bests()] == [0, 1, 2, 3, 4]


def test_population_fitness_recorded_for_every_individual(tmp_path):
    rd = RunDir(tmp_path)
    rd.start(small_config(6))
    run = GeneticAlgorithm(small_config(6)).run(on_generation=rd.record)
    rows = rd.population_fitness()
    assert len(rows) == 4 * 10  # 4 generations x 10 individuals
    for stats in run.history:
        gen = [float(r["fitness"]) for r in rows if int(r["generation"]) == stats.generation]
        assert gen == sorted(gen)
        assert gen[0] == stats.best
        assert gen[-1] == stats.worst
        assert np.mean(gen) == pytest.approx(stats.mean)


def test_truncate_drops_records_after_checkpoint(tmp_path):
    rd = RunDir(tmp_path)
    rd.start(small_config(1))
    GeneticAlgorithm(small_config(1)).run(on_generation=rd.record)
    rd.truncate_to(1)
    assert [b["generation"] for b in rd.bests()] == [0, 1]
    assert {r["generation"] for r in rd.population_fitness()} == {"0", "1"}


def test_cli_train_then_resume(tmp_path, capsys):
    out = tmp_path / "run"
    small = ["--population", "10", "--genes", "100", "--max-time", "20", "--workers", "1", "--seed", "4"]
    cli(["train", "--generations", "1", "--out", str(out), *small])
    cli(["train", "--resume", str(out), "--generations", "3", "--workers", "1"])
    assert "resuming" in capsys.readouterr().out
    rd = RunDir(out)
    assert [b["generation"] for b in rd.bests()] == [0, 1, 2, 3]
    assert len(rd.population_fitness()) == 4 * 10
    assert rd.load_config().generations == 3
    straight = GeneticAlgorithm(replace(rd.load_config(), workers=1)).run()
    assert rd.load_checkpoint().history == straight.history
