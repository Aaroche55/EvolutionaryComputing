"""The batched PyTorch engine (gallop.gpu), run on the CPU in float64, must match the compiled engine exactly."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from evolution.genome import decode, random_genome
from gallop import GenomeExhausted
from gallop.fast import DEATH_CAUSES, simulate
from gallop.gpu import simulate_batch

from conftest import flat_level


def assert_matches(batch, i, r):
    assert batch.progress[i] == r.progress
    assert bool(batch.cleared[i]) == r.cleared
    assert DEATH_CAUSES[batch.death_cause[i]] == r.death_cause
    assert bool(batch.timed_out[i]) == r.timed_out
    assert batch.time[i] == r.time
    assert batch.ticks[i] == r.ticks
    assert batch.steps[i] == r.steps
    assert batch.horseshoes[i] == r.horseshoes
    assert batch.geese_stomped[i] == r.geese_stomped


def test_random_genomes_match_numba():
    rng = np.random.default_rng(0)
    actions = np.array([decode(random_genome(rng, 150)) for _ in range(40)])
    batch = simulate_batch(actions, max_time=30.0)
    for i, row in enumerate(actions):
        assert_matches(batch, i, simulate(row, max_time=30.0))


def test_solution_matches(solution):
    actions = np.array([solution + [0] * (300 - len(solution))])
    batch = simulate_batch(actions, max_time=60.0)
    assert batch.cleared[0]
    assert_matches(batch, 0, simulate(solution, max_time=60.0))


def test_compaction_does_not_change_results():
    rng = np.random.default_rng(1)
    actions = np.array([decode(random_genome(rng, 150)) for _ in range(30)])
    a = simulate_batch(actions, max_time=30.0, compact_every=1)
    b = simulate_batch(actions, max_time=30.0, compact_every=0)
    assert (a.time == b.time).all()
    assert (a.steps == b.steps).all()


@pytest.mark.parametrize(
    "level",
    [
        flat_level(geese=[{"x": 8, "top": 14}, {"x": 16, "top": 14}, {"x": 24, "top": 14}]),
        flat_level(
            extra_terrain=[
                {"type": "plank", "x0": 6, "x1": 9, "y": 11},
                {"type": "stone", "x0": 12, "x1": 13, "y0": 9, "y1": 10},
                {"type": "hay", "x": 18, "y": 13},
            ]
        ),
        {
            **flat_level(),
            "terrain": [
                {"type": "ground", "x0": 0, "x1": 8, "top": 14},
                {"type": "water", "x0": 9, "x1": 39, "top": 14},
            ],
        },
    ],
)
def test_custom_levels_match(level, tmp_path):
    path = tmp_path / "level.json"
    path.write_text(json.dumps(level))
    rng = np.random.default_rng(2)
    actions = rng.choice([2, 5, 5, 3, 0], size=(25, 50))
    batch = simulate_batch(actions, str(path), max_time=10.0)
    results = [simulate(row, level, max_time=10.0) for row in actions]
    for i, r in enumerate(results):
        assert_matches(batch, i, r)


def test_genome_exhausted():
    with pytest.raises(GenomeExhausted):
        simulate_batch(np.zeros((3, 10), dtype=np.int64), max_time=10.0)
