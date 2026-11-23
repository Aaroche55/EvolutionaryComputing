"""The compiled engine (gallop.fast) must give exactly the same results as the reference (Game/World)."""

import numpy as np
import pytest

from evolution import evaluate
from evolution.genome import decode, random_genome
from gallop import Game, GenomeExhausted
from gallop.fast import simulate

from conftest import flat_level


def reference(actions, **kw):
    game = Game(func=list(actions).__getitem__, **kw)
    game.run_full()
    return game.result()


def test_solution_matches(solution):
    assert simulate(solution, max_time=60.0) == reference(solution, max_time=60.0)


def test_random_genomes_match():
    rng = np.random.default_rng(0)
    results = []
    for _ in range(60):
        actions = decode(random_genome(rng, 400))
        r = simulate(actions, max_time=80.0)
        assert r == reference(actions, max_time=80.0)
        results.append(r)
    # The sample covers falling and timing out; test_custom_levels_match covers geese and water.
    assert any(r.death_cause == "fall" for r in results)
    assert any(r.timed_out for r in results)


def test_respawn_and_other_decision_rates_match():
    rng = np.random.default_rng(1)
    for tpa in (1, 7, 24):
        actions = decode(random_genome(rng, 3000))
        fast = simulate(actions, max_time=25.0, end_on_death=False, ticks_per_action=tpa)
        assert fast == reference(actions, max_time=25.0, end_on_death=False, ticks_per_action=tpa)


@pytest.mark.parametrize(
    "level",
    [
        # water
        {
            **flat_level(),
            "terrain": [
                {"type": "ground", "x0": 0, "x1": 8, "top": 14},
                {"type": "water", "x0": 9, "x1": 39, "top": 14},
            ],
        },
        # planks, a stone ceiling (corner correction) and hay
        flat_level(
            extra_terrain=[
                {"type": "plank", "x0": 6, "x1": 9, "y": 11},
                {"type": "stone", "x0": 12, "x1": 13, "y0": 9, "y1": 10},
                {"type": "hay", "x": 18, "y": 13},
            ]
        ),
        # geese to stomp or be killed by
        flat_level(geese=[{"x": 8, "top": 14}, {"x": 16, "top": 14}, {"x": 24, "top": 14}]),
    ],
)
def test_custom_levels_match(level):
    rng = np.random.default_rng(2)
    for _ in range(25):
        actions = rng.choice([2, 5, 5, 3, 0], size=100).tolist()  # mostly right and jump
        assert simulate(actions, level, max_time=20.0) == reference(actions, level=level, max_time=20.0)


def test_genome_exhausted():
    with pytest.raises(GenomeExhausted) as exc:
        simulate([0] * 10, max_time=10.0)
    assert exc.value.step == 10
    assert exc.value.max_steps == 50


def test_rejects_unknown_actions():
    with pytest.raises(ValueError, match="action ids"):
        simulate([0, 6], max_time=1.0)


def test_evaluate_engines_agree():
    rng = np.random.default_rng(3)
    for _ in range(10):
        g = random_genome(rng, 400)
        assert evaluate(g, max_time=80.0) == evaluate(g, max_time=80.0, engine="python")
