import random

import pytest

from gallop import N_ACTIONS, Action, Game, GameError, GenomeExhausted
from gallop.__main__ import sequence_controller

from conftest import flat_level


def random_controller(seed, n=2000):
    rng = random.Random(seed)
    genome = [rng.randrange(N_ACTIONS) for _ in range(n)]
    return lambda t: genome[t]


def test_actions_are_indexable_ints():
    assert [int(a) for a in Action] == list(range(N_ACTIONS))
    assert Action(5) is Action.RIGHT_JUMP
    assert Action.RIGHT_JUMP.direction == 1 and Action.RIGHT_JUMP.jump
    assert Action.LEFT.direction == -1 and not Action.LEFT.jump


def test_func_receives_increasing_step_index():
    seen = []
    game = Game(lambda t: seen.append(t) or Action.RIGHT, max_time=1.0)
    game.run_full()
    assert seen == list(range(len(seen)))
    assert len(seen) == game.max_steps == 5     # default: 5 decisions per second


def test_func_may_return_plain_int():
    assert Game(lambda t: 2, max_time=1.0).run_full() > 0


def test_default_rate_is_five_actions_per_second():
    game = Game(lambda t: Action.NOOP)
    assert game.decision_dt == pytest.approx(0.2)
    assert game.max_steps == 300                  # 60 s default


def test_short_genome_raises_genome_exhausted():
    genome = [Action.RIGHT] * 10
    game = Game(lambda t: genome[t])
    with pytest.raises(GenomeExhausted) as exc:
        game.run_full()
    assert exc.value.step == 10
    assert exc.value.max_steps == game.max_steps
    assert isinstance(exc.value, GameError)
    assert isinstance(exc.value.__cause__, IndexError)
    assert game.steps == 10                       # the 10 genes that existed were all played


def test_genome_exhausted_from_run_one():
    game = Game(lambda t: [Action.RIGHT][t])
    assert game.run_one() is None
    with pytest.raises(GenomeExhausted):
        game.run_one()


def test_full_length_genome_never_exhausts():
    game = Game(lambda t: Action.NOOP)
    genome = [Action.NOOP] * game.max_steps
    Game(lambda t: genome[t]).run_full()          # times out exactly at the end of the genome


def test_genome_exhausted_not_raised_if_game_ends_first(solution):
    # The solution is shorter than max_steps but clears the level before running out.
    assert len(solution) < Game(lambda t: 0).max_steps
    assert Game(lambda t: solution[t]).run_full() == 1.0


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        Game(lambda t: 99).run_one()


def test_run_one_returns_none_until_finished():
    game = Game(lambda t: Action.NOOP, max_time=0.5)
    results = []
    while (r := game.run_one()) is None:
        results.append(r)
    assert len(results) == game.max_steps - 1
    assert r == game.score
    assert game.result().timed_out
    assert game.run_one() == r        # calling again after the end is harmless


def test_run_full_matches_run_one_loop():
    a = Game(random_controller(1))
    b = Game(random_controller(1))
    score_a = a.run_full()
    while (score_b := b.run_one()) is None:
        pass
    assert score_a == score_b
    assert a.result() == b.result()


@pytest.mark.parametrize("seed", range(5))
def test_deterministic(seed):
    r1 = Game(random_controller(seed))
    r2 = Game(random_controller(seed))
    assert r1.run_full() == r2.run_full()
    assert r1.result() == r2.result()


def test_score_in_unit_interval():
    for seed in range(20):
        s = Game(random_controller(seed)).run_full()
        assert 0.0 <= s <= 1.0


def test_hold_right_gets_stuck_at_first_hay_bale():
    game = Game(lambda t: Action.RIGHT)
    score = game.run_full()
    assert 0 < score < 0.1
    assert game.result().timed_out


def test_recorded_solution_clears_level1(solution):
    game = Game(sequence_controller(solution))
    assert game.run_full() == 1.0
    r = game.result()
    assert r.cleared and not r.died
    assert 15 < r.time < 30
    assert r.steps == len(solution)


def test_end_on_death_false_keeps_playing():
    level = flat_level()
    level["terrain"] = [{"type": "ground", "x0": 0, "x1": 5, "top": 14}]   # a pit you always fall into
    game = Game(lambda t: Action.RIGHT, level, end_on_death=False, max_time=10)
    game.run_full()
    r = game.result()
    assert r.died and r.timed_out and not r.cleared
    assert game.world.falls > 1


def test_end_on_death_true_stops_at_first_death():
    game = Game(lambda t: Action.RIGHT, flat_level(geese=[{"x": 8, "top": 14}]))
    game.run_full()
    r = game.result()
    assert r.died and r.death_cause == "goose" and not r.timed_out
    assert r.time < 2


def test_ticks_per_action_changes_decision_rate():
    g1 = Game(lambda t: Action.NOOP, ticks_per_action=1, max_time=1.0)
    g12 = Game(lambda t: Action.NOOP, ticks_per_action=12, max_time=1.0)
    g1.run_full(), g12.run_full()
    assert g1.steps == 120 and g12.steps == 10
    assert g1.decision_dt == pytest.approx(1 / 120)
    with pytest.raises(ValueError):
        Game(lambda t: 0, ticks_per_action=0)
