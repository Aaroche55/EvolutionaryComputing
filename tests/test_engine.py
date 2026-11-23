import pytest

from gallop.actions import Action
from gallop.constants import PH, RUN, T

from conftest import flat_level, make_world


def run(world, action, ticks):
    for _ in range(ticks):
        world.step(action)


def settle(world):
    run(world, Action.NOOP, 30)
    assert world.player.on_ground


def test_player_rests_on_ground():
    w = make_world(flat_level())
    settle(w)
    assert w.player.y == 14 * T - PH
    assert w.player.vy == 0


def test_running_reaches_top_speed():
    w = make_world(flat_level())
    run(w, Action.RIGHT, 60)
    assert w.player.vx == pytest.approx(RUN)
    assert w.player.face == 1
    run(w, Action.LEFT, 60)
    assert w.player.vx == pytest.approx(-RUN)
    assert w.player.face == -1


def test_full_jump_clears_about_four_tiles():
    # Reference comment: "The player clears about 4 tiles up".
    w = make_world(flat_level())
    settle(w)
    ground_y = w.player.y
    peak = ground_y
    w.step(Action.JUMP)
    assert w.player.vy < 0
    for _ in range(120):
        w.step(Action.JUMP)
        peak = min(peak, w.player.y)
    rise = ground_y - peak
    assert 3.5 * T < rise < 5 * T
    assert w.player.on_ground


def test_short_hop_is_lower_than_full_jump():
    def jump_height(hold_ticks):
        w = make_world(flat_level())
        settle(w)
        y0, peak = w.player.y, w.player.y
        for i in range(120):
            w.step(Action.JUMP if i < hold_ticks else Action.NOOP)
            peak = min(peak, w.player.y)
        return y0 - peak

    assert jump_height(5) < jump_height(120) * 0.6


def test_holding_jump_only_jumps_once():
    w = make_world(flat_level())
    settle(w)
    jumps = 0
    was_ground = True
    for _ in range(400):
        w.step(Action.JUMP)
        if was_ground and not w.player.on_ground:
            jumps += 1
        was_ground = w.player.on_ground
    assert jumps == 1


def test_jump_buffer_fires_on_landing():
    w = make_world(flat_level())
    settle(w)
    ground_y = w.player.y
    run(w, Action.JUMP, 1)
    while not (w.player.vy > 0 and w.player.y > ground_y - 10):
        w.step(Action.NOOP)  # release, then fall to ~10px above the ground
    assert not w.player.on_ground
    jumped_again = False
    for _ in range(20):
        w.step(Action.JUMP)  # pressed in the air, before landing
        jumped_again |= w.player.vy < 0
    assert jumped_again


def test_wall_blocks_running():
    w = make_world(flat_level(extra_terrain=[{"type": "stone", "x0": 10, "x1": 10, "y0": 10}]))
    run(w, Action.RIGHT, 240)
    assert w.player.x + w.player.w == 10 * T
    assert w.player.vx == 0


def test_plank_is_one_way():
    w = make_world(flat_level(extra_terrain=[{"type": "plank", "x0": 0, "x1": 10, "y": 12}]))
    # Spawn top is the plank (it is the highest standable tile), so the player stands on it.
    settle(w)
    assert w.player.y + w.player.h == 12 * T
    # Jumping from below passes through: put the player on the ground under it.
    w = make_world(flat_level(extra_terrain=[{"type": "plank", "x0": 4, "x1": 10, "y": 11}]))
    settle(w)
    run(w, Action.RIGHT, 30)
    assert w.player.y + w.player.h == 14 * T
    w.step(Action.JUMP)
    peak = w.player.y
    for _ in range(40):
        w.step(Action.JUMP)
        peak = min(peak, w.player.y)
    assert peak + w.player.h < 11 * T  # rose through the plank
    run(w, Action.NOOP, 120)
    assert w.player.y + w.player.h == 11 * T  # and landed on top of it


def test_falling_in_pit_kills():
    data = flat_level()
    data["terrain"] = [{"type": "ground", "x0": 0, "x1": 5, "top": 14}]
    w = make_world(data)
    run(w, Action.RIGHT, 600)
    assert w.over
    assert not w.cleared
    assert w.death_cause == "fall"


def test_water_kills():
    data = flat_level()
    data["terrain"] = [{"type": "ground", "x0": 0, "x1": 5, "top": 14}, {"type": "water", "x0": 6, "x1": 39, "top": 14}]
    w = make_world(data)
    run(w, Action.RIGHT, 600)
    assert w.death_cause == "water"


def test_walking_into_goose_kills():
    w = make_world(flat_level(geese=[{"x": 8, "top": 14}]))
    run(w, Action.RIGHT, 600)
    assert w.over
    assert w.death_cause == "goose"
    assert w.stomps == 0


def test_stomping_goose_bounces_and_kills_it():
    w = make_world(flat_level(geese=[{"x": 6, "top": 14, "speed": 0.001}]))
    settle(w)
    # Jump straight up next to the goose, then drift right onto it.
    w.step(Action.JUMP)
    bounced = False
    for _ in range(200):
        w.step(Action.RIGHT_JUMP if w.player.vy < 0 else Action.RIGHT)
        if w.stomps:
            bounced = w.player.vy < 0
            break
    assert w.stomps == 1
    assert w.geese[0].dead
    assert bounced
    assert not w.over


def test_geese_turn_at_ledges():
    data = flat_level(geese=[{"x": 12, "top": 14, "speed": 40}])
    data["terrain"] = [
        {"type": "ground", "x0": 0, "x1": 1, "top": 14},
        {"type": "ground", "x0": 10, "x1": 15, "top": 14},
    ]
    w = make_world(data)
    xs = []
    for _ in range(1200):
        w._update_geese(1 / 120)
        xs.append(w.geese[0].x)
    assert min(xs) >= 10 * T - 1
    assert max(xs) + 16 <= 16 * T + 1


def test_horseshoe_collected():
    w = make_world(flat_level(horseshoes=[{"x0": 6, "x1": 8, "y": 13}]))
    run(w, Action.RIGHT, 120)
    assert w.shoes == 3


def test_reaching_goal_clears():
    w = make_world(flat_level(width=40, goal=10))
    run(w, Action.RIGHT, 600)
    assert w.cleared
    assert w.over
    assert w.progress == 1.0


def test_respawn_when_not_ending_on_death():
    w = make_world(flat_level(geese=[{"x": 8, "top": 14}]), end_on_death=False)
    respawned = False
    for _ in range(600):
        x = w.player.x
        w.step(Action.RIGHT)
        respawned |= w.player.x < x - T
    assert w.falls >= 1
    assert respawned
    assert not w.over


def test_progress_is_monotonic_best_distance():
    w = make_world(flat_level(width=40, goal=30))
    run(w, Action.RIGHT, 60)
    run(w, Action.LEFT, 30)  # decelerates, still drifting right
    p = w.progress
    run(w, Action.LEFT, 120)  # now running back towards the start
    assert w.progress == p
    assert 0 < p < 1
