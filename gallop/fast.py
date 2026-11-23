"""A compiled (numba) copy of the engine, for training.

    result = simulate(actions, level=1, ticks_per_action=24, max_time=80.0)

plays a whole game from a list/array of action ids and returns the same Result as

    Game(func=actions.__getitem__, level=..., ticks_per_action=..., max_time=...).run_full()

but runs roughly 100x faster. engine.World stays the reference implementation (it is what the renderer
and the tests use); _play below is a line-by-line translation of World.step and Game.run_one with the
state kept in local variables and small arrays. Both use float64 and the same order of operations, so they
give bit-identical results; tests/test_fast.py checks this on the solution and on random action sequences.

Change engine.py and this file together.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit

from .actions import N_ACTIONS
from .constants import (
    ACC_AIR_IDLE,
    ACC_AIR_RUN,
    ACC_GROUND_IDLE,
    ACC_GROUND_RUN,
    ACC_GROUND_TURN,
    BUFFER,
    COYOTE,
    DEAD_T,
    FALLMAX,
    GOOSE_H,
    GOOSE_W,
    GRAV,
    GROUND,
    HAY,
    JUMP,
    PH,
    PLANK,
    PW,
    RUN,
    STEP,
    STOMP_BOUNCE,
    STOMP_BOUNCE_HELD,
    STOMP_FREEZE,
    STONE,
    WATER,
    T,
)
from .game import DEFAULT_TICKS_PER_ACTION, GenomeExhausted, Result
from .level import Level, load_level
from .schema import LevelSpec

DEATH_CAUSES = (None, "goose", "fall", "water")
_GOOSE, _FALL, _WATER = 1, 2, 3

# Per action id: direction and whether jump is held (see actions.Action).
_DIRECTION = np.array([0, -1, 1, 0, -1, 1], dtype=np.int64)
_JUMP = np.array([False, False, False, True, True, True])
assert len(_DIRECTION) == N_ACTIONS


@njit(cache=True)
def _tile(grid, x, y):
    rows, width = grid.shape
    if x < 0 or x >= width:
        return STONE
    if y < 0 or y >= rows:
        return 0
    return grid[y, x]


@njit(cache=True)
def _solid(grid, x, y):
    t = _tile(grid, x, y)
    return t in (GROUND, STONE, HAY)


@njit(cache=True)
def _approach(v, target, delta):
    return min(v + delta, target) if v < target else max(v - delta, target)


@njit(cache=True)
def _floor(v):
    return math.floor(v)  # an int in numba, as in Python


@njit(cache=True)
def _hits_ceiling(grid, px, w, ty):
    for x in range(_floor(px / T), _floor((px + w - 0.001) / T) + 1):  # noqa: SIM110 (numba: no any() over generators)
        if _solid(grid, x, ty):
            return True
    return False


@njit(cache=True)
def _play(
    actions,
    grid,
    coins,
    geese_init,
    spawn_x,
    spawn_y,
    goal_x,
    ticks_per_action,
    max_ticks,
    end_on_death,
    direction_of,
    jump_of,
):
    """Returns (progress, time) and int64 [cleared, falls, death_cause, timed_out, ticks, steps, horseshoes,
    stomps]. steps = -1 means the action sequence ran out (GenomeExhausted)."""
    rows = grid.shape[0]
    w = float(PW)
    h = float(PH)
    # player
    x = spawn_x
    y = spawn_y
    vx = 0.0
    vy = 0.0
    on_ground = False
    coyote = 0.0
    buffer = 0.0
    dead = 0.0
    # world
    n_geese = geese_init.shape[0]
    gx = geese_init[:, 0].copy()
    gy = geese_init[:, 1].copy()
    gvx = geese_init[:, 2].copy()
    gdead = np.zeros(n_geese, dtype=np.bool_)
    n_coins = coins.shape[0]
    got = np.zeros(n_coins, dtype=np.bool_)
    ticks = 0
    time = 0.0
    freeze = 0.0
    falls = 0
    stomps = 0
    cleared = False
    over = False
    death_cause = 0
    start_x = x + w
    best_x = start_x
    prev_held = False
    pending_jump = False
    steps = 0

    while not (over or ticks >= max_ticks):
        if steps >= actions.shape[0]:
            steps = -1
            break
        action = actions[steps]
        steps += 1
        direction = direction_of[action]
        held = jump_of[action]
        for _ in range(ticks_per_action):
            # ---------- World.step
            ticks += 1
            if held and not prev_held:
                pending_jump = True
            prev_held = held

            if freeze > 0:
                freeze -= STEP
            else:
                if dead > 0:
                    dead -= STEP
                    if dead <= 0:
                        # _respawn
                        x = spawn_x
                        y = spawn_y
                        vx = 0.0
                        vy = 0.0
                        buffer = 0.0
                        on_ground = False
                        dead = 0.0
                else:
                    # ---------- _update_player
                    dt = STEP
                    if on_ground:
                        if direction == 0:
                            acc = ACC_GROUND_IDLE
                        elif vx * direction < 0:
                            acc = ACC_GROUND_TURN
                        else:
                            acc = ACC_GROUND_RUN
                    else:
                        acc = ACC_AIR_IDLE if direction == 0 else ACC_AIR_RUN
                    vx = _approach(vx, direction * RUN, acc * dt)

                    coyote = COYOTE if on_ground else coyote - dt
                    buffer = BUFFER if pending_jump else buffer - dt
                    if buffer > 0 and coyote > 0:
                        vy = -float(JUMP)
                        buffer = 0.0
                        coyote = 0.0
                        on_ground = False

                    g = GRAV
                    if vy < 0 and not held:
                        g *= 2.5
                    elif abs(vy) < 80 and held:
                        g *= 0.5
                    elif vy > 0:
                        g *= 1.25
                    vy = min(vy + g * dt, FALLMAX)

                    # _move_x
                    dx = vx * dt
                    x += dx
                    y0 = _floor(y / T)
                    y1 = _floor((y + h - 0.001) / T)
                    if dx > 0:
                        tx = _floor((x + w) / T)
                        for yy in range(y0, y1 + 1):
                            if _solid(grid, tx, yy):
                                x = float(tx * T) - w
                                vx = min(vx, 0.0)
                                break
                    elif dx < 0:
                        tx = _floor(x / T)
                        for yy in range(y0, y1 + 1):
                            if _solid(grid, tx, yy):
                                x = float((tx + 1) * T)
                                vx = max(vx, 0.0)
                                break

                    # _move_y
                    dy = vy * dt
                    prev_bot = y + h
                    y += dy
                    on_ground = False
                    x0 = _floor(x / T)
                    x1 = _floor((x + w - 0.001) / T)
                    if dy > 0:
                        ty = _floor((y + h) / T)
                        for xx in range(x0, x1 + 1):
                            t = _tile(grid, xx, ty)
                            if t in (GROUND, STONE, HAY) or (t == PLANK and prev_bot <= ty * T + 0.5):
                                y = float(ty * T) - h
                                vy = 0.0
                                on_ground = True
                                break
                    elif dy < 0:
                        ty = _floor(y / T)
                        if _hits_ceiling(grid, x, w, ty):
                            corrected = False
                            for n in range(1, 7):
                                if not _hits_ceiling(grid, x + n, w, ty):
                                    x += n
                                    corrected = True
                                    break
                                if not _hits_ceiling(grid, x - n, w, ty):
                                    x -= n
                                    corrected = True
                                    break
                            if not corrected:
                                y = float((ty + 1) * T)
                                vy = 0.0

                    pending_jump = False
                    time += STEP

                    # ---------- _interact
                    pcx = x + w / 2
                    pcy = y + h / 2
                    for i in range(n_coins):
                        if not got[i] and abs(coins[i, 0] - pcx) < 14 and abs(coins[i, 1] - pcy) < 17:
                            got[i] = True
                    killed = 0
                    for i in range(n_geese):
                        if gdead[i]:
                            continue
                        if x + w > gx[i] + 2 and x < gx[i] + GOOSE_W - 2 and y + h > gy[i] + 3 and y < gy[i] + GOOSE_H:
                            if vy > 0 and y + h - gy[i] < 11:
                                gdead[i] = True
                                stomps += 1
                                vy = -float(STOMP_BOUNCE_HELD if held else STOMP_BOUNCE)
                                coyote = 0.0
                                freeze = STOMP_FREEZE
                            else:
                                killed = _GOOSE
                                break
                    if killed == 0:
                        if _tile(grid, _floor(pcx / T), _floor((y + h - 4) / T)) == WATER:
                            killed = _WATER
                        elif y > rows * T + 40:
                            killed = _FALL
                        elif x + w > goal_x:
                            cleared = True
                            over = True
                    if killed != 0 and not dead > 0:
                        # _kill
                        dead = DEAD_T
                        falls += 1
                        death_cause = killed
                        if end_on_death:
                            over = True

                    best_x = max(best_x, x + w)

                if not over:
                    # ---------- _update_geese
                    for i in range(n_geese):
                        if gdead[i]:
                            continue
                        nx = gx[i] + gvx[i] * STEP
                        fx = nx + GOOSE_W if gvx[i] > 0 else nx
                        tx = _floor(fx / T)
                        my = _floor((gy[i] + GOOSE_H / 2) / T)
                        fy = _floor((gy[i] + GOOSE_H + 1) / T)
                        ft = _tile(grid, tx, fy)
                        standable = ft in (GROUND, STONE, HAY, PLANK)
                        if _solid(grid, tx, my) or not standable:
                            gvx[i] = -gvx[i]
                        else:
                            gx[i] = nx

            if over or ticks >= max_ticks:
                break

    if cleared:
        progress = 1.0
    else:
        span = goal_x - start_x
        progress = min(1.0, max(0.0, (best_x - start_x) / span))
    timed_out = (not over) and ticks >= max_ticks
    counts = np.array([int(cleared), falls, death_cause, int(timed_out), ticks, steps, int(got.sum()), stomps])
    return progress, time, counts


@cache
def _level_arrays(level: int | str | Path) -> tuple:
    return _arrays(load_level(level))


def _arrays(lv: Level) -> tuple:
    grid = np.array(lv.grid, dtype=np.int64)
    coins = np.array(lv.coins, dtype=np.float64).reshape(-1, 2)
    geese = np.array(lv.geese, dtype=np.float64).reshape(-1, 3)
    return grid, coins, geese, float(lv.spawn[0]), float(lv.spawn[1]), float(lv.goal_x)


def simulate(
    actions: npt.ArrayLike,
    level: int | str | Path | dict[str, Any] | LevelSpec | Level = 1,
    *,
    ticks_per_action: int = DEFAULT_TICKS_PER_ACTION,
    max_time: float = 60.0,
    end_on_death: bool = True,
) -> Result:
    """Play a whole game from a sequence of action ids. Same result as Game(...).run_full().

    Raises GenomeExhausted if the game needs more actions than given.
    """
    if ticks_per_action < 1:
        raise ValueError("ticks_per_action must be >= 1")
    acts = np.asarray(actions, dtype=np.int64)
    if acts.size and (acts.min() < 0 or acts.max() >= N_ACTIONS):
        raise ValueError(f"action ids must be in 0..{N_ACTIONS - 1}")
    if isinstance(level, int | str | Path):
        level_data = _level_arrays(level)  # level files are parsed once per process
    else:
        level_data = _arrays(level if isinstance(level, Level) else load_level(level))
    grid, coins, geese, spawn_x, spawn_y, goal_x = level_data
    max_ticks = round(max_time / STEP)
    progress, time, counts = _play(
        acts,
        grid,
        coins,
        geese,
        spawn_x,
        spawn_y,
        goal_x,
        ticks_per_action,
        max_ticks,
        end_on_death,
        _DIRECTION,
        _JUMP,
    )
    cleared, falls, cause, timed_out, ticks, steps, horseshoes, stomps = (int(c) for c in counts)
    if steps < 0:
        max_steps = -(-max_ticks // ticks_per_action)
        raise GenomeExhausted(len(acts), max_steps)
    return Result(
        progress=float(progress),
        cleared=bool(cleared),
        died=falls > 0,
        death_cause=DEATH_CAUSES[cause],
        timed_out=bool(timed_out),
        time=float(time),
        ticks=ticks,
        steps=steps,
        horseshoes=horseshoes,
        horseshoes_total=len(coins),
        geese_stomped=stomps,
    )
