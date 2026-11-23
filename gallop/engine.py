"""Deterministic, headless simulation of one level.

A port of the play-state logic of the reference game (updatePlayer, moveX/moveY, updateGeese,
interact, stepPlay). Rendering, sound, particles, camera and checkpoints are left out: none of
them affect the outcome. The reference's only randomness (goose honk timers, particles) is
cosmetic, so the same inputs always produce the same run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .actions import Action
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
    JUMP,
    PH,
    PLANK,
    PW,
    RUN,
    SOLID_TILES,
    STEP,
    STOMP_BOUNCE,
    STOMP_BOUNCE_HELD,
    STOMP_FREEZE,
    WATER,
    T,
)
from .level import Level

floor = math.floor


def approach(v: float, target: float, delta: float) -> float:
    return min(v + delta, target) if v < target else max(v - delta, target)


@dataclass
class Player:
    x: float
    y: float
    w: float = PW
    h: float = PH
    vx: float = 0.0
    vy: float = 0.0
    face: int = 1
    on_ground: bool = False
    coyote: float = 0.0
    buffer: float = 0.0
    dead: float = 0.0  # > 0 while the death animation plays


@dataclass
class Goose:
    x: float
    y: float
    vx: float
    w: float = GOOSE_W
    h: float = GOOSE_H
    dead: bool = False


@dataclass
class Coin:
    x: float
    y: float
    got: bool = False


class World:
    """The state of one level being played. Advance it with step(), one 1/120 s tick at a time."""

    def __init__(self, level: Level, end_on_death: bool = True):
        self.level = level
        self.end_on_death = end_on_death
        self.player = Player(*level.spawn)
        self.geese = [Goose(x, y, vx) for x, y, vx in level.geese]
        self.coins = [Coin(x, y) for x, y in level.coins]
        self.ticks = 0  # physics steps taken
        self.time = 0.0  # level timer as the reference shows it (paused during hit-stop and death)
        self.freeze = 0.0
        self.falls = 0
        self.stomps = 0
        self.cleared = False
        self.over = False  # True once cleared, or dead with end_on_death
        self.death_cause: str | None = None
        self.start_x = self.player.x + self.player.w
        self.best_x = self.start_x
        self._held = False
        self._pending_jump = False

    # ---------- queries

    def solid_at(self, x: int, y: int) -> bool:
        return self.level.tile(x, y) in SOLID_TILES

    @property
    def shoes(self) -> int:
        return sum(c.got for c in self.coins)

    @property
    def progress(self) -> float:
        """Fraction of the way from spawn to the goal, using the furthest point reached. 1.0 when cleared."""
        if self.cleared:
            return 1.0
        span = self.level.goal_x - self.start_x
        return min(1.0, max(0.0, (self.best_x - self.start_x) / span))

    # ---------- tick

    def step(self, action: Action) -> None:
        """Advance one physics tick with `action` held."""
        if self.over:
            return
        self.ticks += 1
        held = action.jump
        if held and not self._held:
            self._pending_jump = True
        self._held = held

        if self.freeze > 0:
            self.freeze -= STEP
            return
        p = self.player
        if p.dead > 0:
            p.dead -= STEP
            if p.dead <= 0:
                self._respawn()
        else:
            self._update_player(STEP, action.direction, self._pending_jump, held)
            self._pending_jump = False
            self.time += STEP
            self._interact(held)
            self.best_x = max(self.best_x, p.x + p.w)
        if self.over:
            return
        self._update_geese(STEP)

    # ---------- player

    def _update_player(self, dt: float, direction: int, jump_pressed: bool, jump_held: bool) -> None:
        p = self.player
        if p.on_ground:
            acc = ACC_GROUND_IDLE if direction == 0 else (ACC_GROUND_TURN if p.vx * direction < 0 else ACC_GROUND_RUN)
        else:
            acc = ACC_AIR_IDLE if direction == 0 else ACC_AIR_RUN
        p.vx = approach(p.vx, direction * RUN, acc * dt)
        if direction:
            p.face = direction

        p.coyote = COYOTE if p.on_ground else p.coyote - dt
        p.buffer = BUFFER if jump_pressed else p.buffer - dt
        if p.buffer > 0 and p.coyote > 0:
            p.vy = -JUMP
            p.buffer = p.coyote = 0.0
            p.on_ground = False

        g = GRAV
        if p.vy < 0 and not jump_held:
            g *= 2.5  # short hop when jump is released early
        elif abs(p.vy) < 80 and jump_held:
            g *= 0.5  # floaty apex
        elif p.vy > 0:
            g *= 1.25  # snappy fall
        p.vy = min(p.vy + g * dt, FALLMAX)

        self._move_x(p.vx * dt)
        self._move_y(p.vy * dt)

    def _move_x(self, dx: float) -> None:
        p = self.player
        p.x += dx
        y0, y1 = floor(p.y / T), floor((p.y + p.h - 0.001) / T)
        if dx > 0:
            tx = floor((p.x + p.w) / T)
            for y in range(y0, y1 + 1):
                if self.solid_at(tx, y):
                    p.x = tx * T - p.w
                    p.vx = min(p.vx, 0)
                    break
        elif dx < 0:
            tx = floor(p.x / T)
            for y in range(y0, y1 + 1):
                if self.solid_at(tx, y):
                    p.x = (tx + 1) * T
                    p.vx = max(p.vx, 0)
                    break

    def _move_y(self, dy: float) -> None:
        p = self.player
        prev_bot = p.y + p.h
        p.y += dy
        p.on_ground = False
        x0, x1 = floor(p.x / T), floor((p.x + p.w - 0.001) / T)
        if dy > 0:
            ty = floor((p.y + p.h) / T)
            for x in range(x0, x1 + 1):
                t = self.level.tile(x, ty)
                # Planks only catch you from above.
                if t in SOLID_TILES or (t == PLANK and prev_bot <= ty * T + 0.5):
                    p.y = ty * T - p.h
                    p.vy = 0.0
                    p.on_ground = True
                    return
        elif dy < 0:
            ty = floor(p.y / T)

            def hits(px: float) -> bool:
                return any(self.solid_at(x, ty) for x in range(floor(px / T), floor((px + p.w - 0.001) / T) + 1))

            if hits(p.x):
                # Corner correction: slide round a ledge you only clipped.
                for n in range(1, 7):
                    if not hits(p.x + n):
                        p.x += n
                        return
                    if not hits(p.x - n):
                        p.x -= n
                        return
                p.y = (ty + 1) * T
                p.vy = 0.0

    # ---------- world

    def _update_geese(self, dt: float) -> None:
        lv = self.level
        for q in self.geese:
            if q.dead:
                continue
            nx = q.x + q.vx * dt
            fx = nx + q.w if q.vx > 0 else nx
            tx, my, fy = floor(fx / T), floor((q.y + q.h / 2) / T), floor((q.y + q.h + 1) / T)
            ft = lv.tile(tx, fy)
            # Turn round at walls and ledges.
            if self.solid_at(tx, my) or not (ft in SOLID_TILES or ft == PLANK):
                q.vx = -q.vx
            else:
                q.x = nx

    def _interact(self, jump_held: bool) -> None:
        p = self.player
        pcx, pcy = p.x + p.w / 2, p.y + p.h / 2
        for c in self.coins:
            if not c.got and abs(c.x - pcx) < 14 and abs(c.y - pcy) < 17:
                c.got = True
        for q in self.geese:
            if q.dead:
                continue
            if p.x + p.w > q.x + 2 and p.x < q.x + q.w - 2 and p.y + p.h > q.y + 3 and p.y < q.y + q.h:
                if p.vy > 0 and p.y + p.h - q.y < 11:
                    q.dead = True
                    self.stomps += 1
                    p.vy = -(STOMP_BOUNCE_HELD if jump_held else STOMP_BOUNCE)
                    p.coyote = 0.0
                    self.freeze = STOMP_FREEZE
                else:
                    self._kill("goose")
                    return
        if self.level.tile(floor(pcx / T), floor((p.y + p.h - 4) / T)) == WATER:
            self._kill("water")
            return
        if p.y > self.level.rows * T + 40:
            self._kill("fall")
            return
        if p.x + p.w > self.level.goal_x:
            self.cleared = True
            self.over = True

    def _kill(self, cause: str) -> None:
        p = self.player
        if p.dead > 0:
            return
        p.dead = DEAD_T
        self.falls += 1
        self.death_cause = cause
        if self.end_on_death:
            self.over = True

    def _respawn(self) -> None:
        # No checkpoints: always back to the level start.
        p = self.player
        p.x, p.y = self.level.spawn
        p.vx = p.vy = p.buffer = 0.0
        p.on_ground = False
        p.dead = 0.0
