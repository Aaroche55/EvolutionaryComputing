"""The interface the evolutionary algorithm uses.

    game = Game(func=controller, level=1)
    score = game.run_full()          # play to the end, return the progress score

or step it yourself:

    while (score := game.run_one()) is None:
        ...

`func(t)` is called once per decision step with the step index t = 0, 1, 2, ... and must
return an Action (or its int value, 0..N_ACTIONS-1). The action is held for
`ticks_per_action` physics ticks (default 24 -> 5 decisions per second), so decision step t
starts at t * game.decision_dt seconds.

If `func` raises IndexError (e.g. `lambda t: genome[t]` with a genome shorter than
game.max_steps), the game raises GenomeExhausted instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .actions import Action
from .constants import STEP
from .engine import World
from .level import load_level

Controller = Callable[[int], "Action | int"]

DEFAULT_TICKS_PER_ACTION = 24   # 120 Hz physics / 24 = 5 decisions per second


class GameError(Exception):
    """Base class for errors raised by the game."""


class GenomeExhausted(GameError):
    """The controller ran out of actions: it raised IndexError for decision step `step`."""

    def __init__(self, step: int, max_steps: int):
        super().__init__(f"controller has no action for step {step} (a full game needs {max_steps} steps)")
        self.step = step
        self.max_steps = max_steps


@dataclass(frozen=True)
class Result:
    progress: float        # 0..1, furthest distance towards the goal; 1.0 means the level was cleared
    cleared: bool
    died: bool
    death_cause: str | None  # "goose", "fall", "water" or None
    timed_out: bool
    time: float            # level timer in seconds
    ticks: int             # physics ticks simulated
    steps: int             # decisions made (calls to func)
    horseshoes: int
    horseshoes_total: int
    geese_stomped: int


class Game:
    def __init__(
        self,
        func: Controller,
        level: int | str | Path | dict = 1,
        *,
        ticks_per_action: int = DEFAULT_TICKS_PER_ACTION,
        max_time: float = 60.0,
        end_on_death: bool = True,
        render: bool = False,
        speed: float = 1.0,
    ):
        """
        func:             controller, called as func(t) -> Action.
        level:            level number, path to a level JSON file, or a parsed level dict.
        ticks_per_action: physics ticks (1/120 s each) an action is held; 24 -> 5 decisions per second.
        max_time:         game ends (timed out) after this many seconds of simulated time.
        end_on_death:     end the game on the first death. If False the player respawns at the
                          start (as in the reference game) and keeps going until max_time.
        render:           open a pygame window and draw the run (needs the `render` extra).
        speed:            playback speed multiplier when rendering.
        """
        if ticks_per_action < 1:
            raise ValueError("ticks_per_action must be >= 1")
        self.func = func
        self.ticks_per_action = ticks_per_action
        self.max_ticks = round(max_time / STEP)
        self.world = World(load_level(level), end_on_death=end_on_death)
        self.steps = 0
        self.last_action: Action | None = None
        self.renderer = None
        if render:
            from .render import Renderer
            self.renderer = Renderer(self.world, speed=speed)

    @property
    def decision_dt(self) -> float:
        """Seconds of game time each decision step lasts."""
        return self.ticks_per_action * STEP

    @property
    def max_steps(self) -> int:
        """Most decision steps a game can take before timing out (useful as a genome length)."""
        return -(-self.max_ticks // self.ticks_per_action)

    @property
    def finished(self) -> bool:
        return self.world.over or self.world.ticks >= self.max_ticks

    @property
    def score(self) -> float:
        return self.world.progress

    def run_one(self) -> float | None:
        """Make one decision and simulate it. Returns the progress score once the game is over, else None."""
        if self.finished:
            return self.score
        try:
            action = Action(self.func(self.steps))
        except IndexError as e:
            raise GenomeExhausted(self.steps, self.max_steps) from e
        self.last_action = action
        self.steps += 1
        w = self.world
        for _ in range(self.ticks_per_action):
            w.step(action)
            if self.renderer:
                self.renderer.tick(action)
            if self.finished:
                break
        if self.finished:
            if self.renderer:
                self.renderer.finish(self.result())
            return self.score
        return None

    def run_full(self) -> float:
        """Play until the level is cleared, the player dies (end_on_death) or time runs out."""
        while (score := self.run_one()) is None:
            pass
        return score

    def result(self) -> Result:
        w = self.world
        return Result(
            progress=w.progress,
            cleared=w.cleared,
            died=w.falls > 0,
            death_cause=w.death_cause,
            timed_out=not w.over and w.ticks >= self.max_ticks,
            time=w.time,
            ticks=w.ticks,
            steps=self.steps,
            horseshoes=w.shoes,
            horseshoes_total=len(w.coins),
            geese_stomped=w.stomps,
        )
