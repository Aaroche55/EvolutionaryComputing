"""Fitness of a genome: play it through the game and score the result. Lower is better (minimisation).

Three fitness functions, chosen per run with GAConfig.fitness:

"sum" (the original):
    fitness = (1 - progress) + min(time, max_time) / max_time
  progress is the game's 0..1 progress score (1.0 = level cleared) and time is the level timer in seconds,
  so both terms lie in [0, 1] and fitness lies in [0, 2]. A genome that clears the level scores
  time / max_time; one that never moves right and times out scores 2. Dying early is rewarded, since it
  keeps the time term small.

"tiered":
    cleared:      fitness = min(time, max_time) / max_time      in [0, 1]
    not cleared:  fitness = 2 - progress                        in [1, 2]
  Any clear beats any non-clear; non-clears are ranked by distance only, so dying early earns nothing.

"points":
    cleared:      fitness = -(horseshoes + GOOSE_POINTS * stomps)      <= 0
    not cleared:  fitness = 2 - progress                                in [1, 2]
  Time is ignored: among clears, only horseshoes collected and geese stomped count. Points only count on
  a clear, so any clear (even with nothing collected) beats any non-clear. The run must still finish
  within max_time.
"""

from __future__ import annotations

from dataclasses import dataclass

from gallop import Game, Result
from gallop.fast import simulate

from .genome import Genome, decode

DEFAULT_MAX_TIME = 200.0  # seconds; 1000 genes at 5 decisions per second
DEFAULT_TICKS_PER_ACTION = 24  # 120 Hz physics / 24 = 5 decisions per second


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    result: Result


FITNESS_FUNCTIONS = ("sum", "tiered", "points")

GOOSE_POINTS = 5  # points per goose stomped ("points" fitness); a horseshoe is 1. Level 1 has 38 shoes, 4 geese


def fitness(result: Result, max_time: float = DEFAULT_MAX_TIME, function: str = "sum") -> float:
    time_term = min(result.time, max_time) / max_time
    if function == "sum":
        return (1.0 - result.progress) + time_term
    if function == "tiered":
        return time_term if result.cleared else 2.0 - result.progress
    if function == "points":
        if not result.cleared:
            return 2.0 - result.progress
        return -float(result.horseshoes + GOOSE_POINTS * result.geese_stomped)
    raise ValueError(f"unknown fitness function {function!r}; choose from {FITNESS_FUNCTIONS}")


def evaluate(
    genome: Genome,
    max_time: float = DEFAULT_MAX_TIME,
    ticks_per_action: int = DEFAULT_TICKS_PER_ACTION,
    level: int = 1,
    fitness_function: str = "sum",
    engine: str = "numba",
) -> Evaluation:
    """Play one full game controlled by `genome` and return its fitness and the game result.

    engine "numba" uses the compiled gallop.fast.simulate; "python" plays through gallop.Game (the reference,
    about 150x slower). Both give identical results.
    """
    actions = decode(genome)
    if engine == "numba":
        result = simulate(actions, level, ticks_per_action=ticks_per_action, max_time=max_time)
    elif engine == "python":
        game = Game(func=actions.__getitem__, level=level, ticks_per_action=ticks_per_action, max_time=max_time)
        game.run_full()
        result = game.result()
    else:
        raise ValueError(f"unknown engine {engine!r}; choose 'numba' or 'python'")
    return Evaluation(fitness(result, max_time, fitness_function), result)
