# Ballinasloe Gallop — EA environment

A headless, deterministic Python port of level 1 ("The Fair Green") of `og_game/ballinasloe-gallop.html`,
built as the environment for an evolutionary algorithm.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.13 (`.python-version`) because pygame-ce
has no wheels for 3.15 yet. The headless game only depends on pydantic (level file validation).

```sh
uv sync --extra render     # pygame-ce for the viewer, plus dev tools (pytest, ruff, ty)
uv run pytest              # run the tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # type check
```

## Controller API

```python
from gallop import Game, Action, N_ACTIONS

def controller(t: int) -> Action | int:   # t = decision step 0, 1, 2, ...
    return genome[t]                       # any int in 0..N_ACTIONS-1

game = Game(func=controller, level=1)
score = game.run_full()                    # progress score in [0, 1]; 1.0 = level cleared

# or one decision at a time: None until the game is over, then the score
game = Game(func=controller, level=1)
while (score := game.run_one()) is None:
    pass

game.result()   # Result(progress, cleared, died, death_cause, timed_out, time, ticks, steps,
                #        horseshoes, horseshoes_total, geese_stomped)
```

| Action | value | |
|---|---|---|
| `NOOP` | 0 | |
| `LEFT` | 1 | |
| `RIGHT` | 2 | |
| `JUMP` | 3 | jump held |
| `LEFT_JUMP` | 4 | |
| `RIGHT_JUMP` | 5 | |

Jump behaves like the button in the original: a jump starts when jump goes from released to held, and
holding it longer jumps higher. Holding jump for consecutive steps does **not** jump again on landing.

`Game` options:

- `ticks_per_action=24`: physics runs at 120 Hz; each decision is held for this many ticks (24 gives
  5 decisions/s, i.e. one every 0.2 s). `game.decision_dt` is the length of one decision in seconds.
- `max_time=60.0`: simulated seconds before the run times out. `game.max_steps` is the most decisions a
  run can take (300 with the defaults), which is a natural genome length.
- `end_on_death=True`: the run ends on the first death. With `False` the player respawns at the start
  (no checkpoints) until time runs out.
- `render=False`: `True` opens a pygame window and plays the run in real time (`speed=` to speed it up).
- `level`: `1`, a path to a level JSON file, or a parsed dict.

**Short genomes**: if `func` raises `IndexError` (e.g. `genome[t]` past the end of the genome), the game
raises `gallop.GenomeExhausted` (a subclass of `gallop.GameError`) with `.step` and `.max_steps` set. A
genome of length `game.max_steps` can never run out; a shorter one only fails if the run is still going
when it ends.

```python
from gallop import GenomeExhausted
try:
    score = Game(func=lambda t: genome[t]).run_full()
except GenomeExhausted as e:
    ...  # e.step = first missing step
```

**Progress score** = the furthest the player has got from the spawn towards the goal line, as a fraction
(0–1). It never goes down when the player walks back, and is exactly 1.0 when the level is cleared. Use the
fields of `game.result()` (time, horseshoes, death cause…) to build a richer fitness.

## Levels

Levels are JSON files in `gallop/levels/`. Coordinates are tiles (16 px); row 0 is the top and row 16 the
bottom. `terrain` entries are applied in order:

- `ground {x0, x1, top}`: solid from row `top` to the bottom
- `stone {x0, x1, y0, y1?}`: solid block
- `plank {x0, x1, y}`: one-way platform (you can jump up through it)
- `hay {x, y}`: one solid tile
- `water {x0, x1, top}`: kills; only fills empty cells

plus `horseshoes [{x0, x1, y}]`, `geese [{x, top, speed}]`, `spawn_x` and `goal {x}`.

The format is defined by the pydantic models in `gallop/schema.py` (`LevelSpec`). Loading a level validates
it, so unknown keys, unknown terrain types, reversed spans (`x1 < x0`), negative coordinates and a goal or
spawn outside the level raise `pydantic.ValidationError` with the offending field.

## Genetic algorithm

`evolution/` evolves 6000-bit genomes (1000 genes of 6 bits; gene value mod 6 = action) to play level 1.
Fitness is minimised: `(1 - progress) + min(time, 200) / 200`. Defaults: population 400, random parent
pairs, single point crossover, bit-flip mutation at 1/6000, parents + children culled back to 400 by
negative tournaments of size 3. See `journal.md` for the reasoning.

```sh
uv run python -m evolution train --generations 100 --seed 1         # new run in runs/seed1 (all CPU cores)
uv run python -m evolution train --resume runs/seed1 --generations 200   # continue it (after Ctrl-C too)
uv run python -m evolution replay runs/seed1                        # watch the best genome so far
uv run python -m evolution replay runs/seed1 --generation 10        # ...or the best of generation 10
```

A run directory is updated after every generation:

| file | contents |
|---|---|
| `config.json` | the full configuration, enough to reproduce the run |
| `history.csv` | best, worst, mean, median and std fitness per generation |
| `best_per_generation.jsonl` | the best individual of every generation (fitness, game result, bitstring) |
| `best.json`, `best_actions.json` | the best so far; `best_actions.json` also plays with `python -m gallop replay` |
| `checkpoint.npz` | the latest whole population and RNG state, used by `--resume` |

Ctrl-C loses at most the generation in progress. A resumed run gives exactly the same results as one that was
never stopped, and runs are fully determined by `--seed` whatever `--workers` is.

## Command line

```sh
uv run python -m gallop play                                     # play it yourself (arrows + space)
uv run python -m gallop replay tests/data/level1_solution.json   # watch a recorded action list
uv run python -m gallop bench                                    # headless speed
```

## Differences from the reference game

- Level 1 only; checkpoints removed (a death restarts from the spawn).
- No sound, particles, camera shake or title/menu screens; none of them change the outcome.
- The reference game's only randomness (goose honk timers) is cosmetic, so it was dropped and runs are
  fully deterministic.
- A jump pressed during the 0.045 s stomp hit-stop is kept until the next tick instead of being dropped.
- The viewer draws flat rectangles (hitboxes) rather than the original pixel art.

## Layout

```
gallop/
  actions.py     Action enum
  constants.py   physics constants from the reference
  schema.py      pydantic models for the level JSON format
  level.py       JSON level loading -> tile grid
  engine.py      World: deterministic 120 Hz simulation
  game.py        Game / Result: the controller API
  render.py      optional pygame viewer + keyboard controller
  __main__.py    CLI
  levels/level1.json
evolution/
  genome.py      bitstring genome: random init, decode, encode
  fitness.py     play a genome, compute fitness
  operators.py   crossover, mutation, negative tournament cull
  ga.py          GAConfig and the generation loop
  rundir.py      saving/loading run directories and checkpoints
  __main__.py    CLI: train, resume, replay
tests/           level, physics, API and GA tests; data/level1_solution.json clears level 1
```
