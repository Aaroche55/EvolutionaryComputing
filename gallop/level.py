"""Loading level definitions from JSON and building the tile grid.

A level file lists terrain primitives in order (later ones overwrite earlier ones, except
water which only fills empty cells), horseshoe rows, geese and the goal. Coordinates are in
tiles; row 0 is the top of the screen and `rows - 1` the bottom. The file format is defined
and validated by schema.LevelSpec; see levels/level1.json for an example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import EMPTY, GOOSE_H, GROUND, HAY, PH, PLANK, SOLID_TILES, STONE, WATER, T
from .schema import Ground, Hay, LevelSpec, Plank, Stone, Water

LEVELS_DIR = Path(__file__).parent / "levels"


@dataclass
class Level:
    name: str
    width: int  # in tiles
    rows: int
    grid: list[list[int]]  # grid[y][x] -> tile id
    coins: list[tuple[float, float]]  # horseshoe centres, px
    geese: list[tuple[float, float, float]]  # (x, y, vx) of each goose's top-left, px
    spawn: tuple[float, float]  # player top-left, px
    goal_x: float  # crossing this (player right edge) clears the level
    spec: LevelSpec | None = field(repr=False, default=None)

    def tile(self, x: int, y: int) -> int:
        if x < 0 or x >= self.width:
            return STONE  # the level edges act as walls
        if y < 0 or y >= self.rows:
            return EMPTY
        return self.grid[y][x]

    def top_at(self, x: int) -> int:
        """Row of the highest standable tile in column x (rows if the column is a pit)."""
        for y in range(self.rows):
            t = self.grid[y][x]
            if t in SOLID_TILES or t == PLANK:
                return y
        return self.rows


def load_level(level: int | str | Path | dict[str, Any] | LevelSpec) -> Level:
    """Load a level by number (1 -> levels/level1.json), path, parsed dict, or LevelSpec.

    Raises pydantic.ValidationError if the level data is malformed.
    """
    if isinstance(level, LevelSpec):
        spec = level
    elif isinstance(level, dict):
        spec = LevelSpec.model_validate(level)
    else:
        path = LEVELS_DIR / f"level{level}.json" if isinstance(level, int) else Path(level)
        spec = LevelSpec.model_validate_json(path.read_text())
    return build_level(spec)


def build_level(spec: LevelSpec) -> Level:
    w, rows = spec.width, spec.rows
    grid = [[EMPTY] * w for _ in range(rows)]

    def put(x: int, y: int, t: int) -> None:
        if 0 <= x < w and 0 <= y < rows:
            grid[y][x] = t

    for p in spec.terrain:
        match p:
            case Ground():
                for x in range(p.x0, p.x1 + 1):
                    for y in range(p.top, rows):
                        put(x, y, GROUND)
            case Stone():
                y1 = rows - 1 if p.y1 is None else p.y1
                for x in range(p.x0, p.x1 + 1):
                    for y in range(p.y0, y1 + 1):
                        put(x, y, STONE)
            case Plank():
                for x in range(p.x0, p.x1 + 1):
                    put(x, p.y, PLANK)
            case Water():
                for x in range(p.x0, min(p.x1, w - 1) + 1):
                    for y in range(p.top, rows):
                        if grid[y][x] == EMPTY:
                            grid[y][x] = WATER
            case Hay():
                put(p.x, p.y, HAY)

    coins = [(float(x * T + 8), float(row.y * T + 8)) for row in spec.horseshoes for x in range(row.x0, row.x1 + 1)]
    geese = [(float(g.x * T), float(g.top * T - GOOSE_H), -g.speed) for g in spec.geese]

    lv = Level(
        name=spec.name,
        width=w,
        rows=rows,
        grid=grid,
        coins=coins,
        geese=geese,
        spawn=(0.0, 0.0),
        goal_x=0.0,
        spec=spec,
    )
    lv.spawn = (float(spec.spawn_x * T), float(lv.top_at(spec.spawn_x) * T - PH))
    # The goal is a shop; the level clears when the player reaches its door (reference: door.x + 6).
    lv.goal_x = spec.goal.x * T + 40 + 6
    return lv
