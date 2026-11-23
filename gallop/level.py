"""Loading level definitions from JSON and building the tile grid.

A level file lists terrain primitives in order (later ones overwrite earlier ones, except
water which only fills empty cells), horseshoe rows, geese and the goal. Coordinates are in
tiles; row 0 is the top of the screen and `rows - 1` the bottom. See levels/level1.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .constants import EMPTY, GOOSE_H, GOOSE_SPEED, GROUND, HAY, PH, PLANK, SOLID_TILES, STONE, T, WATER

LEVELS_DIR = Path(__file__).parent / "levels"


@dataclass
class Level:
    name: str
    width: int                      # in tiles
    rows: int
    grid: list[list[int]]           # grid[y][x] -> tile id
    coins: list[tuple[float, float]]                 # horseshoe centres, px
    geese: list[tuple[float, float, float]]          # (x, y, vx) of each goose's top-left, px
    spawn: tuple[float, float]                       # player top-left, px
    goal_x: float                                    # crossing this (player right edge) clears the level
    source: dict = field(repr=False, default_factory=dict)

    def tile(self, x: int, y: int) -> int:
        if x < 0 or x >= self.width:
            return STONE                # the level edges act as walls
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


def load_level(level: int | str | Path | dict) -> Level:
    """Load a level by number (1 -> levels/level1.json), path, or an already-parsed dict."""
    if isinstance(level, dict):
        data = level
    else:
        path = LEVELS_DIR / f"level{level}.json" if isinstance(level, int) else Path(level)
        data = json.loads(path.read_text())
    return build_level(data)


def build_level(data: dict) -> Level:
    w, rows = data["width"], data.get("rows", 17)
    grid = [[EMPTY] * w for _ in range(rows)]

    def put(x, y, t):
        if 0 <= x < w and 0 <= y < rows:
            grid[y][x] = t

    for p in data["terrain"]:
        kind = p["type"]
        if kind == "ground":
            for x in range(p["x0"], p["x1"] + 1):
                for y in range(p["top"], rows):
                    put(x, y, GROUND)
        elif kind == "stone":
            y1 = p.get("y1", rows - 1)
            for x in range(p["x0"], p["x1"] + 1):
                for y in range(p["y0"], y1 + 1):
                    put(x, y, STONE)
        elif kind == "plank":
            for x in range(p["x0"], p["x1"] + 1):
                put(x, p["y"], PLANK)
        elif kind == "water":
            for x in range(p["x0"], p["x1"] + 1):
                for y in range(p["top"], rows):
                    if 0 <= x < w and grid[y][x] == EMPTY:
                        grid[y][x] = WATER
        elif kind == "hay":
            put(p["x"], p["y"], HAY)
        else:
            raise ValueError(f"unknown terrain type {kind!r}")

    coins = [(x * T + 8, row["y"] * T + 8) for row in data.get("horseshoes", []) for x in range(row["x0"], row["x1"] + 1)]
    geese = [(g["x"] * T, g["top"] * T - GOOSE_H, -g.get("speed", GOOSE_SPEED)) for g in data.get("geese", [])]

    lv = Level(name=data.get("name", ""), width=w, rows=rows, grid=grid, coins=coins, geese=geese,
               spawn=(0.0, 0.0), goal_x=0.0, source=data)
    sx = data.get("spawn_x", 2)
    lv.spawn = (sx * T, lv.top_at(sx) * T - PH)
    # The goal is a shop; the level clears when the player reaches its door (reference: door.x + 6).
    lv.goal_x = data["goal"]["x"] * T + 40 + 6
    return lv
