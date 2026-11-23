import json
from pathlib import Path

import pytest

from gallop.engine import World
from gallop.level import load_level

DATA = Path(__file__).parent / "data"


@pytest.fixture
def solution() -> list[int]:
    """A recorded action sequence (default 24 ticks per action, 5 per second) that clears level 1."""
    return json.loads((DATA / "level1_solution.json").read_text())


def flat_level(width=40, extra_terrain=(), geese=(), horseshoes=(), goal=35):
    """A small flat test level: ground at row 14 across the whole width."""
    return {
        "name": "test", "width": width, "rows": 17, "spawn_x": 2,
        "terrain": [{"type": "ground", "x0": 0, "x1": width - 1, "top": 14}, *extra_terrain],
        "geese": list(geese), "horseshoes": list(horseshoes), "goal": {"x": goal},
    }


def make_world(data, **kw) -> World:
    return World(load_level(data), **kw)
