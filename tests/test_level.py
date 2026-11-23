from gallop.constants import EMPTY, GROUND, HAY, PH, PLANK, T, WATER
from gallop.level import load_level


def test_level1_loads_from_json():
    lv = load_level(1)
    assert lv.name == "The Fair Green"
    assert (lv.width, lv.rows) == (200, 17)
    assert len(lv.grid) == 17 and all(len(r) == 200 for r in lv.grid)
    assert len(lv.coins) == 38
    assert len(lv.geese) == 4


def test_level1_terrain_matches_reference():
    lv = load_level(1)
    assert lv.grid[14][0] == GROUND and lv.grid[13][0] == EMPTY
    assert all(lv.top_at(x) == lv.rows for x in (29, 30, 31))  # first pit
    assert lv.grid[12][17] == HAY and lv.grid[13][16] == HAY
    assert lv.top_at(60) == 12  # raised ground
    assert lv.grid[8][84] == PLANK
    assert lv.top_at(100) == 13


def test_spawn_and_goal_positions():
    lv = load_level(1)
    assert lv.spawn == (2 * T, 14 * T - PH)
    assert lv.goal_x == 185 * T + 46


def test_out_of_bounds_tiles():
    lv = load_level(1)
    assert lv.tile(-1, 5) != EMPTY  # walls at the level edges
    assert lv.tile(lv.width, 5) != EMPTY
    assert lv.tile(5, -1) == EMPTY  # open sky above
    assert lv.tile(5, 99) == EMPTY


def test_level_from_dict_and_water_only_fills_gaps():
    lv = load_level(
        {
            "width": 10,
            "rows": 6,
            "spawn_x": 1,
            "terrain": [{"type": "ground", "x0": 0, "x1": 3, "top": 4}, {"type": "water", "x0": 0, "x1": 9, "top": 5}],
            "goal": {"x": 7},
        }
    )
    assert lv.grid[5][0] == GROUND
    assert lv.grid[5][6] == WATER
    assert lv.grid[4][6] == EMPTY
