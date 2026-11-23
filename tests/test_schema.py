import json

import pytest
from pydantic import ValidationError

from gallop.level import LEVELS_DIR, load_level
from gallop.schema import Ground, Hay, LevelSpec

from conftest import flat_level


def test_level1_file_is_valid():
    spec = LevelSpec.model_validate_json((LEVELS_DIR / "level1.json").read_text())
    assert spec.name == "The Fair Green"
    assert isinstance(spec.terrain[0], Ground)
    assert isinstance(spec.terrain[1], Hay)


def test_round_trips_through_json():
    spec = LevelSpec.model_validate(flat_level())
    assert LevelSpec.model_validate_json(spec.model_dump_json()) == spec


def test_load_level_accepts_spec():
    spec = LevelSpec.model_validate(flat_level())
    assert load_level(spec).width == 40


def test_defaults_applied():
    spec = LevelSpec.model_validate(flat_level(geese=[{"x": 5, "top": 14}]))
    assert spec.rows == 17
    assert spec.geese[0].speed == 32


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["terrain"].append({"type": "lava", "x0": 0, "x1": 1}), "does not match any of the expected tags"),
        (lambda d: d["terrain"].append({"type": "plank", "x0": 5, "x1": 2, "y": 10}), "x1 .* must be >= x0"),
        (lambda d: d["terrain"].append({"type": "hay", "x": 3}), "Field required"),
        (lambda d: d["terrain"].append({"type": "hay", "x": -1, "y": 3}), "greater than or equal to 0"),
        (lambda d: d.update(widht=50), "Extra inputs are not permitted"),
        (lambda d: d.pop("goal"), "Field required"),
        (lambda d: d.update(goal={"x": 40}), "goal.x .* outside the level"),
        (lambda d: d.update(spawn_x=99), "spawn_x .* outside the level"),
        (lambda d: d.update(geese=[{"x": 5, "top": 14, "speed": -3}]), "greater than 0"),
    ],
    ids=[
        "unknown-terrain",
        "reversed-span",
        "missing-field",
        "negative",
        "typo-key",
        "no-goal",
        "goal-out-of-bounds",
        "spawn-out-of-bounds",
        "bad-goose-speed",
    ],
)
def test_invalid_levels_rejected(change, message):
    data = flat_level()
    change(data)
    with pytest.raises(ValidationError, match=message):
        load_level(data)


def test_invalid_json_file_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"width": 10, "terrain": []}))
    with pytest.raises(ValidationError, match="goal"):
        load_level(path)
