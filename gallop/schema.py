"""Pydantic models for level JSON files.

Validation happens once when a level is loaded, so typos and out-of-range coordinates fail loudly
with a clear message instead of producing a subtly broken level. The simulation itself only ever
sees the built tile grid (see level.py) and never touches these models.

Coordinates are in tiles: x runs 0..width-1 left to right, y runs 0..rows-1 top to bottom.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from .constants import GOOSE_SPEED


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Span(_Model):
    """A horizontal run of columns x0..x1 (inclusive)."""

    x0: NonNegativeInt
    x1: NonNegativeInt

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x1 < self.x0:
            raise ValueError(f"x1 ({self.x1}) must be >= x0 ({self.x0})")
        return self


class Ground(_Span):
    """Solid earth from row `top` down to the bottom of the level."""

    type: Literal["ground"]
    top: NonNegativeInt


class Stone(_Span):
    """Solid stone block from row y0 to y1 (default: the bottom of the level)."""

    type: Literal["stone"]
    y0: NonNegativeInt
    y1: NonNegativeInt | None = None


class Plank(_Span):
    """One-way platform: you can jump up through it and land on top."""

    type: Literal["plank"]
    y: NonNegativeInt


class Water(_Span):
    """Deadly water from row `top` down; only fills cells nothing else occupies."""

    type: Literal["water"]
    top: NonNegativeInt


class Hay(_Model):
    """A single solid hay bale."""

    type: Literal["hay"]
    x: NonNegativeInt
    y: NonNegativeInt


Terrain = Annotated[Ground | Stone | Plank | Water | Hay, Field(discriminator="type")]


class HorseshoeRow(_Span):
    y: NonNegativeInt


class GooseSpec(_Model):
    x: NonNegativeInt
    top: NonNegativeInt = Field(description="row of the ground the goose walks on")
    speed: float = Field(default=GOOSE_SPEED, gt=0, description="pixels per second; geese start walking left")


class Goal(_Model):
    x: NonNegativeInt


class LevelSpec(_Model):
    name: str = ""
    subtitle: str = ""
    width: PositiveInt
    rows: PositiveInt = 17
    spawn_x: NonNegativeInt = 2
    terrain: list[Terrain]
    horseshoes: list[HorseshoeRow] = []
    geese: list[GooseSpec] = []
    goal: Goal

    @model_validator(mode="after")
    def _in_bounds(self) -> Self:
        if self.spawn_x >= self.width:
            raise ValueError(f"spawn_x ({self.spawn_x}) is outside the level (width {self.width})")
        if self.goal.x >= self.width:
            raise ValueError(f"goal.x ({self.goal.x}) is outside the level (width {self.width})")
        return self
