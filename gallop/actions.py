"""The actions a controller (human or evolved) can choose each decision step."""

from enum import IntEnum


class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5

    @property
    def direction(self) -> int:
        """-1 for left, +1 for right, 0 for neither."""
        if self in (Action.LEFT, Action.LEFT_JUMP):
            return -1
        if self in (Action.RIGHT, Action.RIGHT_JUMP):
            return 1
        return 0

    @property
    def jump(self) -> bool:
        """True while the jump button is held. A jump starts on the step it goes from released to held."""
        return self in (Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP)


N_ACTIONS = len(Action)
