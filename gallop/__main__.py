"""Command line entry points.

python -m gallop play                  play level 1 with the keyboard
python -m gallop replay actions.json   watch a recorded action sequence (a JSON list of ints)
python -m gallop bench                 time headless runs
"""

from __future__ import annotations

import argparse
import json
import time

from .actions import Action
from .game import DEFAULT_TICKS_PER_ACTION, Game


def sequence_controller(actions: list[int]):
    """Controller that plays back a fixed list of actions, then does nothing."""
    return lambda t: actions[t] if t < len(actions) else Action.NOOP


def level_arg(s: str) -> int | str:
    """A level number (1) or a path to a level JSON file."""
    return int(s) if s.isdigit() else s


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gallop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play with the keyboard")
    p.add_argument("--level", type=level_arg, default=1)
    r = sub.add_parser("replay", help="render a recorded action sequence")
    r.add_argument("file")
    r.add_argument("--level", type=level_arg, default=1)
    r.add_argument("--ticks-per-action", type=int, default=DEFAULT_TICKS_PER_ACTION)
    r.add_argument("--speed", type=float, default=1.0)
    b = sub.add_parser("bench", help="time headless runs")
    b.add_argument("-n", type=int, default=100)
    args = ap.parse_args(argv)

    if args.cmd == "play":
        from .render import keyboard_controller

        game = Game(keyboard_controller, args.level, ticks_per_action=1, max_time=300, end_on_death=False, render=True)
    elif args.cmd == "replay":
        actions = json.loads(open(args.file).read())
        game = Game(
            sequence_controller(actions),
            args.level,
            ticks_per_action=args.ticks_per_action,
            render=True,
            speed=args.speed,
        )
    else:
        t0 = time.perf_counter()
        for _ in range(args.n):
            Game(lambda t: Action.RIGHT_JUMP if t % 2 else Action.RIGHT).run_full()
        dt = time.perf_counter() - t0
        print(f"{args.n} games in {dt:.2f}s ({dt / args.n * 1000:.1f} ms/game)")
        return
    game.run_full()
    print(game.result())


if __name__ == "__main__":
    main()
