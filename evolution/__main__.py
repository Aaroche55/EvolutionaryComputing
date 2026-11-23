"""Command line entry points.

    python -m evolution train --generations 100 --seed 1        new run, saved to runs/seed1/
    python -m evolution train --resume runs/seed1 --generations 200   continue a stopped/finished run
    python -m evolution replay runs/seed1                       watch the best genome so far
    python -m evolution replay runs/seed1 --generation 10       watch the best of generation 10
    python -m evolution plot runs/seed1                         fitness and clear time -> runs/seed1/progress.png

See rundir.py for the files a run directory holds.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from .fitness import FITNESS_FUNCTIONS
from .ga import Checkpoint, GAConfig, GeneticAlgorithm
from .genome import decode
from .rundir import RunDir, genome_from_record


def report(cp: Checkpoint) -> None:
    s = cp.stats
    print(
        f"gen {s.generation:4d}  evals {s.evaluations:7d}  best {s.best:.4f}  mean {s.mean:.4f}  "
        f"median {s.median:.4f}  best progress {s.best_progress:.3f}  cleared {s.cleared}",
        flush=True,
    )


def train(args: argparse.Namespace) -> None:
    workers = args.workers or os.cpu_count() or 1
    if args.resume:
        run_dir = RunDir(args.resume)
        resume = run_dir.load_checkpoint()
        config = run_dir.load_config()
        config = replace(config, workers=workers, generations=args.generations or config.generations)
        run_dir.save_config(config)
        run_dir.truncate_to(resume.generation)
        print(f"resuming {run_dir.path} from generation {resume.generation} to {config.generations}")
    else:
        d = GAConfig()
        config = GAConfig(
            population_size=args.population or d.population_size,
            generations=args.generations if args.generations is not None else d.generations,
            n_genes=args.genes or d.n_genes,
            tournament_size=args.tournament_size or d.tournament_size,
            mutation_rate=args.mutation_rate,
            max_time=args.max_time or d.max_time,
            fitness=args.fitness,
            seed=args.seed,
            workers=workers,
        )
        run_dir = RunDir(args.out or Path("runs") / f"seed{config.seed}")
        run_dir.start(config)
        resume = None

    def on_generation(cp: Checkpoint) -> None:
        run_dir.record(cp)
        report(cp)

    try:
        run = GeneticAlgorithm(config).run(on_generation=on_generation, resume=resume)
    except KeyboardInterrupt:
        done = run_dir.load_checkpoint().generation if run_dir.checkpoint_file.exists() else None
        print(f"\nstopped after generation {done}. Continue with: python -m evolution train --resume {run_dir.path}")
        return
    r = run.best.result
    print(f"best fitness {run.best.fitness:.4f}: progress {r.progress:.3f}, time {r.time:.2f}s, cleared {r.cleared}")
    print(f"saved to {run_dir.path}/   watch it: python -m evolution replay {run_dir.path}")


def replay(args: argparse.Namespace) -> None:
    from gallop import Game

    run_dir = RunDir(args.run)
    config = run_dir.load_config()
    record = run_dir.best_record(args.generation)
    actions = decode(genome_from_record(record))
    print(
        f"generation {record['generation']}: fitness {record['fitness']:.4f}, "
        f"progress {record['result']['progress']:.3f}, time {record['result']['time']:.2f}s"
    )
    game = Game(
        func=actions.__getitem__,
        ticks_per_action=config.ticks_per_action,
        max_time=config.max_time,
        render=True,
        speed=args.speed,
    )
    game.run_full()
    print(game.result())


def plot(args: argparse.Namespace) -> None:
    from .plot import plot as plot_run

    run_dir = RunDir(args.run)
    out = plot_run(run_dir, args.out or run_dir.path / "progress.png")
    print(f"saved {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="evolution", description="Evolve action sequences for Ballinasloe Gallop.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="start a new run, or continue one with --resume")
    t.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue the run saved in RUN_DIR")
    t.add_argument("--generations", type=int, help="last generation to run to (default 100)")
    t.add_argument("--workers", type=int, help="processes for evaluation (default: all CPU cores)")
    new = t.add_argument_group("new runs only (a resumed run keeps its saved settings)")
    new.add_argument("--population", type=int)
    new.add_argument("--genes", type=int, help="actions per genome, 6 bits each (default 1000)")
    new.add_argument("--tournament-size", type=int)
    new.add_argument("--mutation-rate", type=float, help="per-bit flip probability (default 1/L)")
    new.add_argument("--max-time", type=float, help="seconds of game time per evaluation (default 200)")
    new.add_argument(
        "--fitness", choices=FITNESS_FUNCTIONS, default="sum", help="fitness function, see fitness.py (default sum)"
    )
    new.add_argument("--seed", type=int, default=0)
    new.add_argument("--out", type=Path, help="run directory (default runs/seed<seed>)")

    r = sub.add_parser("replay", help="watch a run's best genome")
    r.add_argument("run", type=Path, help="run directory, e.g. runs/seed1")
    r.add_argument("--generation", type=int, help="best of this generation (default: best so far)")
    r.add_argument("--speed", type=float, default=1.0)

    p = sub.add_parser("plot", help="plot fitness and clear time across generations (needs the plot extra)")
    p.add_argument("run", type=Path, help="run directory, e.g. runs/seed1")
    p.add_argument("--out", type=Path, help="image file (default RUN/progress.png)")

    args = ap.parse_args(argv)
    {"train": train, "replay": replay, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
