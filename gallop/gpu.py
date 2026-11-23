"""A batched copy of the engine in PyTorch: every game of a population advances one tick at a time, together.

    results = simulate_batch(actions, level=1, ticks_per_action=24, max_time=80.0, device="mps")

`actions` is an (N, steps) array of action ids, one row per game. Each piece of game state is a tensor with
one entry per game, and every `if` of the engine becomes a mask: both sides are computed for all games and
torch.where keeps the right one. This is the form a GPU can run, since it does the same operation to many
numbers at once.

Run "eagerly", every tensor operation is a separate GPU kernel launch, a few hundred per tick, and the
launch overhead dominates. compile=True runs the tick through torch.compile, which fuses those operations
into a handful of kernels.

Precision: on the CPU with dtype=float64 this gives bit-identical results to engine.py and fast.py (tests
check it). Apple GPUs (MPS) only support float32, so there results differ: a game that grazes a ledge may
land or fall differently. Only end_on_death=True is supported (what training uses).

Finished games are dropped from the batch now and then ("compaction"), so a population that is mostly
dead after a few seconds doesn't keep paying for the dead games.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from .actions import N_ACTIONS
from .constants import (
    ACC_AIR_IDLE,
    ACC_AIR_RUN,
    ACC_GROUND_IDLE,
    ACC_GROUND_RUN,
    ACC_GROUND_TURN,
    BUFFER,
    COYOTE,
    EMPTY,
    FALLMAX,
    GOOSE_H,
    GOOSE_W,
    GRAV,
    GROUND,
    HAY,
    JUMP,
    PH,
    PLANK,
    PW,
    RUN,
    STEP,
    STOMP_BOUNCE,
    STOMP_BOUNCE_HELD,
    STOMP_FREEZE,
    STONE,
    WATER,
    T,
)
from .fast import DEATH_CAUSES, _level_arrays
from .game import DEFAULT_TICKS_PER_ACTION, GenomeExhausted

_GOOSE, _FALL, _WATER = 1, 2, 3
_CORNER_OFFSETS = [o for n in range(1, 7) for o in (n, -n)]  # corner correction tries +1, -1, +2, -2, ...
_W, _H = float(PW), float(PH)

State = dict[str, torch.Tensor]


@dataclass
class BatchResult:
    """Results of N games, one entry per game (numpy arrays). Field meanings as in gallop.Result."""

    progress: npt.NDArray[np.float64]
    cleared: npt.NDArray[np.bool_]
    death_cause: npt.NDArray[np.int64]  # index into DEATH_CAUSES: 0 none, 1 goose, 2 fall, 3 water
    timed_out: npt.NDArray[np.bool_]
    time: npt.NDArray[np.float64]
    ticks: npt.NDArray[np.int64]
    steps: npt.NDArray[np.int64]
    horseshoes: npt.NDArray[np.int64]
    geese_stomped: npt.NDArray[np.int64]

    def cause(self, i: int) -> str | None:
        return DEATH_CAUSES[int(self.death_cause[i])]


@dataclass(frozen=True)
class _Level:
    grid: torch.Tensor  # flattened tile ids
    coins: torch.Tensor  # (C, 2)
    rows: int
    width: int
    goal_x: float
    max_ticks: int


def _floor(v: torch.Tensor) -> torch.Tensor:
    return torch.floor(v).to(torch.int64)


def _tile(lv: _Level, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    flat = ty.clamp(0, lv.rows - 1) * lv.width + tx.clamp(0, lv.width - 1)
    t = lv.grid[flat]
    t = torch.where((ty < 0) | (ty >= lv.rows), EMPTY, t)
    return torch.where((tx < 0) | (tx >= lv.width), STONE, t)


def _solid(t: torch.Tensor) -> torch.Tensor:
    return (t == GROUND) | (t == STONE) | (t == HAY)


def _hits_ceiling(lv: _Level, px: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    x0, x1 = _floor(px / T), _floor((px + _W - 0.001) / T)
    hit = torch.zeros_like(ty, dtype=torch.bool)
    for k in range(3):  # 18 px wide: spans at most 3 columns
        hit = hit | ((x0 + k <= x1) & _solid(_tile(lv, x0 + k, ty)))
    return hit


def _tick(s: State, d: torch.Tensor, held: torch.Tensor, lv: _Level) -> State:
    """One 1/120 s tick of every game: World.step with each `if` turned into a mask."""
    dtype = s["x"].dtype

    def const(value: float) -> torch.Tensor:
        # torch.where(c, 1.0, 2.0) on two Python floats would give float32, whatever the state's dtype.
        return s["x"].new_full((), value)

    a = ~s["over"] & (s["ticks"] < lv.max_ticks)
    ticks = s["ticks"] + a
    pending = s["pending"] | (a & held & ~s["prev_held"])
    prev_held = torch.where(a, held, s["prev_held"])
    frz = a & (s["freeze"] > 0)
    freeze = torch.where(frz, s["freeze"] - STEP, s["freeze"])
    live = a & ~frz

    # ---------- _update_player
    dt = STEP
    x, y, vx, vy, on_ground = s["x"], s["y"], s["vx"], s["vy"], s["on_ground"]
    acc = torch.where(
        on_ground,
        torch.where(
            d == 0, const(ACC_GROUND_IDLE), torch.where(vx * d < 0, const(ACC_GROUND_TURN), const(ACC_GROUND_RUN))
        ),
        torch.where(d == 0, const(ACC_AIR_IDLE), const(ACC_AIR_RUN)),
    )
    target, delta = d * RUN, acc * dt
    vx = torch.where(vx < target, torch.minimum(vx + delta, target), torch.maximum(vx - delta, target))
    coyote = torch.where(on_ground, COYOTE, s["coyote"] - dt)
    buffer = torch.where(pending, BUFFER, s["buffer"] - dt)
    jump = (buffer > 0) & (coyote > 0)
    vy = torch.where(jump, -float(JUMP), vy)
    buffer = torch.where(jump, 0.0, buffer)
    coyote = torch.where(jump, 0.0, coyote)
    g = torch.where(
        (vy < 0) & ~held,
        const(GRAV * 2.5),
        torch.where((vy.abs() < 80) & held, const(GRAV * 0.5), torch.where(vy > 0, const(GRAV * 1.25), const(GRAV))),
    )
    vy = torch.clamp(vy + g * dt, max=FALLMAX)

    # _move_x
    dx = vx * dt
    x = x + dx
    y0, y1 = _floor(y / T), _floor((y + _H - 0.001) / T)
    tx_r, tx_l = _floor((x + _W) / T), _floor(x / T)
    hit_r = torch.zeros_like(live)
    hit_l = torch.zeros_like(live)
    for k in range(3):  # 22 px tall: spans at most 3 rows
        row_ok = y0 + k <= y1
        hit_r = hit_r | (row_ok & _solid(_tile(lv, tx_r, y0 + k)))
        hit_l = hit_l | (row_ok & _solid(_tile(lv, tx_l, y0 + k)))
    hit_r = hit_r & (dx > 0)
    hit_l = hit_l & (dx < 0)
    x = torch.where(hit_r, (tx_r * T).to(dtype) - _W, torch.where(hit_l, ((tx_l + 1) * T).to(dtype), x))
    vx = torch.where(hit_r, torch.clamp(vx, max=0.0), torch.where(hit_l, torch.clamp(vx, min=0.0), vx))

    # _move_y
    dy = vy * dt
    prev_bot = y + _H
    y = y + dy
    x0, x1 = _floor(x / T), _floor((x + _W - 0.001) / T)
    ty_d = _floor((y + _H) / T)
    land = torch.zeros_like(live)
    for k in range(3):
        t = _tile(lv, x0 + k, ty_d)
        land = land | ((x0 + k <= x1) & (_solid(t) | ((t == PLANK) & (prev_bot <= (ty_d * T).to(dtype) + 0.5))))
    land = land & (dy > 0)
    y = torch.where(land, (ty_d * T).to(dtype) - _H, y)
    vy = torch.where(land, 0.0, vy)
    on_ground = land

    ty_u = _floor(y / T)
    bonk = (dy < 0) & _hits_ceiling(lv, x, ty_u)
    shift = torch.zeros_like(x)
    shifted = torch.zeros_like(live)
    for o in _CORNER_OFFSETS:
        free = bonk & ~shifted & ~_hits_ceiling(lv, x + o, ty_u)
        shift = torch.where(free, float(o), shift)
        shifted = shifted | free
    x = torch.where(shifted, x + shift, x)
    stopped = bonk & ~shifted
    y = torch.where(stopped, ((ty_u + 1) * T).to(dtype), y)
    vy = torch.where(stopped, 0.0, vy)

    # Commit the player update for live games only.
    x = torch.where(live, x, s["x"])
    y = torch.where(live, y, s["y"])
    vx = torch.where(live, vx, s["vx"])
    vy = torch.where(live, vy, s["vy"])
    on_ground = torch.where(live, on_ground, s["on_ground"])
    coyote = torch.where(live, coyote, s["coyote"])
    buffer = torch.where(live, buffer, s["buffer"])
    pending = pending & ~live
    time = torch.where(live, s["time"] + STEP, s["time"])

    # ---------- _interact
    pcx, pcy = x + _W / 2, y + _H / 2
    got = s["got"]
    if lv.coins.shape[0]:
        near = ((lv.coins[:, 0] - pcx[:, None]).abs() < 14) & ((lv.coins[:, 1] - pcy[:, None]).abs() < 17)
        got = got | (live[:, None] & near)
    killed = torch.zeros_like(live)
    stomps = s["stomps"]
    gdead_cols = []
    gx_all, gy_all = s["gx"], s["gy"]
    for i in range(gx_all.shape[1]):
        gx, gy, gdead = gx_all[:, i], gy_all[:, i], s["gdead"][:, i]
        touch = (
            live
            & ~killed
            & ~gdead
            & (x + _W > gx + 2)
            & (x < gx + GOOSE_W - 2)
            & (y + _H > gy + 3)
            & (y < gy + GOOSE_H)
        )
        stomp = touch & (vy > 0) & (y + _H - gy < 11)
        gdead_cols.append(gdead | stomp)
        stomps = stomps + stomp
        bounce = torch.where(held, const(-float(STOMP_BOUNCE_HELD)), const(-float(STOMP_BOUNCE)))
        vy = torch.where(stomp, bounce, vy)
        coyote = torch.where(stomp, 0.0, coyote)
        freeze = torch.where(stomp, STOMP_FREEZE, freeze)
        killed = killed | (touch & ~stomp)
    gdead_all = torch.stack(gdead_cols, dim=1) if gdead_cols else s["gdead"]
    rest = live & ~killed
    water = rest & (_tile(lv, _floor(pcx / T), _floor((y + _H - 4) / T)) == WATER)
    fall = rest & ~water & (y > lv.rows * T + 40)
    clear = rest & ~water & ~fall & (x + _W > lv.goal_x)
    cause = torch.where(water, _WATER, torch.where(fall, _FALL, torch.where(killed, _GOOSE, 0)))
    dead = killed | water | fall
    over = s["over"] | dead | clear
    best_x = torch.where(live, torch.maximum(s["best_x"], x + _W), s["best_x"])

    # ---------- _update_geese (games still running and not frozen)
    move = live & ~over
    gx_cols, gvx_cols = [], []
    for i in range(gx_all.shape[1]):
        alive = move & ~gdead_all[:, i]
        gx, gy, gvx = gx_all[:, i], gy_all[:, i], s["gvx"][:, i]
        nx = gx + gvx * STEP
        fx = torch.where(gvx > 0, nx + GOOSE_W, nx)
        tx = _floor(fx / T)
        my, fy = _floor((gy + GOOSE_H / 2) / T), _floor((gy + GOOSE_H + 1) / T)
        ft = _tile(lv, tx, fy)
        turn = _solid(_tile(lv, tx, my)) | ~(_solid(ft) | (ft == PLANK))
        gvx_cols.append(torch.where(alive & turn, -gvx, gvx))
        gx_cols.append(torch.where(alive & ~turn, nx, gx))

    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "on_ground": on_ground,
        "coyote": coyote,
        "buffer": buffer,
        "freeze": freeze,
        "time": time,
        "ticks": ticks,
        "over": over,
        "cleared": s["cleared"] | clear,
        "cause": torch.where(dead, cause, s["cause"]),
        "best_x": best_x,
        "prev_held": prev_held,
        "pending": pending,
        "stomps": stomps,
        "gx": torch.stack(gx_cols, dim=1) if gx_cols else s["gx"],
        "gy": gy_all,
        "gvx": torch.stack(gvx_cols, dim=1) if gvx_cols else s["gvx"],
        "gdead": gdead_all,
        "got": got,
    }


def _decision(s: State, d: torch.Tensor, held: torch.Tensor, lv: _Level, ticks_per_action: int) -> State:
    """All the ticks of one decision step."""
    for _ in range(ticks_per_action):
        s = _tick(s, d, held, lv)
    return s


# For torch.compile the state is packed into five tensors, one row per field and one column per game. Metal
# kernels can take at most 31 buffers; with ~22 separate state tensors (plus their outputs) the fused
# kernels exceed that and fail to build, while columns of one packed tensor share a buffer.
_FLOATS = ("x", "y", "vx", "vy", "coyote", "buffer", "freeze", "time", "best_x")
_INTS = ("ticks", "cause", "stomps")
_BOOLS = ("on_ground", "over", "cleared", "prev_held", "pending")
Packed = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _pack(s: State) -> Packed:
    return (
        torch.stack([s[k] for k in _FLOATS]),
        torch.stack([s[k] for k in _INTS] + [s[k].to(torch.int64) for k in _BOOLS]),
        torch.stack([s["gx"].T, s["gy"].T, s["gvx"].T]),
        s["gdead"].T.contiguous(),
        s["got"].T.contiguous(),
    )


def _unpack(p: Packed) -> State:
    floats, ints, geese, gdead, got = p
    s = {k: floats[i] for i, k in enumerate(_FLOATS)}
    s.update({k: ints[i] for i, k in enumerate(_INTS)})
    s.update({k: ints[len(_INTS) + i] != 0 for i, k in enumerate(_BOOLS)})
    s.update(gx=geese[0].T, gy=geese[1].T, gvx=geese[2].T, gdead=gdead.T, got=got.T)
    return s


def _tick_packed(p: Packed, d: torch.Tensor, held: torch.Tensor, lv: _Level) -> Packed:
    return _pack(_tick(_unpack(p), d, held, lv))


_compiled: dict[str, Callable] = {}


def _decision_fn(compile: bool) -> Callable:
    if not compile:
        return _decision
    if "tick" not in _compiled:
        # One tick per compiled graph (a 24-tick graph takes minutes to compile). dynamic=True: the batch
        # shrinks with compaction and shouldn't recompile each time. Metal kernels take at most 31 buffers
        # (inputs, outputs and size arguments), so fused kernels are capped at 16 distinct input/output buffers;
        # without the cap the generated kernels fail to build.
        _compiled["tick"] = torch.compile(_tick_packed, dynamic=True, options={"max_fusion_unique_io_buffers": 16})
    tick = _compiled["tick"]

    def decision(s: State, d: torch.Tensor, held: torch.Tensor, lv: _Level, ticks_per_action: int) -> State:
        p = _pack(s)
        for _ in range(ticks_per_action):
            p = tick(p, d, held, lv)
        return _unpack(p)

    return decision


def simulate_batch(
    actions: npt.ArrayLike | torch.Tensor,
    level: int | str | Path = 1,
    *,
    ticks_per_action: int = DEFAULT_TICKS_PER_ACTION,
    max_time: float = 60.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    compact_every: int = 5,
    compile: bool = False,
) -> BatchResult:
    """Play N games at once, one per row of `actions`.

    dtype defaults to float64, or float32 on MPS (which has no float64). compact_every: drop finished games
    from the batch every this many decision steps (0 = never). compile: fuse each decision step with
    torch.compile (slow first call, much faster after).
    """
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float32 if device.type == "mps" else torch.float64
    max_ticks = round(max_time / STEP)
    max_steps = -(-max_ticks // ticks_per_action)

    acts = torch.as_tensor(actions)
    if acts.ndim != 2:
        raise ValueError("actions must be a 2-D array: one row of action ids per game")
    n = acts.shape[0]
    if acts.shape[1] < max_steps:
        raise GenomeExhausted(acts.shape[1], max_steps)
    if n and (int(acts.min()) < 0 or int(acts.max()) >= N_ACTIONS):
        raise ValueError(f"action ids must be in 0..{N_ACTIONS - 1}")
    acts = acts[:, :max_steps].to(device=device, dtype=torch.uint8)

    grid_np, coins_np, geese_np, spawn_x, spawn_y, goal_x = _level_arrays(level)
    lv = _Level(
        grid=torch.as_tensor(grid_np.ravel(), device=device, dtype=torch.int64),
        coins=torch.as_tensor(coins_np, device=device, dtype=dtype),
        rows=grid_np.shape[0],
        width=grid_np.shape[1],
        goal_x=goal_x,
        max_ticks=max_ticks,
    )
    geese = torch.as_tensor(geese_np, device=device, dtype=dtype)
    n_geese, n_coins = geese.shape[0], lv.coins.shape[0]
    direction_of = torch.tensor([0, -1, 1, 0, -1, 1], device=device, dtype=dtype)
    jump_of = torch.tensor([False, False, False, True, True, True], device=device)

    def full(value, dt=dtype):
        return torch.full((n,), value, device=device, dtype=dt)

    s: State = {
        "x": full(spawn_x),
        "y": full(spawn_y),
        "vx": full(0.0),
        "vy": full(0.0),
        "on_ground": full(False, torch.bool),
        "coyote": full(0.0),
        "buffer": full(0.0),
        "freeze": full(0.0),
        "time": full(0.0),
        "ticks": full(0, torch.int64),
        "over": full(False, torch.bool),
        "cleared": full(False, torch.bool),
        "cause": full(0, torch.int64),
        "best_x": full(spawn_x + _W),
        "prev_held": full(False, torch.bool),
        "pending": full(False, torch.bool),
        "stomps": full(0, torch.int64),
        "gx": geese[:, 0].expand(n, n_geese).clone(),
        "gy": geese[:, 1].expand(n, n_geese).clone(),
        "gvx": geese[:, 2].expand(n, n_geese).clone(),
        "gdead": torch.zeros((n, n_geese), device=device, dtype=torch.bool),
        "got": torch.zeros((n, n_coins), device=device, dtype=torch.bool),
    }
    idx = torch.arange(n, device=device)  # original row of each game still in the batch
    steps = full(0, torch.int64)
    out = {k: v.clone() for k, v in s.items()}  # final state, written as games leave the batch
    out_steps = steps.clone()
    decide = _decision_fn(compile)

    def retire(mask: torch.Tensor) -> None:
        rows_out = idx[mask]
        for k, v in s.items():
            out[k][rows_out] = v[mask]
        out_steps[rows_out] = steps[mask]

    for step in range(max_steps):
        running = ~s["over"] & (s["ticks"] < max_ticks)
        if compact_every and step and step % compact_every == 0:
            done = ~running
            if bool(done.any()):
                retire(done)
                s = {k: v[running] for k, v in s.items()}
                idx, steps, acts = idx[running], steps[running], acts[running]
                running = running[running]
            if idx.numel() == 0:
                break
        steps = steps + running
        action = acts[:, step].long()
        s = decide(s, direction_of[action], jump_of[action], lv, ticks_per_action)
    retire(torch.ones_like(idx, dtype=torch.bool))

    span = goal_x - (spawn_x + _W)
    best = out["best_x"].cpu().double()
    cleared = out["cleared"].cpu()
    progress = torch.where(cleared, 1.0, ((best - (spawn_x + _W)) / span).clamp(0.0, 1.0))
    ticks = out["ticks"].cpu()
    return BatchResult(
        progress=progress.numpy(),
        cleared=cleared.numpy(),
        death_cause=out["cause"].cpu().numpy(),
        timed_out=(~out["over"].cpu() & (ticks >= max_ticks)).numpy(),
        time=out["time"].cpu().double().numpy(),
        ticks=ticks.numpy(),
        steps=out_steps.cpu().numpy(),
        horseshoes=out["got"].sum(dim=1).cpu().numpy(),
        geese_stomped=out["stomps"].cpu().numpy(),
    )
