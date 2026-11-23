"""Optional pygame viewer. The simulation never imports this; it is only for watching runs and playing by hand.

Drawing is deliberately plain (flat rectangles in the reference palette) — it shows exactly what the
physics sees: tiles, hitboxes, horseshoes, geese and the goal line.
"""

from __future__ import annotations

import pygame

from .actions import Action
from .constants import GROUND, HAY, PLANK, STONE, T, WATER
from .engine import World

VW, VH, SCALE = 480, 272, 2
COLORS = {
    "sky": (127, 180, 214), "ground": (107, 74, 47), "grass": (104, 179, 70), "stone": (163, 159, 149),
    "plank": (164, 114, 63), "water": (58, 127, 174), "hay": (224, 182, 76), "pony": (154, 98, 53),
    "rider": (122, 31, 61), "goose": (244, 241, 234), "beak": (240, 138, 36), "shoe": (217, 221, 227),
    "goal": (214, 38, 49), "ink": (251, 241, 220), "shadow": (35, 26, 43),
}
TILE_COLORS = {GROUND: "ground", STONE: "stone", PLANK: "plank", WATER: "water", HAY: "hay"}


class Renderer:
    def __init__(self, world: World, speed: float = 1.0):
        pygame.init()
        pygame.display.set_caption(f"Ballinasloe Gallop — {world.level.name}")
        self.screen = pygame.display.set_mode((VW * SCALE, VH * SCALE))
        self.low = pygame.Surface((VW, VH))
        self.font = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()
        self.world = world
        self.fps = 60 * speed
        self.cam_x = 0.0

    def tick(self, action: Action) -> None:
        """Called after every physics tick; draws every other one (60 fps) and keeps real-time pace."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                raise SystemExit
        if self.world.ticks % 2:
            return
        self.draw(action)
        self.clock.tick(self.fps)

    def finish(self, result) -> None:
        msg = "Cleared!" if result.cleared else ("Died: " + result.death_cause if result.died else "Out of time")
        self.draw(None, f"{msg}   progress {result.progress:.3f}   time {result.time:.2f}s   "
                        f"shoes {result.horseshoes}/{result.horseshoes_total}")
        end = pygame.time.get_ticks() + 2500
        while pygame.time.get_ticks() < end:
            for e in pygame.event.get():
                if e.type in (pygame.QUIT, pygame.KEYDOWN):
                    end = 0
            self.clock.tick(30)
        pygame.quit()

    def draw(self, action: Action | None, banner: str | None = None) -> None:
        w, lv, P, g = self.world, self.world.level, self.world.player, self.low
        target = min(max(P.x + P.w / 2 - VW / 2 + P.face * 36, 0), lv.width * T - VW)
        self.cam_x += (target - self.cam_x) * 0.2
        cx = round(self.cam_x)

        g.fill(COLORS["sky"])
        x0, x1 = max(0, cx // T), min(lv.width - 1, (cx + VW) // T)
        for x in range(x0, x1 + 1):
            for y in range(lv.rows):
                t = lv.grid[y][x]
                if not t:
                    continue
                sx, sy = x * T - cx, y * T
                if t == PLANK:
                    g.fill(COLORS["plank"], (sx, sy, T, 5))
                    continue
                g.fill(COLORS[TILE_COLORS[t]], (sx, sy, T, T))
                if t == GROUND and (y == 0 or lv.grid[y - 1][x] != GROUND):
                    g.fill(COLORS["grass"], (sx, sy, T, 5))

        gx = lv.goal_x - cx
        g.fill(COLORS["goal"], (gx - 46, lv.top_at(min(lv.width - 1, int(lv.goal_x // T))) * T - 70, 96, 70))
        pygame.draw.line(g, COLORS["ink"], (gx, 0), (gx, VH))

        for c in w.coins:
            if not c.got:
                pygame.draw.rect(g, COLORS["shoe"], (c.x - cx - 4, c.y - 5, 9, 9), 2)
        for q in w.geese:
            if not q.dead:
                g.fill(COLORS["goose"], (q.x - cx, q.y, q.w, q.h))
                bx = q.x - cx + (q.w if q.vx > 0 else -3)
                g.fill(COLORS["beak"], (bx, q.y + 1, 3, 2))
        if P.dead <= 0 or not w.over:
            g.fill(COLORS["pony"], (P.x - cx, P.y + 8, P.w, P.h - 8))
            g.fill(COLORS["rider"], (P.x - cx + 6, P.y, 6, 9))

        hud = f"{lv.name}   t={w.time:5.2f}s   progress {w.progress:.3f}   shoes {w.shoes}/{len(w.coins)}"
        if action is not None:
            hud += f"   {action.name}"
        g.blit(self.font.render(hud, True, COLORS["shadow"]), (6, 6))
        if banner:
            g.fill(COLORS["shadow"], (0, VH // 2 - 14, VW, 28))
            g.blit(self.font.render(banner, True, COLORS["ink"]), (10, VH // 2 - 6))
        pygame.transform.scale(g, self.screen.get_size(), self.screen)
        pygame.display.flip()


def keyboard_controller(_t: int) -> Action:
    """A controller for a human player: arrows / A D to move, Space / Up / W to jump."""
    k = pygame.key.get_pressed()
    left = k[pygame.K_LEFT] or k[pygame.K_a]
    right = k[pygame.K_RIGHT] or k[pygame.K_d]
    jump = k[pygame.K_SPACE] or k[pygame.K_UP] or k[pygame.K_w]
    d = (1 if right else 0) - (1 if left else 0)
    if jump:
        return {-1: Action.LEFT_JUMP, 0: Action.JUMP, 1: Action.RIGHT_JUMP}[d]
    return {-1: Action.LEFT, 0: Action.NOOP, 1: Action.RIGHT}[d]

