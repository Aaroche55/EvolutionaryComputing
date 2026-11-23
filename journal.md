# Research & Design Journal

## 2026-11-23 — Planning session

Inspiration: Code Bullet's videos of AIs learning to play games [1].

1. **Rewrite the game in Python** with an interface that lets an agent control the character.
   Done with Claude (AI-assisted) [2]: `gallop/` is a headless port of level 1 of
   `og_game/ballinasloe-gallop.html`, where `Game(func, level).run_full()` returns a 0–1 progress score.
2. **Initial population:** random bitstrings. Each group of bits encodes one action (e.g. left, left, right).

**Fitness: ideas so far**

- progress⁻¹ → scale → weight?
- time → scale → weight?
- (open)

## 2026-11-23 — GA design decisions

Implemented in `evolution/`. Decisions, with the reason for each:

- **Representation:** a bitstring of 6 × 1000 bits. Each 6-bit gene is read as an integer and taken
  modulo the 6 actions. Modulo slightly favours actions 0–3 (11/64 each) over LEFT_JUMP and RIGHT_JUMP
  (10/64 each); we accepted this bias to keep the encoding simple.
- **Decision rate and time limit:** 5 actions per second, `max_time` = 200 s, so 1000 genes cover a whole run
  and a genome can never run out. We considered 300 s, but that would need 1500 genes.
- **Fitness (minimise):** `(1 − progress) + min(time, max_time) / max_time`. Both terms lie in [0, 1].
  We chose 1 − progress over 1/progress (the whiteboard idea): 1/progress is undefined when progress = 0 and
  would swamp the time term.
- **Initialisation:** every bit is an independent fair coin flip.
- **Selection and replacement:** each generation, random parent pairs make 400 children. Parents and
  children are pooled (800), and negative tournaments (k = 3: draw 3, remove the worst) cull back to 400.
  Parents carry forward, and the best fitness can never get worse.
- **Crossover:** single point at a random bit position, two children per pair.
- **Mutation:** bit flip with rate 1/L (1/6000, about one flip per child). This wasn't in the original
  plan. We added it because crossover alone can only recombine bits already in the population.
- **Reproducibility:** one seed drives the whole run, and results are identical whatever the number of
  worker processes.
- **Checkpointing:** after every generation we save that generation's best individual, plus the whole
  current population and RNG state. Training can then stop and resume with identical results. We keep only
  the latest population, about 300 KB, rather than every generation's, since resuming only needs the latest.

**First observation (20 generations, seed 1):** best fitness fell from 0.756 to 0.547, and the best got 65% of
the way through the level. However, the time term rewards dying early: about 92% of random genomes fall into
a pit within seconds, and these score better than genomes that survive but get stuck. Best progress
sometimes drops while best fitness improves (generation 2 → 3: 0.438 → 0.388). We should revisit this, e.g.
by counting the time term only for runs that clear the level.

## References

[1] Code Bullet, YouTube channel. https://www.youtube.com/@CodeBullet

[2] Anthropic, Claude Code (model: Claude Opus 5.5), used to port the game to Python.
