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

## 2026-11-23 — Experiments: selection pressure, fitness, mutation and population size

Reference point: a hand-made solution clears in 20.46 s; running flat out on flat ground would take 20.44 s,
so about 20.5 s is close to the best possible. Each setting below was tried with **one seed only**, and every
run used a different seed, so these are leads to test, not conclusions.

| Run | k | Fitness | Genes | Mutation / bit | Pop. | First clear | Best clear time |
|---|---|---|---|---|---|---|---|
| seed 2 | 3 | sum | 1000 | 1/6000 | 400 | gen 161 | 42.86 s, plateaued (gen 500) |
| seed 3 | 5 | sum | 1000 | 1/6000 | 400 | gen 49 | 38.22 s (gen 100, not continued) |
| seed 4 | 10 | sum | 1000 | 1/6000 | 400 | gen 28 | 47.72 s, stuck from gen 160 to 200 |
| seed 5 | 5 | tiered | 400 | 5/2400 | 400 | gen 49 | 36.46 s, converged (gen 400, 160k games) |
| seed 6 | 5 | tiered | 400 | 5/2400 | 2000 | gen 34 | **25.91 s**, converged (gen 500, 1M games) |

**Tournament size (k).** Higher k found a first clear sooner (k = 3: gen 161, k = 5: 49, k = 10: 28) but
made the whole population copies of that first clear within about 12 generations. With k = 10, all 400
individuals had identical fitness from generation 160: crossover of identical parents changes nothing, and
the few mutants that differed were culled at once. The final time then depends on how fast that first route
was. k = 5 was the best compromise we saw.

**Fitness scaling doesn't matter, ordering does.** We first thought weighting the time term more heavily
would push for faster clears. But negative tournaments only ask which contender is worse, never by how much,
so rescaling a term changes nothing. Only changes that reorder individuals matter. We added a **tiered**
fitness (`--fitness tiered`): a clear scores `time / max_time` (0–1) and any non-clear scores
`2 − progress` (1–2). Any clear now beats any non-clear, and dying early is no longer rewarded (see the
first observation above). The original "sum" fitness remains the default, so old runs still resume.

**Effective mutation rate.** At 1/L only the genes actually played matter: a 48 s clear uses about 240 of
1000 genes, so about 3 in 4 children played exactly like their parent. We cut the genome to 400 genes (80 s,
which still leaves room for slow clears) and raised the rate to 5/L (about 5 flips per child, about 2.5 in
played genes). `max_time` must now fit in the genome (genes / 5 s), and a config that doesn't is rejected.
With these changes the population stayed diverse far longer: seed 5 still had a 7 s gap between its fastest
and median clear 50 generations after its first clear, where the earlier runs had none.

**Population size.** Raising the population from 400 to 2000 (seed 6) gave the biggest improvement:
25.91 s, against 36.46 s for seed 5. Per game played, though, it was slower at first: after 160k games
seed 6 was at 41.8 s and seed 5 at 36.5 s. But seed 5 had stopped improving by then, while seed 6 kept
finding faster routes for another 300 generations, with the gains per 100 generations shrinking:

| Generations | Gain |
|---|---|
| 100 → 200 | 11.1 s |
| 200 → 300 | 1.9 s |
| 300 → 400 | 1.1 s |
| 400 → 500 | 0.2 s |

At generation 500 the median clear time was within 0.06 s of the best, so we treated the run as converged.
On 12 cores, 100 generations take about 2 minutes at a population of 400 and about 10 minutes at 2000.

**Takeaways so far.** Every run ends converged on one route; the settings decide how good that route is
before diversity runs out. More diversity (moderate k, mutation that hits played genes, a larger population)
consistently gave better final times. The best GA result is still 5.4 s slower than the hand-made solution.

**Next.** Repeat the promising settings over several seeds (the spec asks for 30 seeded runs) before
claiming any effect; consider explicit diversity preservation, e.g. not letting duplicate genomes survive
the cull.

## 2026-11-23 — Speeding up training: Numba vs GPU

At population 2000 a batch of 100 generations took about 10 minutes, too slow for 30 seeded runs per
setting. Profiling showed that playing the games was essentially all of the cost (about 5 µs per physics
tick in pure Python, 15–30 ms per game); the GA operators are negligible. Games are independent, and they
already ran in parallel on all 12 cores, so the remaining lever was the speed of a single game.

**Numba (adopted).** `gallop/fast.py` is a line-by-line translation of the engine into plain variables and
numpy arrays, compiled to machine code with Numba. It uses the same float64 arithmetic in the same order, so
results are bit-identical to the reference engine. We checked this on 685 games covering clears, falls,
goose deaths, stomps, timeouts and respawns, and by re-running seed 6 for 100 generations, which reproduced
the original history exactly. One game now takes about 0.1 ms (≈150× faster), and 100 generations at
population 2000 take about 12 s instead of about 10 minutes. The cost is that the game rules now exist
twice; equivalence tests catch any drift between the two. The Python engine remains the reference, used for
replay and rendering.

**GPU (tried, rejected).** We also wrote a batched PyTorch engine (`gallop/gpu.py`, on the `gpu-engine`
branch) that advances every game of a population one tick at a time, with each `if` of the engine turned
into a mask. It runs on the Apple GPU (MPS) and was compiled with `torch.compile` to fuse operations into
fewer kernels. Getting it to compile at all needed the state packed into a few tensors, and a cap on buffers
per kernel, because Metal allows at most 31. On the CPU in float64 it is bit-identical to Numba.

| Games | Numba (12 cores) | GPU (compiled) | GPU ÷ Numba |
|---|---|---|---|
| 10 k, late-training population | 108 k games/s | 6.5 k games/s | 0.06× |
| 100 k, late-training population | 118 k games/s | 24 k games/s | 0.21× |
| 1 M, late-training population | 106 k games/s | 29 k games/s | 0.27× |
| 1 M, random population | 40 k games/s | 10 k games/s | 0.26× |

The GPU stayed about 4× slower even at a million games, for three reasons:
- Ticks are sequential, so every one of up to 9600 ticks costs several kernel launches and a full pass of
  the state through memory, while Numba keeps a game's state in registers.
- Masking means every game pays for every branch on every tick, e.g. the 13 corner-correction checks.
- The work per game is small and branchy, unlike the large dense arithmetic GPUs are built for.

Precision was a second problem. Apple GPUs have no float64, and in float32 only 93% of past runs' best
genomes ended the same way as in float64, with 41 of 205 clears lost. Evolved genomes are tuned to the exact
physics, so training on a float32 GPU would evolve against slightly different physics.

**Takeaways.** For a small, branchy, sequential simulation, compiling the per-game loop and running games
in parallel on CPU cores beats data-parallel GPU execution. Measuring first (where does the time go?)
pointed straight at the engine. With training now about 50× faster, 30 seeded runs per setting are
practical.

## References

[1] Code Bullet, YouTube channel. https://www.youtube.com/@CodeBullet

[2] Anthropic, Claude Code (model: Claude Opus 5.5), used to port the game to Python, implement the GA and the Numba and GPU engines, and run and summarise
the experiments.
