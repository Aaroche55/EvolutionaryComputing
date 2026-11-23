# Research & Design Journal

Written to the [Research and Design Journal spec](<project spec/Evolutionary Algorithms_ Research and Design Journal - CS6524 - SOFTWARE MEETS WORLD 2026_7 SEM1 - University Of Limerick.pdf>)
(CS6524, due 11 Dec 2026). It records our ideas as they develop, our sources of inspiration and how
they shaped the design, and alternatives, dead ends and reflections. Every source is cited under References.

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

## References

[1] Code Bullet, YouTube channel. https://www.youtube.com/@CodeBullet

[2] Anthropic, Claude Code (model: Claude Opus 5.5), used to port the game to Python. See the AI usage
declaration in the design documentation.
