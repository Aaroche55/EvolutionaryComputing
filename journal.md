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

## References

[1] Code Bullet, YouTube channel. https://www.youtube.com/@CodeBullet

[2] Anthropic, Claude Code (model: Claude Opus 5.5), used to port the game to Python.
