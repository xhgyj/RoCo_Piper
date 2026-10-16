# Frozen assembly-plan fixtures

These two assembly-plan/v2 inputs were copied byte-for-byte from historical runs on 2026-10-15 so offline tests no longer depend on mutable run directories. Do not regenerate or normalize these files during tests. The original evidence remains untouched.

| Fixture | Original repository path | SHA-256 |
| --- | --- | --- |
| example1-compound-pick.json | runs/example1-compound-pick-20260915T130000Z/generated-plan.json | 02b5f86513f1daaf6763b787c5a791b1acd0a7656e8cf0fb1e5420e623f9cae3 |
| example1-scene-smoke.json | runs/scene-smoke-20260914T093942Z/input/plan.json | ff2ccfc1ee89286cb21ab94ad96f315785bd8be60599244f29ce743bddb7b92a |

Both carry the same semantic plan ID but different captured wall_time_seconds diagnostics. Keep both exact byte sequences to preserve the prior tests' inputs and provenance. Neither fixture certifies current runtime success.
