# Release validation

Validation date: 2026-10-05.

- Released full-training presets match all three saved configurations exactly, excluding only `logdir`.
- Stored PIVC256, PI256, and VC256 checkpoints were inspected. Each posterior `dyn/obs0/kernel` and its optimizer moments have shape `(6145, 512)`, with no extra option-vector dimensions. Stored step counters match the criterion records.
- Task/criterion checks passed for PIVC and PI: 12/16 boundaries, rejection of incomplete/imbalanced/fall sessions, first-reward state persistence, per-start scoring, bright/dim cues, and one-step-delayed reward/reset behavior.
- Physical task tests passed 1,000 balanced sequence samples, config-0 timing, barrier collision, physical reward collection, fall penalty, and timeout behavior.
- VC tests passed 32 frame-transition cases, 8 rotated timing cases, 24 physical return routes, 8 physical goal-seeking routes, all 4 collision orientations, and actual rendered observation/reward-delay checks.
- Verified checkpoint tests passed retention of the newly saved checkpoint, cleanup of older differently named folders, second save, and reload.
- `pip check` passed; installed package versions matched every applicable entry in the recorded requirements lock. Shapely 2.1.2 was installed separately into an isolated validation dependency directory.

GPU integration results are recorded in `smoke-validation.json`. Each run uses the packaged launcher with `smoke --batch_length 256 --run.steps 2048`: 2 environment workers, batch 4, the real 256-step sequences, training ratio 8, and the full model. These runs use separate validation output directories, not the original training runs. Checkpoint arrays, update counters, posterior width, criterion state, and original checkpoint hashes are verified after execution.

The validation environment reused the established pinned Python packages through an isolated virtual environment, plus a fresh Shapely installation. A complete clean installation of every locked dependency was not repeated. Full training to criterion was not repeated, and these smoke runs do not measure task acquisition or generalization. The final source snapshot is from the live Linux workspace; historical source archives attached to each original checkpoint were not available.
