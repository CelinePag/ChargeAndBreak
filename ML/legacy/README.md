# legacy/ — the 2026-08 MLP, restored verbatim

The previous ML tree (deleted, in git history) trained a neural network to
**classify** which of 12 actions the teacher chose, and reported far better
numbers than this project's cost-regression MLP did. It is restored here and
run **exactly as it was built**, so its row in the results table is the real
thing rather than a reimplementation.

## What is NOT comparable, and we say so

This model differs from the current arms in several ways at once:

| | legacy | current arms |
|---|---|---|
| features | 23 dashboard + 6·K forward nodes (141 at K=20) | 91, typed lookahead |
| teacher runs | 772 (deduped copies in `ML/data/miptail/`) | 830 |
| what is learned | which action was chosen (12 classes) | cost of every action |
| framework | PyTorch | scikit-learn / LightGBM |
| decision code | its own `rollout.py` forcing + clamp | shared `policy_core.py` |

So a difference between this row and the others cannot be attributed to any
single cause. It is included as **"what we had before"**, not as a controlled
comparison. The controlled test of the same *framing* is the `clf_*` arm,
which holds features, instances, splits and decision rule fixed.

## Removed

**DAgger is not restored** (`dagger.py`, `merge_dagger.py`), nor the RL
experiment (`rl_env.py`, `rl_ppo.py`), nor the old plotting and sweep
scripts. Only behavioural cloning, which is what the current arms do. The
`--dagger` flags have been stripped from `train.py` so it cannot be invoked.

## Files

| file | role |
|---|---|
| `code/extract_dataset.py` | teacher JSONs -> its own `dataset.npz` (23 + 6K features) |
| `code/model.py` | masked MLP: trunk + class head + charge-duration head |
| `code/train.py` | behavioural cloning, class-weighted cross-entropy |
| `code/rollout.py` | drives the simulator via the `external_policy` hook |
