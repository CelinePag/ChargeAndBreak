# _archive/ — superseded runs from the val-selection protocol

Everything here was produced under the earlier design: models fitted on seeds
1-17, configurations CHOSEN on seeds 18-21, and only the chosen one reported
on test. Tags were also inconsistent (`base`, `gbt_v1`, `nn`, `nn_log1p`) and
did not say which arm they belonged to.

Superseded by the protocol in `ML/code/configs.py`: every configuration is an
independent model, fitted on seeds 1-19, early-stopped on 20-21, and reported
on the whole test batch. Kept only so the earlier numbers quoted in the
conversation remain traceable. **Do not cite these.**
