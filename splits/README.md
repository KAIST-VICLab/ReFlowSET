# Frozen splits

These are the exact lists every number in the paper was produced on. Only the EO
side is stored; the SAR path is derived from it, so the two can never drift apart:

| Dataset | Rule |
|---|---|
| QXS-SAROPT | `opt_256_oc_0.2/` → `sar_256_oc_0.2/` |
| SAR2Opt | the EO directory's last character is replaced by `A` (`testB` → `testA`) |

| List | Items |
|---|---|
| `qxs-saropt/train_eo_list.txt` | 16,001 |
| `qxs-saropt/test_eo_list.txt` | 3,999 |
| `sar2opt/train_eo_list.txt` | 1,450 |
| `sar2opt/test_eo_list.txt` | 627 |

Paths are relative to the dataset root. No imagery is included.

**There is no held-out validation split.** Training monitors a sample drawn from
the training data; it is used to watch the loss, never for model selection. The
released checkpoints are the last step of a fixed-length schedule, not a best-of.
