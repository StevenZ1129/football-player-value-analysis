# Model training report

## Prediction task

The target is `delta_log_v = log(V_t+1) - log(V_t)`, where `V_t+1` is the
next recorded valuation for the same player.  Every performance feature is
calculated from appearances strictly before the current valuation date `t`.

## Models

- `M0_no_change`: predicts zero for every observation.
- `M_long`: static player attributes, cumulative prior performance, and context.
- `M_combined`: all `M_long` inputs plus performance in the five prior appearances.

Both learned models use Ridge regression with alpha 10.  They use
the same rows, chronological split, imputation, scaling, and categorical
encoding.  The recent-performance block is the only design difference.

## Chronological split

- Test period begins: 2023-06-26
- Training rows: 201,339
- Test rows: 53,600
- Boundary rows excluded: 9,763
- Test players: 11,205

A row is in training only when its next valuation was recorded before the test
cutoff.  A row is in testing only when its current valuation date is on or
after the cutoff.  This prevents training outcomes from crossing into the test
feature period.

## Main chronological-test results

| Model | MAE | RMSE | R-squared | Direction accuracy |
| --- | ---: | ---: | ---: | ---: |
| M0 no change | 0.1944 | 0.2925 | -0.0019 | 28.84% |
| Long-term | 0.1775 | 0.2579 | 0.2211 | 53.46% |
| Combined | 0.1762 | 0.2563 | 0.2306 | 54.23% |

The long-term model changes MAE by 8.67% relative to M0.  Adding the
five-match block changes MAE by 0.72% relative to the long-term
model.  Direction accuracy uses a plus/minus 0.01 delta-log
band for no material change.

## Interpretation limits

The target is conditional on another valuation record occurring.  The model
does not predict when Transfermarkt will issue that record.  Player profile
fields are snapshot attributes; current/highest market value and current club
from the player profile table are deliberately excluded because they can reveal
future information in historical rows.  The same five-match-eligible sample is
used for both learned models, so the comparison does not cover players without
five earlier recorded appearances.
