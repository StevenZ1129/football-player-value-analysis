# Target feasibility report

## Target definition

The target is the next recorded adjustment for the same player:

`delta_log_value = log(next market value) - log(current market value)`

## Data summary

- Raw valuation records: 656,301
- Clean valuation records: 656,300
- Usable consecutive-record targets: 614,772
- Players represented in the target: 40,465
- Source date range: 2000-01-20 to 2026-06-12
- Median days to the next valuation record: 168

## Distribution summary

- Exact zero adjustment: 31.07%
- Absolute delta log value below 0.01: 31.07%
- Absolute delta log value below 0.05: 31.45%
- Increase: 35.76%
- Decrease: 33.18%
- 1st percentile / median / 99th percentile: -0.693 / 0.000 / 1.609
- M0 no-change MAE: 0.267
- M0 no-change RMSE: 0.438

Among 574,307 rows with a previous adjustment, the Pearson correlation between consecutive adjustments is 0.255.

## Feasibility conclusion

The target has enough observations and contains both positive and negative changes, so
it is feasible for a regression comparison. The source contains a material share of consecutive records with no change. The proposed model
therefore predicts the magnitude and direction of the next recorded change, conditional
on a valuation record occurring. It does not predict whether Transfermarkt will update a
player or whether the value remains unchanged during periods with no valuation record.

The main analysis should retain the no-change M0 baseline, use a chronological split,
and report subgroup errors by age and position. If the intended claim includes all
player-time periods, the project should create a fixed monthly or quarterly panel with
carried-forward values and treat unchanged periods as zero adjustments. That is a
different target from the next recorded adjustment analyzed here.

If a direct regression model stays close to M0, a useful secondary diagnostic is to
separate the problem into zero versus nonzero adjustment and, conditional on a nonzero
adjustment, its direction and magnitude. This should remain a robustness analysis unless
the main regression comparison fails.
