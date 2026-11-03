# Player Market-Value Target Feasibility

This project checks whether the proposed prediction target, the next recorded
market-value adjustment, is suitable for modeling with the dcaribou
Transfermarkt dataset.

The target is defined as:

```text
delta_log_value = log(next market value) - log(current market value)
```

## Data source

The script downloads the official compressed `player_valuations` table from:

```text
https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/player_valuations.csv.gz
```

The source repository is:

```text
https://github.com/dcaribou/transfermarkt-datasets
```

The repository states that valuation rows are created when a player's market
value changes. The resulting target distribution is therefore conditional on a
recorded valuation change. It does not contain all player-time periods in which
the value remained unchanged.

## Run

From the `project` directory:

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\src\analyze_target_feasibility.py
```

## Train the three-model comparison

Install the packages in `requirements.txt`, then run:

```powershell
python .\src\train_value_models.py
```

The training script downloads `players` and `appearances` when they are
missing and compares three models on one chronological split:

- `M0_no_change`: predicts `delta_log_v = 0`.
- `M_long`: static player attributes, cumulative performance before the
  current valuation, and valuation context.
- `M_combined`: the same long-term model plus performance in the five
  appearances strictly before the current valuation date.

Training targets must be fully observed before the test start date. Rows whose
valuation interval crosses the train/test boundary are excluded. The script
saves fitted models in `models` and metrics, feature definitions, coefficients,
test predictions, and age/position subgroup errors in `output/models`.

The script downloads the raw file only when it is missing. It then creates the
following files in `output`:

- `data_summary.csv`: source and target data checks
- `target_distribution_summary.csv`: descriptive statistics
- `small_change_shares.csv`: zero and small-adjustment proportions
- `direction_summary.csv`: increase/decrease/zero proportions
- `yearly_summary.csv`: target distribution by next valuation year
- `target_histogram_bins.csv`: auditable histogram bins and counts
- `target_distribution.png`: four-panel distribution figure

## Dependencies

- pandas
- numpy
- Pillow
