"""Analyze feasibility of the next recorded football market-value adjustment.

The script downloads the dcaribou Transfermarkt player valuation table, keeps
the raw file unchanged, constructs consecutive-player valuation targets, and
writes auditable CSV summaries plus a PNG distribution figure.
"""

from __future__ import annotations

import argparse
import math
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/"
    "data/player_valuations.csv.gz"
)
REQUIRED_COLUMNS = {"date", "player_id", "market_value_in_eur"}
ZERO_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize the distribution of the next market-value adjustment."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Source CSV.GZ URL")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=project_root / "data" / "raw" / "player_valuations.csv.gz",
        help="Location of the downloaded raw CSV.GZ file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output",
        help="Directory for summaries and plots",
    )
    return parser.parse_args()


def download_if_missing(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing raw file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "player-value-feasibility/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Saved raw file: {destination}")


def metric_frame(items: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(items, columns=["metric", "value"])


def prepare_target(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(raw_path, compression="gzip", low_memory=False)
    missing_columns = REQUIRED_COLUMNS.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    checks: dict[str, object] = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(raw.columns)),
        "raw_unique_players": int(raw["player_id"].nunique(dropna=True)),
        "raw_missing_player_id": int(raw["player_id"].isna().sum()),
        "raw_missing_date": int(raw["date"].isna().sum()),
        "raw_missing_market_value": int(raw["market_value_in_eur"].isna().sum()),
        "raw_exact_duplicate_rows": int(raw.duplicated().sum()),
    }

    values = raw.copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["player_id"] = pd.to_numeric(values["player_id"], errors="coerce")
    values["market_value_in_eur"] = pd.to_numeric(
        values["market_value_in_eur"], errors="coerce"
    )

    checks["invalid_parsed_date"] = int(values["date"].isna().sum())
    checks["invalid_or_missing_player_id"] = int(values["player_id"].isna().sum())
    checks["invalid_or_missing_market_value"] = int(
        values["market_value_in_eur"].isna().sum()
    )
    checks["nonpositive_market_value"] = int(
        (values["market_value_in_eur"] <= 0).fillna(False).sum()
    )

    values = values.dropna(subset=["date", "player_id", "market_value_in_eur"])
    values = values.loc[values["market_value_in_eur"] > 0].copy()
    values["player_id"] = values["player_id"].astype("int64")
    values = values.drop_duplicates()

    value_counts = values.groupby(["player_id", "date"])[
        "market_value_in_eur"
    ].nunique()
    conflict_keys = value_counts.loc[value_counts > 1].index
    checks["conflicting_player_date_keys"] = int(len(conflict_keys))
    if len(conflict_keys):
        index = pd.MultiIndex.from_frame(values[["player_id", "date"]])
        values = values.loc[~index.isin(conflict_keys)].copy()

    same_key_duplicates = values.duplicated(["player_id", "date"], keep="first")
    checks["same_value_player_date_duplicates_removed"] = int(
        same_key_duplicates.sum()
    )
    values = values.loc[~same_key_duplicates].copy()
    values = values.sort_values(["player_id", "date"]).reset_index(drop=True)

    grouped = values.groupby("player_id", sort=False)
    values["previous_date"] = grouped["date"].shift(1)
    values["previous_market_value_eur"] = grouped["market_value_in_eur"].shift(1)
    values["next_date"] = grouped["date"].shift(-1)
    values["next_market_value_eur"] = grouped["market_value_in_eur"].shift(-1)

    values["previous_delta_log_v"] = np.log(
        values["market_value_in_eur"] / values["previous_market_value_eur"]
    )
    values["delta_log_v"] = np.log(
        values["next_market_value_eur"] / values["market_value_in_eur"]
    )
    values["delta_pct"] = (
        values["next_market_value_eur"] / values["market_value_in_eur"] - 1.0
    )
    values["days_to_next_valuation"] = (
        values["next_date"] - values["date"]
    ).dt.days

    target = values.dropna(
        subset=["next_date", "next_market_value_eur", "delta_log_v"]
    ).copy()
    target["direction"] = np.select(
        [
            target["delta_log_v"] > ZERO_TOLERANCE,
            target["delta_log_v"] < -ZERO_TOLERANCE,
        ],
        ["increase", "decrease"],
        default="zero",
    )
    target["next_valuation_year"] = target["next_date"].dt.year.astype("int64")

    checks.update(
        {
            "clean_valuation_rows": int(len(values)),
            "clean_unique_players": int(values["player_id"].nunique()),
            "clean_min_date": values["date"].min().date().isoformat(),
            "clean_max_date": values["date"].max().date().isoformat(),
            "usable_target_rows": int(len(target)),
            "target_unique_players": int(target["player_id"].nunique()),
        }
    )
    return values, target, checks


def build_summaries(
    values: pd.DataFrame, target: pd.DataFrame, checks: dict[str, object]
) -> dict[str, pd.DataFrame]:
    delta = target["delta_log_v"]
    abs_delta = delta.abs()
    quantiles = delta.quantile(
        [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    )

    target_statistics = metric_frame(
        [
            ("count", int(delta.count())),
            ("mean_delta_log_v", float(delta.mean())),
            ("std_delta_log_v", float(delta.std())),
            *[(f"quantile_{q:.2f}", float(value)) for q, value in quantiles.items()],
            ("mean_absolute_delta_log_v", float(abs_delta.mean())),
            ("m0_no_change_mae", float(abs_delta.mean())),
            ("m0_no_change_rmse", float(np.sqrt(np.mean(np.square(delta))))),
            ("median_days_to_next_valuation", float(target["days_to_next_valuation"].median())),
            ("mean_days_to_next_valuation", float(target["days_to_next_valuation"].mean())),
            ("p90_days_to_next_valuation", float(target["days_to_next_valuation"].quantile(0.90))),
        ]
    )

    thresholds = [0.0, 0.01, 0.025, 0.05, 0.10]
    small_change_rows = []
    for threshold in thresholds:
        if threshold == 0:
            count = int((abs_delta <= ZERO_TOLERANCE).sum())
        else:
            count = int((abs_delta < threshold).sum())
        small_change_rows.append(
            {
                "absolute_delta_log_threshold": threshold,
                "count": count,
                "share": count / len(target),
            }
        )
    small_change_shares = pd.DataFrame(small_change_rows)

    direction_summary = (
        target["direction"]
        .value_counts(dropna=False)
        .reindex(["increase", "decrease", "zero"], fill_value=0)
        .rename_axis("direction")
        .reset_index(name="count")
    )
    direction_summary["share"] = direction_summary["count"] / len(target)

    yearly_summary = (
        target.groupby("next_valuation_year", as_index=False)
        .agg(
            observations=("delta_log_v", "size"),
            unique_players=("player_id", "nunique"),
            mean_delta_log_v=("delta_log_v", "mean"),
            median_delta_log_v=("delta_log_v", "median"),
            mean_absolute_delta_log_v=("delta_log_v", lambda x: x.abs().mean()),
            increase_share=(
                "delta_log_v",
                lambda x: (x > ZERO_TOLERANCE).mean(),
            ),
            decrease_share=(
                "delta_log_v",
                lambda x: (x < -ZERO_TOLERANCE).mean(),
            ),
            zero_share=(
                "delta_log_v",
                lambda x: (x.abs() <= ZERO_TOLERANCE).mean(),
            ),
            median_days_to_next_valuation=("days_to_next_valuation", "median"),
        )
        .sort_values("next_valuation_year")
    )

    paired = target.dropna(subset=["previous_delta_log_v"]).copy()
    if len(paired):
        momentum_error = paired["delta_log_v"] - paired["previous_delta_log_v"]
        pearson = paired["delta_log_v"].corr(paired["previous_delta_log_v"])
        rank_correlation = paired["delta_log_v"].rank().corr(
            paired["previous_delta_log_v"].rank()
        )
        persistence = metric_frame(
            [
                ("paired_observations", int(len(paired))),
                ("consecutive_change_pearson_correlation", float(pearson)),
                ("consecutive_change_rank_correlation", float(rank_correlation)),
                ("previous_change_baseline_mae", float(momentum_error.abs().mean())),
                ("previous_change_baseline_rmse", float(np.sqrt(np.mean(momentum_error**2)))),
            ]
        )
    else:
        persistence = metric_frame([("paired_observations", 0)])

    lower = float(delta.quantile(0.01))
    upper = float(delta.quantile(0.99))
    bin_edges = np.linspace(lower, upper, 61)
    counts, edges = np.histogram(delta.clip(lower, upper), bins=bin_edges)
    histogram_bins = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
        }
    )

    data_summary = metric_frame(list(checks.items()))
    player_record_summary = metric_frame(
        [
            ("median_valuation_records_per_player", float(values.groupby("player_id").size().median())),
            ("p90_valuation_records_per_player", float(values.groupby("player_id").size().quantile(0.90))),
            ("max_valuation_records_per_player", int(values.groupby("player_id").size().max())),
        ]
    )
    data_summary = pd.concat([data_summary, player_record_summary], ignore_index=True)

    return {
        "data_summary": data_summary,
        "target_distribution_summary": target_statistics,
        "small_change_shares": small_change_shares,
        "direction_summary": direction_summary,
        "yearly_summary": yearly_summary,
        "target_histogram_bins": histogram_bins,
        "persistence_summary": persistence,
    }


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    plot = (left + 70, top + 55, right - 24, bottom - 58)
    x0, y0, x1, y1 = plot
    draw.line((x0, y1, x1, y1), fill="#374151", width=2)
    draw.line((x0, y0, x0, y1), fill="#374151", width=2)
    return plot


def draw_histogram(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: np.ndarray,
    title: str,
    x_label: str,
    color: str,
    clip_quantiles: tuple[float, float],
    bins: int = 45,
    zero_line: bool = False,
) -> None:
    title_font = load_font(23, bold=True)
    label_font = load_font(17)
    tick_font = load_font(14)
    left, top, right, bottom = box
    draw.text((left + 12, top + 8), title, font=title_font, fill="#111827")
    x0, y0, x1, y1 = draw_axes(draw, box)

    finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, clip_quantiles)
    clipped = np.clip(finite, low, high)
    counts, edges = np.histogram(clipped, bins=bins, range=(low, high))
    max_count = max(int(counts.max()), 1)
    bar_width = (x1 - x0) / bins
    for index, count in enumerate(counts):
        bar_left = x0 + index * bar_width + 1
        bar_right = x0 + (index + 1) * bar_width - 1
        bar_top = y1 - (count / max_count) * (y1 - y0)
        draw.rectangle((bar_left, bar_top, bar_right, y1), fill=color)

    if zero_line and low <= 0 <= high:
        zero_x = x0 + (0 - low) / (high - low) * (x1 - x0)
        draw.line((zero_x, y0, zero_x, y1), fill="#B91C1C", width=3)
        draw.text((zero_x + 5, y0 + 4), "0", font=tick_font, fill="#B91C1C")

    draw.text((x0, y1 + 10), f"{low:.3f}", font=tick_font, fill="#374151")
    high_text = f"{high:.3f}"
    high_width = draw.textbbox((0, 0), high_text, font=tick_font)[2]
    draw.text((x1 - high_width, y1 + 10), high_text, font=tick_font, fill="#374151")
    label_width = draw.textbbox((0, 0), x_label, font=label_font)[2]
    draw.text(((x0 + x1 - label_width) / 2, bottom - 31), x_label, font=label_font, fill="#111827")
    draw.text((x0 + 8, y0 + 8), f"n={len(finite):,}", font=tick_font, fill="#4B5563")


def draw_direction_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    direction_summary: pd.DataFrame,
) -> None:
    title_font = load_font(23, bold=True)
    label_font = load_font(17)
    tick_font = load_font(14)
    left, top, right, bottom = box
    draw.text((left + 12, top + 8), "Direction of next adjustment", font=title_font, fill="#111827")
    x0, y0, x1, y1 = draw_axes(draw, box)
    colors = {"increase": "#2563EB", "decrease": "#D97706", "zero": "#9CA3AF"}
    rows = direction_summary.set_index("direction")
    directions = ["increase", "decrease", "zero"]
    slot = (x1 - x0) / len(directions)
    for index, direction in enumerate(directions):
        share = float(rows.loc[direction, "share"])
        bar_left = x0 + index * slot + slot * 0.22
        bar_right = x0 + (index + 1) * slot - slot * 0.22
        bar_top = y1 - share * (y1 - y0)
        draw.rectangle((bar_left, bar_top, bar_right, y1), fill=colors[direction])
        pct = f"{share:.1%}"
        pct_width = draw.textbbox((0, 0), pct, font=label_font)[2]
        draw.text(((bar_left + bar_right - pct_width) / 2, bar_top - 27), pct, font=label_font, fill="#111827")
        label_width = draw.textbbox((0, 0), direction, font=tick_font)[2]
        draw.text(((bar_left + bar_right - label_width) / 2, y1 + 12), direction, font=tick_font, fill="#374151")
    draw.text((x0 + 8, y0 + 8), "Share of target rows", font=tick_font, fill="#4B5563")


def make_distribution_figure(
    target: pd.DataFrame, direction_summary: pd.DataFrame, output_path: Path
) -> None:
    width, height = 1600, 1100
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    subtitle_font = load_font(18)
    draw.text((55, 24), "Next recorded market-value adjustment", font=title_font, fill="#111827")
    draw.text(
        (55, 66),
        "dcaribou player_valuations; consecutive observations within player",
        font=subtitle_font,
        fill="#4B5563",
    )

    boxes = [
        (40, 105, 790, 565),
        (810, 105, 1560, 565),
        (40, 590, 790, 1055),
        (810, 590, 1560, 1055),
    ]
    for box in boxes:
        draw.rounded_rectangle(box, radius=12, outline="#D1D5DB", width=2, fill="#FFFFFF")

    draw_histogram(
        draw,
        boxes[0],
        target["delta_log_v"].to_numpy(dtype=float),
        "Distribution of delta log value",
        "delta log value (1st-99th percentile)",
        "#4F81BD",
        (0.01, 0.99),
        zero_line=True,
    )
    draw_histogram(
        draw,
        boxes[1],
        target["delta_log_v"].abs().to_numpy(dtype=float),
        "Magnitude of adjustment",
        "absolute delta log value (0-99th percentile)",
        "#70AD47",
        (0.00, 0.99),
    )
    draw_direction_bars(draw, boxes[2], direction_summary)
    draw_histogram(
        draw,
        boxes[3],
        target["days_to_next_valuation"].to_numpy(dtype=float),
        "Time to next valuation record",
        "days (0-99th percentile)",
        "#8064A2",
        (0.00, 0.99),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def scalar(summary: pd.DataFrame, metric: str) -> float:
    value = summary.loc[summary["metric"] == metric, "value"]
    if value.empty:
        return float("nan")
    return float(value.iloc[0])


def write_report(
    output_path: Path,
    summaries: dict[str, pd.DataFrame],
) -> None:
    data_summary = summaries["data_summary"]
    target_summary = summaries["target_distribution_summary"]
    small = summaries["small_change_shares"]
    directions = summaries["direction_summary"].set_index("direction")
    persistence = summaries["persistence_summary"]

    data_lookup = dict(zip(data_summary["metric"], data_summary["value"]))
    target_rows = int(float(data_lookup["usable_target_rows"]))
    unique_players = int(float(data_lookup["target_unique_players"]))
    zero_share = float(
        small.loc[small["absolute_delta_log_threshold"] == 0.0, "share"].iloc[0]
    )
    one_pct_share = float(
        small.loc[small["absolute_delta_log_threshold"] == 0.01, "share"].iloc[0]
    )
    five_pct_share = float(
        small.loc[small["absolute_delta_log_threshold"] == 0.05, "share"].iloc[0]
    )
    increase_share = float(directions.loc["increase", "share"])
    decrease_share = float(directions.loc["decrease", "share"])
    median_days = scalar(target_summary, "median_days_to_next_valuation")
    m0_mae = scalar(target_summary, "m0_no_change_mae")
    m0_rmse = scalar(target_summary, "m0_no_change_rmse")
    q01 = scalar(target_summary, "quantile_0.01")
    median = scalar(target_summary, "quantile_0.50")
    q99 = scalar(target_summary, "quantile_0.99")

    paired = int(scalar(persistence, "paired_observations"))
    if paired:
        correlation = scalar(persistence, "consecutive_change_pearson_correlation")
        persistence_text = (
            f"Among {paired:,} rows with a previous adjustment, the Pearson "
            f"correlation between consecutive adjustments is {correlation:.3f}."
        )
    else:
        persistence_text = "There were no rows available for a persistence check."

    if zero_share < 0.01:
        zero_interpretation = (
            "Exact zero adjustments are almost absent because the source table records "
            "valuation events when market value changes."
        )
    else:
        zero_interpretation = (
            "The source contains a material share of consecutive records with no change."
        )

    report = f"""# Target feasibility report

## Target definition

The target is the next recorded adjustment for the same player:

`delta_log_value = log(next market value) - log(current market value)`

## Data summary

- Raw valuation records: {int(float(data_lookup['raw_rows'])):,}
- Clean valuation records: {int(float(data_lookup['clean_valuation_rows'])):,}
- Usable consecutive-record targets: {target_rows:,}
- Players represented in the target: {unique_players:,}
- Source date range: {data_lookup['clean_min_date']} to {data_lookup['clean_max_date']}
- Median days to the next valuation record: {median_days:,.0f}

## Distribution summary

- Exact zero adjustment: {zero_share:.2%}
- Absolute delta log value below 0.01: {one_pct_share:.2%}
- Absolute delta log value below 0.05: {five_pct_share:.2%}
- Increase: {increase_share:.2%}
- Decrease: {decrease_share:.2%}
- 1st percentile / median / 99th percentile: {q01:.3f} / {median:.3f} / {q99:.3f}
- M0 no-change MAE: {m0_mae:.3f}
- M0 no-change RMSE: {m0_rmse:.3f}

{persistence_text}

## Feasibility conclusion

The target has enough observations and contains both positive and negative changes, so
it is feasible for a regression comparison. {zero_interpretation} The proposed model
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
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_if_missing(args.url, args.raw_path)

    values, target, checks = prepare_target(args.raw_path)
    summaries = build_summaries(values, target, checks)
    for name, frame in summaries.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)

    make_distribution_figure(
        target,
        summaries["direction_summary"],
        args.output_dir / "target_distribution.png",
    )
    write_report(args.output_dir / "target_feasibility_report.md", summaries)

    print(f"Analyzed {len(target):,} consecutive valuation targets.")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
