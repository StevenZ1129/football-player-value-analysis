"""Train baseline, long-term, and combined player-value regression models.

The prediction target is the next recorded market-value adjustment:

    delta_log_v = log(V_{i,t+1}) - log(V_{i,t})

The long-term and combined models use the same observations, chronological
split, preprocessing, and Ridge regression estimator.  The combined model
differs only by adding performance from the five appearances strictly before
the current valuation date.  This makes the comparison an interpretable
ablation of the incremental value of recent form.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_target_feasibility import prepare_target


DATA_BASE_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
)
SOURCE_URLS = {
    "player_valuations": f"{DATA_BASE_URL}/player_valuations.csv.gz",
    "players": f"{DATA_BASE_URL}/players.csv.gz",
    "appearances": f"{DATA_BASE_URL}/appearances.csv.gz",
}

STATIC_NUMERIC = ["age_years", "height_in_cm"]
STATIC_CATEGORICAL = [
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
]
LONG_TERM_NUMERIC = [
    "career_matches",
    "career_minutes",
    "career_goals",
    "career_assists",
    "career_yellow_cards",
    "career_red_cards",
    "career_goals_per90",
    "career_assists_per90",
    "career_minutes_per_match",
]
CONTEXT_NUMERIC = [
    "log_market_value",
    "previous_delta_log_v",
    "days_since_previous_valuation",
    "valuation_count_prior",
    "has_previous_valuation",
]
CONTEXT_CATEGORICAL = [
    "player_club_domestic_competition_id",
    "valuation_month",
]
LAST_FIVE_NUMERIC = [
    "last5_minutes",
    "last5_goals",
    "last5_assists",
    "last5_yellow_cards",
    "last5_red_cards",
    "last5_goals_per90",
    "last5_assists_per90",
    "last5_span_days",
]

LONG_NUMERIC_FEATURES = STATIC_NUMERIC + LONG_TERM_NUMERIC + CONTEXT_NUMERIC
LONG_CATEGORICAL_FEATURES = STATIC_CATEGORICAL + CONTEXT_CATEGORICAL
COMBINED_NUMERIC_FEATURES = LONG_NUMERIC_FEATURES + LAST_FIVE_NUMERIC
COMBINED_CATEGORICAL_FEATURES = LONG_CATEGORICAL_FEATURES

APPEARANCE_STATS = [
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    parser = argparse.ArgumentParser(
        description=(
            "Train M0, long-term, and long-term plus last-five-match models "
            "for the next recorded market-value adjustment."
        )
    )
    parser.add_argument(
        "--valuations-path",
        type=Path,
        default=raw_dir / "player_valuations.csv.gz",
    )
    parser.add_argument(
        "--players-path",
        type=Path,
        default=raw_dir / "players.csv.gz",
    )
    parser.add_argument(
        "--appearances-path",
        type=Path,
        default=raw_dir / "appearances.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "models",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "models",
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default=None,
        help=(
            "First current-valuation date in the test set (YYYY-MM-DD). "
            "If omitted, the 80th percentile of eligible current dates is used."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Approximate chronological test share when no start date is supplied.",
    )
    parser.add_argument(
        "--min-prior-matches",
        type=int,
        default=5,
        help="Require this many appearances strictly before the valuation date.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Regularization strength shared by both learned models.",
    )
    parser.add_argument(
        "--min-category-frequency",
        type=int,
        default=50,
        help="Minimum training frequency for a separate one-hot category.",
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=0.01,
        help="Absolute delta-log band treated as no material change.",
    )
    args = parser.parse_args()
    if not 0 < args.test_fraction < 0.5:
        parser.error("--test-fraction must be between 0 and 0.5")
    if args.min_prior_matches < 1 or args.min_prior_matches > 5:
        parser.error("--min-prior-matches must be between 1 and 5")
    if args.ridge_alpha < 0:
        parser.error("--ridge-alpha must be nonnegative")
    return args


def download_if_missing(url: str, destination: Path) -> None:
    """Download a source file without replacing an existing nonempty raw file."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing raw file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "player-value-model-training/1.0"},
    )
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Saved raw file: {destination}")


def load_players(path: Path) -> pd.DataFrame:
    columns = [
        "player_id",
        "name",
        "date_of_birth",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
        "country_of_citizenship",
    ]
    players = pd.read_csv(path, compression="gzip", usecols=columns, low_memory=False)
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    players["date_of_birth"] = pd.to_datetime(
        players["date_of_birth"], errors="coerce"
    )
    players["height_in_cm"] = pd.to_numeric(
        players["height_in_cm"], errors="coerce"
    )
    players = players.dropna(subset=["player_id"]).copy()
    players["player_id"] = players["player_id"].astype("int64")
    players = players.drop_duplicates("player_id", keep="last")
    return players


def safe_per90(numerator: pd.Series, minutes: pd.Series) -> pd.Series:
    result = np.zeros(len(numerator), dtype="float64")
    valid = minutes.to_numpy(dtype="float64") > 0
    result[valid] = (
        numerator.to_numpy(dtype="float64")[valid]
        * 90.0
        / minutes.to_numpy(dtype="float64")[valid]
    )
    return pd.Series(result, index=numerator.index)


def load_appearance_snapshots(path: Path) -> pd.DataFrame:
    """Create cumulative and rolling-five features at every appearance date."""
    columns = ["appearance_id", "player_id", "date", *APPEARANCE_STATS]
    appearances = pd.read_csv(
        path,
        compression="gzip",
        usecols=columns,
        low_memory=False,
    )
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["player_id"] = pd.to_numeric(
        appearances["player_id"], errors="coerce"
    )
    appearances = appearances.dropna(subset=["player_id", "date"]).copy()
    appearances["player_id"] = appearances["player_id"].astype("int64")
    appearances = appearances.drop_duplicates("appearance_id", keep="last")

    for column in APPEARANCE_STATS:
        appearances[column] = (
            pd.to_numeric(appearances[column], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )

    appearances = appearances.sort_values(
        ["player_id", "date", "appearance_id"]
    ).reset_index(drop=True)
    grouped = appearances.groupby("player_id", sort=False)
    appearances["career_matches"] = grouped.cumcount().add(1).astype("int32")

    for statistic in APPEARANCE_STATS:
        career_name = f"career_{statistic.replace('_played', '')}"
        appearances[career_name] = grouped[statistic].cumsum()
        rolling = (
            grouped[statistic]
            .rolling(window=5, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        last_five_name = f"last5_{statistic.replace('_played', '')}"
        appearances[last_five_name] = rolling

    appearances["last5_matches_observed"] = np.minimum(
        appearances["career_matches"], 5
    ).astype("int8")
    appearances["last5_span_days"] = (
        appearances["date"] - grouped["date"].shift(4)
    ).dt.days
    appearances["last5_span_days"] = appearances["last5_span_days"].fillna(0.0)

    appearances["career_goals_per90"] = safe_per90(
        appearances["career_goals"], appearances["career_minutes"]
    )
    appearances["career_assists_per90"] = safe_per90(
        appearances["career_assists"], appearances["career_minutes"]
    )
    appearances["career_minutes_per_match"] = (
        appearances["career_minutes"] / appearances["career_matches"]
    )
    appearances["last5_goals_per90"] = safe_per90(
        appearances["last5_goals"], appearances["last5_minutes"]
    )
    appearances["last5_assists_per90"] = safe_per90(
        appearances["last5_assists"], appearances["last5_minutes"]
    )
    snapshot_columns = [
        "player_id",
        "date",
        "career_matches",
        "last5_matches_observed",
        *LONG_TERM_NUMERIC[1:],
        *LAST_FIVE_NUMERIC,
    ]
    snapshots = appearances[snapshot_columns].rename(
        columns={"date": "appearance_date"}
    )
    return snapshots


def add_valuation_features(
    values: pd.DataFrame, target: pd.DataFrame
) -> pd.DataFrame:
    values = values.copy()
    values["valuation_count_prior"] = (
        values.groupby("player_id", sort=False).cumcount().astype("int32")
    )
    values["days_since_previous_valuation"] = (
        values["date"] - values["previous_date"]
    ).dt.days
    values["log_market_value"] = np.log(values["market_value_in_eur"])
    values["has_previous_valuation"] = values["previous_date"].notna().astype("int8")
    values["valuation_month"] = values["date"].dt.month.astype("Int64").astype("string")
    latest_date_by_player = values.groupby("player_id")["date"].max()

    added_columns = [
        "player_id",
        "date",
        "valuation_count_prior",
        "days_since_previous_valuation",
        "log_market_value",
        "has_previous_valuation",
        "valuation_month",
    ]
    frame = target.drop(
        columns=[
            column
            for column in added_columns[2:]
            if column in target.columns
        ],
        errors="ignore",
    ).merge(values[added_columns], on=["player_id", "date"], how="left")
    frame["is_latest_recorded_target"] = (
        frame["next_date"] == frame["player_id"].map(latest_date_by_player)
    )
    return frame


def build_model_frame(
    valuations_path: Path,
    players_path: Path,
    appearances_path: Path,
    min_prior_matches: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    values, target, target_checks = prepare_target(valuations_path)
    target = add_valuation_features(values, target)
    players = load_players(players_path)
    snapshots = load_appearance_snapshots(appearances_path)

    left = target.sort_values(["date", "player_id"]).reset_index(drop=True)
    right = snapshots.sort_values(
        ["appearance_date", "player_id"]
    ).reset_index(drop=True)
    frame = pd.merge_asof(
        left,
        right,
        by="player_id",
        left_on="date",
        right_on="appearance_date",
        direction="backward",
        allow_exact_matches=False,
    )
    frame = frame.merge(players, on="player_id", how="left", validate="many_to_one")
    frame["age_years"] = (
        (frame["date"] - frame["date_of_birth"]).dt.days / 365.25
    )
    frame.loc[~frame["age_years"].between(14, 50), "age_years"] = np.nan

    before_filter_rows = len(frame)
    complete_history = (
        frame["last5_matches_observed"].fillna(0) >= min_prior_matches
    )
    frame = frame.loc[complete_history].copy()
    frame = frame.loc[np.isfinite(frame["delta_log_v"])].copy()

    leakage_appearance_rows = int(
        (frame["appearance_date"] >= frame["date"]).fillna(False).sum()
    )
    invalid_target_order_rows = int((frame["next_date"] <= frame["date"]).sum())
    if leakage_appearance_rows or invalid_target_order_rows:
        raise AssertionError(
            "Temporal leakage check failed: "
            f"appearance_rows={leakage_appearance_rows}, "
            f"target_order_rows={invalid_target_order_rows}"
        )

    categorical_columns = sorted(
        set(LONG_CATEGORICAL_FEATURES + COMBINED_CATEGORICAL_FEATURES)
    )
    for column in categorical_columns:
        frame[column] = frame[column].astype("object")
        frame.loc[frame[column].isna(), column] = np.nan

    numeric_columns = sorted(
        set(LONG_NUMERIC_FEATURES + COMBINED_NUMERIC_FEATURES)
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    checks: dict[str, object] = {
        **target_checks,
        "target_rows_before_match_history_filter": int(before_filter_rows),
        "modeling_rows": int(len(frame)),
        "modeling_unique_players": int(frame["player_id"].nunique()),
        "minimum_prior_matches": int(min_prior_matches),
        "minimum_modeling_date": frame["date"].min().date().isoformat(),
        "maximum_modeling_date": frame["date"].max().date().isoformat(),
        "appearance_feature_rows_on_or_after_valuation": leakage_appearance_rows,
        "targets_not_after_current_valuation": invalid_target_order_rows,
        "missing_player_profile_share": float(frame["date_of_birth"].isna().mean()),
    }
    return frame, checks


def choose_cutoff(
    frame: pd.DataFrame,
    test_start_date: str | None,
    test_fraction: float,
) -> pd.Timestamp:
    if test_start_date:
        cutoff = pd.to_datetime(test_start_date, errors="raise")
    else:
        cutoff = pd.Timestamp(frame["date"].quantile(1.0 - test_fraction))
    return cutoff.normalize()


def chronological_split(
    frame: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train_mask = frame["next_date"] < cutoff
    test_mask = frame["date"] >= cutoff
    boundary_mask = ~(train_mask | test_mask)
    train = frame.loc[train_mask].copy()
    test = frame.loc[test_mask].copy()
    if len(train) < 1_000 or len(test) < 1_000:
        raise ValueError(
            f"Chronological split at {cutoff.date()} is too small: "
            f"train={len(train):,}, test={len(test):,}."
        )
    if train["next_date"].max() >= cutoff:
        raise AssertionError("A training target is not fully observed before cutoff")
    if test["date"].min() < cutoff:
        raise AssertionError("A test feature date precedes cutoff")
    return train, test, int(boundary_mask.sum())


def make_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    alpha: float,
    min_category_frequency: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_category_frequency,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "regressor",
                Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=10_000),
            ),
        ]
    )


def direction(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.select(
        [values > threshold, values < -threshold],
        ["increase", "decrease"],
        default="no_material_change",
    )


def regression_metrics(
    model_name: str,
    split_name: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
    transformed_feature_count: int,
) -> dict[str, object]:
    errors = observed - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    denominator = float(np.sum(np.square(observed - observed.mean())))
    r2 = (
        float(1.0 - np.sum(np.square(errors)) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "model": model_name,
        "evaluation_split": split_name,
        "observations": int(len(observed)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": float(
            np.mean(direction(observed, threshold) == direction(predicted, threshold))
        ),
        "mean_prediction": float(np.mean(predicted)),
        "mean_observed": float(np.mean(observed)),
        "transformed_feature_count": int(transformed_feature_count),
    }


def feature_definition_frame() -> pd.DataFrame:
    meanings = {
        "age_years": "Age at the current valuation date.",
        "height_in_cm": "Player height from the profile table.",
        "position": "Broad playing position.",
        "sub_position": "Detailed playing position.",
        "foot": "Preferred foot.",
        "country_of_citizenship": "Player citizenship from the profile table.",
        "log_market_value": "Log of market value at the current valuation date.",
        "previous_delta_log_v": "Previous recorded log market-value adjustment.",
        "days_since_previous_valuation": "Days since the preceding valuation record.",
        "valuation_count_prior": "Number of valuation records before the current one.",
        "has_previous_valuation": "Indicator that a preceding valuation exists.",
        "player_club_domestic_competition_id": "Domestic competition at valuation t.",
        "valuation_month": "Calendar month of valuation t.",
        "career_matches": "Appearances strictly before valuation t.",
        "career_minutes": "Cumulative minutes strictly before valuation t.",
        "career_goals": "Cumulative goals strictly before valuation t.",
        "career_assists": "Cumulative assists strictly before valuation t.",
        "career_yellow_cards": "Cumulative yellow cards before valuation t.",
        "career_red_cards": "Cumulative red cards before valuation t.",
        "career_goals_per90": "Cumulative goals per 90 minutes before valuation t.",
        "career_assists_per90": "Cumulative assists per 90 minutes before valuation t.",
        "career_minutes_per_match": "Cumulative average minutes per appearance.",
        "last5_minutes": "Minutes in the five appearances before valuation t.",
        "last5_goals": "Goals in the five appearances before valuation t.",
        "last5_assists": "Assists in the five appearances before valuation t.",
        "last5_yellow_cards": "Yellow cards in the five prior appearances.",
        "last5_red_cards": "Red cards in the five prior appearances.",
        "last5_goals_per90": "Goals per 90 in the five prior appearances.",
        "last5_assists_per90": "Assists per 90 in the five prior appearances.",
        "last5_span_days": "Days covered by the five prior appearances.",
    }
    all_features = list(
        dict.fromkeys(COMBINED_NUMERIC_FEATURES + COMBINED_CATEGORICAL_FEATURES)
    )
    rows = []
    for feature in all_features:
        if feature in STATIC_NUMERIC + STATIC_CATEGORICAL:
            block = "static"
        elif feature in LONG_TERM_NUMERIC:
            block = "long_term"
        elif feature in CONTEXT_NUMERIC + CONTEXT_CATEGORICAL:
            block = "context"
        else:
            block = "last_five_matches"
        rows.append(
            {
                "feature": feature,
                "block": block,
                "used_in_long_model": feature
                in LONG_NUMERIC_FEATURES + LONG_CATEGORICAL_FEATURES,
                "used_in_combined_model": True,
                "definition": meanings[feature],
            }
        )
    return pd.DataFrame(rows)


def coefficient_frame(model_name: str, pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["regressor"].coef_
    frame = pd.DataFrame(
        {
            "model": model_name,
            "transformed_feature": names,
            "coefficient": coefficients,
        }
    )
    frame["absolute_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("absolute_coefficient", ascending=False)


def subgroup_metric_frame(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    minimum_size: int = 50,
) -> pd.DataFrame:
    """Summarize errors for age, position, and age-by-position views."""
    working = test[["age_years", "position", "delta_log_v"]].copy()
    working["age_group"] = pd.cut(
        working["age_years"],
        bins=[-np.inf, 20, 24, 29, np.inf],
        labels=["20_or_younger", "21_to_24", "25_to_29", "30_or_older"],
    ).astype("object")
    working.loc[working["age_group"].isna(), "age_group"] = "unknown"
    working["position"] = working["position"].fillna("Unknown").astype(str)
    for model_name, predicted in predictions.items():
        working[f"prediction__{model_name}"] = predicted

    rows: list[dict[str, object]] = []
    group_specs = {
        "age_group": ["age_group"],
        "position": ["position"],
        "age_by_position": ["age_group", "position"],
    }
    for subgroup_type, group_columns in group_specs.items():
        grouper: str | list[str] = (
            group_columns[0] if len(group_columns) == 1 else group_columns
        )
        for keys, group in working.groupby(grouper, dropna=False, observed=True):
            if len(group) < minimum_size:
                continue
            key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
            key_lookup = dict(zip(group_columns, key_values))
            observed = group["delta_log_v"].to_numpy(dtype="float64")
            for model_name in predictions:
                predicted = group[f"prediction__{model_name}"].to_numpy(
                    dtype="float64"
                )
                error = observed - predicted
                rows.append(
                    {
                        "subgroup_type": subgroup_type,
                        "age_group": key_lookup.get("age_group", "all"),
                        "position": key_lookup.get("position", "all"),
                        "model": model_name,
                        "observations": int(len(group)),
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": float(math.sqrt(np.mean(np.square(error)))),
                        "mean_error": float(np.mean(error)),
                    }
                )
    return pd.DataFrame(rows)


def save_metadata(
    path: Path,
    cutoff: pd.Timestamp,
    checks: dict[str, object],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "target": "delta_log_v = log(next_market_value) - log(current_market_value)",
        "prediction_event": "next recorded market-value adjustment",
        "test_start_date": cutoff.date().isoformat(),
        "split_rule": {
            "train": "next_date < test_start_date",
            "test": "current valuation date >= test_start_date",
            "boundary_rows": "excluded",
        },
        "estimator": "Ridge regression",
        "ridge_alpha": args.ridge_alpha,
        "minimum_prior_matches": args.min_prior_matches,
        "direction_threshold": args.direction_threshold,
        "long_numeric_features": LONG_NUMERIC_FEATURES,
        "long_categorical_features": LONG_CATEGORICAL_FEATURES,
        "combined_added_features": LAST_FIVE_NUMERIC,
        "source_urls": SOURCE_URLS,
        "data_checks": checks,
    }
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def model_feature_count(pipeline: Pipeline) -> int:
    return int(len(pipeline.named_steps["preprocess"].get_feature_names_out()))


def evaluate_split(
    split_name: str,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    threshold: float,
    feature_counts: dict[str, int],
) -> list[dict[str, object]]:
    observed = frame["delta_log_v"].to_numpy(dtype="float64")
    return [
        regression_metrics(
            model_name,
            split_name,
            observed,
            prediction,
            threshold,
            feature_counts[model_name],
        )
        for model_name, prediction in predictions.items()
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for name, path in (
        ("player_valuations", args.valuations_path),
        ("players", args.players_path),
        ("appearances", args.appearances_path),
    ):
        download_if_missing(SOURCE_URLS[name], path)

    print("Building leakage-safe valuation and appearance features...")
    frame, checks = build_model_frame(
        args.valuations_path,
        args.players_path,
        args.appearances_path,
        args.min_prior_matches,
    )
    cutoff = choose_cutoff(frame, args.test_start_date, args.test_fraction)
    train, test, boundary_rows = chronological_split(frame, cutoff)
    checks.update(
        {
            "test_start_date": cutoff.date().isoformat(),
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "boundary_rows_excluded": int(boundary_rows),
            "training_unique_players": int(train["player_id"].nunique()),
            "test_unique_players": int(test["player_id"].nunique()),
            "latest_recorded_test_rows": int(
                test["is_latest_recorded_target"].sum()
            ),
        }
    )

    long_model = make_regression_pipeline(
        LONG_NUMERIC_FEATURES,
        LONG_CATEGORICAL_FEATURES,
        args.ridge_alpha,
        args.min_category_frequency,
    )
    combined_model = make_regression_pipeline(
        COMBINED_NUMERIC_FEATURES,
        COMBINED_CATEGORICAL_FEATURES,
        args.ridge_alpha,
        args.min_category_frequency,
    )
    y_train = train["delta_log_v"].to_numpy(dtype="float64")

    print(f"Training M_long on {len(train):,} rows...")
    long_model.fit(
        train[LONG_NUMERIC_FEATURES + LONG_CATEGORICAL_FEATURES], y_train
    )
    print(f"Training M_combined on {len(train):,} rows...")
    combined_model.fit(
        train[COMBINED_NUMERIC_FEATURES + COMBINED_CATEGORICAL_FEATURES],
        y_train,
    )

    long_prediction = long_model.predict(
        test[LONG_NUMERIC_FEATURES + LONG_CATEGORICAL_FEATURES]
    )
    combined_prediction = combined_model.predict(
        test[COMBINED_NUMERIC_FEATURES + COMBINED_CATEGORICAL_FEATURES]
    )
    predictions = {
        "M0_no_change": np.zeros(len(test), dtype="float64"),
        "M_long": np.asarray(long_prediction, dtype="float64"),
        "M_combined": np.asarray(combined_prediction, dtype="float64"),
    }
    feature_counts = {
        "M0_no_change": 0,
        "M_long": model_feature_count(long_model),
        "M_combined": model_feature_count(combined_model),
    }

    metric_rows = evaluate_split(
        "chronological_test",
        test,
        predictions,
        args.direction_threshold,
        feature_counts,
    )
    latest_mask = test["is_latest_recorded_target"].to_numpy(dtype=bool)
    if latest_mask.any():
        latest_predictions = {
            name: values[latest_mask] for name, values in predictions.items()
        }
        metric_rows.extend(
            evaluate_split(
                "latest_recorded_target_per_player",
                test.loc[latest_mask],
                latest_predictions,
                args.direction_threshold,
                feature_counts,
            )
        )
    metrics = pd.DataFrame(metric_rows)
    m0_by_split = metrics.loc[metrics["model"] == "M0_no_change", [
        "evaluation_split",
        "mae",
        "rmse",
    ]].rename(columns={"mae": "m0_mae", "rmse": "m0_rmse"})
    metrics = metrics.merge(m0_by_split, on="evaluation_split", how="left")
    metrics["mae_improvement_vs_m0"] = 1.0 - metrics["mae"] / metrics["m0_mae"]
    metrics["rmse_improvement_vs_m0"] = 1.0 - metrics["rmse"] / metrics["m0_rmse"]
    metrics = metrics.drop(columns=["m0_mae", "m0_rmse"])

    prediction_output = test[
        [
            "player_id",
            "name",
            "position",
            "player_club_domestic_competition_id",
            "date",
            "next_date",
            "market_value_in_eur",
            "next_market_value_eur",
            "delta_log_v",
            "is_latest_recorded_target",
            "career_matches",
            "last5_minutes",
            "last5_goals",
            "last5_assists",
        ]
    ].copy()
    prediction_output["prediction_m0"] = predictions["M0_no_change"]
    prediction_output["prediction_long"] = predictions["M_long"]
    prediction_output["prediction_combined"] = predictions["M_combined"]
    prediction_output["observed_direction"] = direction(
        prediction_output["delta_log_v"].to_numpy(dtype="float64"),
        args.direction_threshold,
    )
    prediction_output["combined_predicted_direction"] = direction(
        prediction_output["prediction_combined"].to_numpy(dtype="float64"),
        args.direction_threshold,
    )

    coefficients = pd.concat(
        [
            coefficient_frame("M_long", long_model),
            coefficient_frame("M_combined", combined_model),
        ],
        ignore_index=True,
    )
    subgroup_metrics = subgroup_metric_frame(test, predictions)
    data_summary = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in checks.items()]
    )

    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    data_summary.to_csv(args.output_dir / "model_data_summary.csv", index=False)
    feature_definition_frame().to_csv(
        args.output_dir / "feature_definitions.csv", index=False
    )
    coefficients.to_csv(args.output_dir / "standardized_coefficients.csv", index=False)
    subgroup_metrics.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    prediction_output.to_csv(
        args.output_dir / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    joblib.dump(long_model, args.model_dir / "long_term_ridge.joblib", compress=3)
    joblib.dump(combined_model, args.model_dir / "combined_ridge.joblib", compress=3)
    save_metadata(
        args.model_dir / "model_metadata.json", cutoff, checks, args
    )

    print(metrics.to_string(index=False))
    print(f"Model outputs written to: {args.output_dir}")
    print(f"Trained model files written to: {args.model_dir}")


if __name__ == "__main__":
    main()
