"""
data_preprocessing.py

Preprocessing pipeline for CMAPSS FD001 dataset.

Pipeline:
    data/raw/  ->  data/processed/

Steps:
    1. Load raw train + test files (space-separated, no header, txt files)
    2. Assign column names (unit_id, cycle, op_settings, sensors)
    3. Drop constant / near-zero-variance sensors
    4. Compute RUL (Remaining Useful Life) for training data
    5. Create binary proxy anomaly labels (last N cycles = anomalous)
    6. Scale sensor readings using MinMaxScaler
    7. Save processed train + test splits to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import yaml

# ─────────────────────────────────────────────
# CMAPSS column names (no header in raw files)
# ─────────────────────────────────────────────
COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors known to be constant or near-constant in FD001 (safe to drop)
# These carry no signal for anomaly detection
SENSORS_TO_DROP = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def load_raw(filepath: Path) -> pd.DataFrame:
    """
    Load a raw CMAPSS .txt file (space-separated, no header).
    Returns a cleaned DataFrame with proper column names.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )
    # Drop trailing NaN columns that sometimes appear
    df.dropna(axis=1, how="all", inplace=True)
    return df


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each row in training data.

    RUL = max_cycle_for_that_engine - current_cycle

    This tells us how many cycles remain before failure.
    Lower RUL = closer to failure = more anomalous.
    """
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_proxy_labels(df: pd.DataFrame, anomaly_threshold: int = 30) -> pd.DataFrame:
    """
    Create binary proxy anomaly labels based on RUL.

    Label logic:
        RUL <= anomaly_threshold  ->  anomaly = 1  (degraded / near failure)
        RUL >  anomaly_threshold  ->  anomaly = 0  (healthy)

    Default threshold: 30 cycles before failure.
    This is a well-established heuristic for CMAPSS.

    The threshold is configurable via configs/data.yaml.
    """
    df["anomaly_label"] = (df["RUL"] <= anomaly_threshold).astype(int)
    return df


def drop_low_variance_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove sensors that are constant or near-constant in FD001.
    These add noise without contributing signal.
    """
    cols_to_drop = [c for c in SENSORS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of sensor column names present in the DataFrame.
    Excludes metadata and label columns.
    """
    non_sensor_cols = {"unit_id", "cycle", "RUL", "anomaly_label",
                       "op_setting_1", "op_setting_2", "op_setting_3"}
    return [c for c in df.columns if c not in non_sensor_cols]


def scale_sensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler_save_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit MinMaxScaler on training sensors only, then transform both splits.

    IMPORTANT:
        Scaler is fit on TRAIN only to prevent data leakage.
        Same scaler is then applied to TEST.
        Scaler is saved to artifacts/scalers/ for use at inference time.
    """
    sensor_cols = get_sensor_columns(train_df)

    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    # Save scaler so we can load it later without retraining
    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"[INFO] Scaler saved to: {scaler_save_path}")

    return train_df, test_df


def load_test_rul(rul_filepath: Path) -> pd.DataFrame:
    """
    Load the ground-truth RUL values for the test set.

    In CMAPSS, the test file ends at some point before failure.
    RUL_FD001.txt gives the true RUL at the last observed cycle
    for each engine in the test set.

    Used for evaluation — not used during training.
    """
    rul_df = pd.read_csv(rul_filepath, header=None, names=["true_RUL"])
    rul_df["unit_id"] = rul_df.index + 1  # engines are 1-indexed
    return rul_df


def run_preprocessing(config_path: str = "configs/data.yaml") -> None:
    """
    Full preprocessing pipeline entry point.

    Reads paths and settings from config, runs all steps,
    and saves processed files to data/processed/.

    Called by:  cmapps-tad preprocess
    """

    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path       = Path(config["dataset"]["raw_path"])
    processed_path = Path(config["dataset"]["processed_path"])
    scaler_path    = Path(config["artifacts"]["scaler_path"])
    subset         = config["dataset"].get("subset", "FD001")
    anomaly_thresh = config["dataset"].get("anomaly_threshold", 30)

    processed_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Preprocessing subset: {subset}")
    print(f"[INFO] Anomaly label threshold (RUL <=): {anomaly_thresh} cycles")

    # ── Load raw files ────────────────────────────────────────────────────────
    train_file = raw_path / f"train_{subset}.txt"
    test_file  = raw_path / f"test_{subset}.txt"
    rul_file   = raw_path / f"RUL_{subset}.txt"

    for f in [train_file, test_file, rul_file]:
        if not f.exists():
            raise FileNotFoundError(f"[ERROR] Expected file not found: {f}")

    print(f"[INFO] Loading raw files from: {raw_path}")
    train_df = load_raw(train_file)
    test_df  = load_raw(test_file)
    rul_df   = load_test_rul(rul_file)

    print(f"[INFO] Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    # ── Drop low-variance sensors ─────────────────────────────────────────────
    train_df = drop_low_variance_sensors(train_df)
    test_df  = drop_low_variance_sensors(test_df)

    # ── Compute RUL + proxy labels (train only) ───────────────────────────────
    train_df = compute_rul(train_df)
    train_df = add_proxy_labels(train_df, anomaly_threshold=anomaly_thresh)

    # ── Scale sensors ─────────────────────────────────────────────────────────
    train_df, test_df = scale_sensors(train_df, test_df, Path(scaler_path))

    # ── Save processed outputs ────────────────────────────────────────────────
    # Saved as Parquet for efficient loading later 
    train_out = processed_path / f"train_{subset}.parquet"
    test_out  = processed_path / f"test_{subset}.parquet"
    rul_out   = processed_path / f"rul_{subset}.parquet"

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out,  index=False)
    rul_df.to_parquet(rul_out,    index=False)

    print(f"[INFO] Saved train  -> {train_out}")
    print(f"[INFO] Saved test   -> {test_out}")
    print(f"[INFO] Saved RUL    -> {rul_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    anomaly_count  = train_df["anomaly_label"].sum()
    total          = len(train_df)
    anomaly_pct    = 100 * anomaly_count / total
    sensor_cols    = get_sensor_columns(train_df)

    print("\n── Preprocessing Summary ────────────────────────────────")
    print(f"  Subset              : {subset}")
    print(f"  Train rows          : {total:,}")
    print(f"  Test rows           : {len(test_df):,}")
    print(f"  Sensors used        : {len(sensor_cols)}  {sensor_cols}")
    print(f"  Anomaly label (=1)  : {anomaly_count:,} rows  ({anomaly_pct:.1f}%)")
    print(f"  Healthy label (=0)  : {total - anomaly_count:,} rows  ({100 - anomaly_pct:.1f}%)")
    print("─────────────────────────────────────────────────────────\n")
    print("[SUCCESS] Preprocessing complete.")