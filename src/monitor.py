"""Monitor logged predictions: volume, input drift and live model performance.

Command line (from the project root):

    python -m src.monitor                           # print the monitoring report
    python -m src.monitor --feedback 3f9a1c2e 1     # record the real condition (1 = Good, 0 = Poor)
"""
import argparse

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.predict import LOG_COLUMNS, PREDICTION_LOG, load_model_info
from src.preprocessing import PROJECT_ROOT, RAW_FEATURES

REFERENCE_PATH = PROJECT_ROOT / "models" / "reference_inputs.csv"

MIN_ROWS_FOR_DRIFT = 30       # need enough logged predictions for a meaningful test
MIN_LABELLED_FOR_PERF = 20    # need enough confirmed outcomes to judge performance
DRIFT_P_VALUE = 0.01
F1_DROP_ALERT = 0.15          # alert if live F1 is this far below the test-set F1


def load_log() -> pd.DataFrame:
    """Read the prediction log (empty DataFrame if nothing has been logged yet)."""
    if not PREDICTION_LOG.exists() or PREDICTION_LOG.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(PREDICTION_LOG, dtype={"prediction_id": str})


def summarize(log: pd.DataFrame) -> dict:
    """Basic usage statistics."""
    if log.empty:
        return {"total_predictions": 0}
    return {
        "total_predictions": int(len(log)),
        "share_predicted_good": float((log["prediction"] == 1).mean()),
        "labelled_predictions": int(log["actual"].notna().sum()),
        "first_prediction": log["timestamp"].min(),
        "last_prediction": log["timestamp"].max(),
        "model_versions": sorted(log["model_version"].astype(str).unique()),
    }


def drift_report(log: pd.DataFrame) -> pd.DataFrame | None:
    """Compare logged inputs with the training inputs (two-sample Kolmogorov-Smirnov test).

    Returns None when there are too few logged predictions or no reference data.
    """
    if len(log) < MIN_ROWS_FOR_DRIFT or not REFERENCE_PATH.exists():
        return None
    reference = pd.read_csv(REFERENCE_PATH)
    rows = []
    for col in RAW_FEATURES:
        stat, p_value = ks_2samp(reference[col], log[col].astype(float))
        rows.append({
            "feature": col,
            "training_mean": round(reference[col].mean(), 1),
            "live_mean": round(log[col].astype(float).mean(), 1),
            "ks_statistic": round(stat, 3),
            "p_value": round(p_value, 4),
            "drift_detected": bool(p_value < DRIFT_P_VALUE),
        })
    return pd.DataFrame(rows)


def performance_report(log: pd.DataFrame) -> dict | None:
    """Live performance on predictions whose real condition has been recorded."""
    labelled = log.dropna(subset=["actual"]) if not log.empty else log
    if len(labelled) < MIN_LABELLED_FOR_PERF:
        return None
    y_true = labelled["actual"].astype(int)
    y_pred = labelled["prediction"].astype(int)
    baseline = load_model_info().get("metrics", {}).get("test_f1")
    live_f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "labelled_rows": int(len(labelled)),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(live_f1, 3),
        "baseline_test_f1": baseline,
        "alert": bool(baseline is not None and live_f1 < baseline - F1_DROP_ALERT),
    }


def record_actual(prediction_id: str, actual: int) -> bool:
    """Store the real condition of a bridge that was predicted earlier."""
    if actual not in (0, 1):
        raise ValueError("actual must be 0 (Poor) or 1 (Good)")
    log = load_log()
    mask = log["prediction_id"] == str(prediction_id)
    if not mask.any():
        return False
    log["actual"] = log["actual"].astype("float64")
    log.loc[mask, "actual"] = float(actual)
    log["actual"] = log["actual"].astype("Int64")  # write 1 / 0 / blank, not 1.0
    log.to_csv(PREDICTION_LOG, index=False)
    return True


def print_report() -> None:
    log = load_log()
    print("=== Prediction log ===")
    for key, value in summarize(log).items():
        print(f"{key}: {value}")

    print("\n=== Input drift (vs training data) ===")
    drift = drift_report(log)
    if drift is None:
        print(f"Not enough data yet (need at least {MIN_ROWS_FOR_DRIFT} logged predictions).")
    else:
        print(drift.to_string(index=False))

    print("\n=== Live performance ===")
    perf = performance_report(log)
    if perf is None:
        print(f"Not enough confirmed outcomes yet (need at least {MIN_LABELLED_FOR_PERF}).")
    else:
        for key, value in perf.items():
            print(f"{key}: {value}")
        if perf["alert"]:
            print("ALERT: live F1 is well below the test-set F1. Consider retraining.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor logged predictions.")
    parser.add_argument("--feedback", nargs=2, metavar=("PREDICTION_ID", "ACTUAL"),
                        help="record the real condition (1 = Good, 0 = Poor) of a past prediction")
    args = parser.parse_args()
    if args.feedback:
        found = record_actual(args.feedback[0], int(args.feedback[1]))
        print("Recorded." if found else "Prediction id not found.")
    else:
        print_report()


if __name__ == "__main__":
    main()
