"""Load the trained pipeline, make predictions and log them.

Command line example (from the project root):

    python -m src.predict --age 80 --traffic 3500
"""
import argparse
import csv
import json
import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache

import joblib
import pandas as pd

from src.preprocessing import AGE_THRESHOLD, PROJECT_ROOT, RAW_FEATURES, TRAFFIC_THRESHOLD

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
MODEL_INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"
LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG = LOG_DIR / "predictions.csv"
APP_LOG = LOG_DIR / "app.log"

LOG_COLUMNS = [
    "prediction_id", "timestamp", "age_of_bridge", "traffic_volume",
    "prediction", "score_good", "model_version", "actual",
]

logger = logging.getLogger("bridge.predict")


def _setup_file_logging() -> None:
    """Send application messages to logs/app.log (once)."""
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(APP_LOG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    """Load the saved pipeline (cached after the first call)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the model first: python -m src.train")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_model_info() -> dict:
    """Metadata saved next to the model (version, metrics, training ranges)."""
    if not MODEL_INFO_PATH.exists():
        return {}
    return json.loads(MODEL_INFO_PATH.read_text())


def validate_input(age: float, traffic: float) -> list[str]:
    """Raise for impossible inputs; return warnings for inputs outside the training range."""
    if age <= 0 or traffic < 0:
        raise ValueError("Age must be greater than 0 and traffic volume cannot be negative.")
    warnings = []
    ranges = load_model_info().get("training_input_range", {})
    for name, value, label in [("age_of_bridge", age, "Age"), ("traffic_volume", traffic, "Traffic volume")]:
        if name in ranges and not ranges[name][0] <= value <= ranges[name][1]:
            low, high = ranges[name]
            warnings.append(
                f"{label} ({value:g}) is outside the range seen in training data "
                f"({low:g} to {high:g}); treat the prediction with extra caution."
            )
    return warnings


def log_prediction(record: dict) -> None:
    """Append one prediction to logs/predictions.csv."""
    LOG_DIR.mkdir(exist_ok=True)
    new_file = not PREDICTION_LOG.exists() or PREDICTION_LOG.stat().st_size == 0
    with open(PREDICTION_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow({col: record.get(col, "") for col in LOG_COLUMNS})


def predict_bridge(age: float, traffic: float, log: bool = True) -> dict:
    """Predict the condition of one bridge from its age and daily traffic volume."""
    _setup_file_logging()
    warnings = validate_input(age, traffic)

    model = load_model()
    X = pd.DataFrame([[age, traffic]], columns=RAW_FEATURES)
    prediction = int(model.predict(X)[0])
    score = float(model.predict_proba(X)[0, 1])
    info = load_model_info()

    result = {
        "prediction_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "age_of_bridge": age,
        "traffic_volume": traffic,
        "prediction": prediction,
        "label": "Good" if prediction == 1 else "Poor",
        "score_good": round(score, 3),  # class-weighted score, NOT a calibrated probability
        "in_old_and_busy_zone": bool(age > AGE_THRESHOLD and traffic > TRAFFIC_THRESHOLD),
        "model_version": info.get("model_version", "unknown"),
        "warnings": warnings,
    }
    logger.info("prediction id=%s age=%s traffic=%s -> %s",
                result["prediction_id"], age, traffic, result["label"])
    if log:
        log_prediction(result)
    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict a DataFrame with `age_of_bridge` and `traffic_volume` columns."""
    model = load_model()
    out = df[RAW_FEATURES].copy()
    out["prediction"] = model.predict(out[RAW_FEATURES])
    out["label"] = out["prediction"].map({1: "Good", 0: "Poor"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the condition of a bridge.")
    parser.add_argument("--age", type=float, required=True, help="age of the bridge in years")
    parser.add_argument("--traffic", type=float, required=True, help="traffic volume")
    parser.add_argument("--no-log", action="store_true", help="do not write to the prediction log")
    args = parser.parse_args()

    try:
        result = predict_bridge(args.age, args.traffic, log=not args.no_log)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
    print(f"Predicted condition: {result['label']}  (model {result['model_version']})")
    for warning in result["warnings"]:
        print("Warning:", warning)


if __name__ == "__main__":
    main()
