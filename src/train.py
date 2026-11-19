"""Train, evaluate, save and track the bridge condition model.

Run from the project root:

    python -m src.train
    python -m src.train --max-depth 3 --run-name shallow-tree --no-save
"""
import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    AGE_THRESHOLD,
    MODEL_FEATURES,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    RAW_FEATURES,
    TRAFFIC_THRESHOLD,
    clean_data,
    get_preprocessor,
    load_raw_data,
    split_data,
)

logger = logging.getLogger("bridge.train")

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
REFERENCE_PATH = MODELS_DIR / "reference_inputs.csv"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# Best hyperparameters found with GridSearchCV / RandomizedSearchCV in Sprint 3
DEFAULT_PARAMS = {
    "max_depth": 4,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
}
TEST_SIZE = 0.20
SPLIT_SEED = 42


def build_pipeline(params: dict) -> Pipeline:
    """Full end-to-end pipeline: feature engineering -> scaling -> classifier."""
    return Pipeline(
        [
            ("preprocessing", get_preprocessor()),
            ("model", DecisionTreeClassifier(**params)),
        ]
    )


def evaluate(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Metrics on the training and the unseen test data."""
    test_pred = pipeline.predict(X_test)
    train_pred = pipeline.predict(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    return {
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred),
        "test_recall": recall_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred),
        "test_roc_auc": roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]),
        "train_f1": f1_score(y_train, train_pred),
        "cv_f1_mean": cv_f1.mean(),
        "cv_f1_std": cv_f1.std(),
    }


def log_to_mlflow(run_name, params, metrics, model_path, experiment="bridge-condition"):
    """Record parameters, metrics and the model file in a local MLflow store."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed, skipping experiment tracking")
        return
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, "model_type": "DecisionTreeClassifier",
                           "features": ",".join(MODEL_FEATURES), "test_size": TEST_SIZE})
        mlflow.log_metrics(metrics)
        if model_path is not None:
            mlflow.log_artifact(str(model_path))
    logger.info("Run '%s' logged to MLflow", run_name)


def main(args) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    df = clean_data(load_raw_data(args.data))
    logger.info("Loaded %d unique bridges (after cleaning)", len(df))

    X_train, X_test, y_train, y_test = split_data(df, TEST_SIZE, SPLIT_SEED)
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    params = {**DEFAULT_PARAMS}
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    if args.min_samples_leaf is not None:
        params["min_samples_leaf"] = args.min_samples_leaf

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_train, y_train, X_test, y_test)
    for name, value in metrics.items():
        logger.info("%-15s %.3f", name, value)

    model_path = None
    if not args.no_save:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = args.model_out
        joblib.dump(pipeline, model_path)

        in_zone = X_train[(X_train["age_of_bridge"] > AGE_THRESHOLD)
                          & (X_train["traffic_volume"] > TRAFFIC_THRESHOLD)]
        info = {
            "model_version": "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_type": "DecisionTreeClassifier",
            "sklearn_version": sklearn.__version__,
            "params": params,
            "raw_features": RAW_FEATURES,
            "model_features": MODEL_FEATURES,
            "target": "bridge_condition (1 = Good, 0 = Poor)",
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "training_input_range": {
                col: [float(X_train[col].min()), float(X_train[col].max())] for col in RAW_FEATURES
            },
            "zone_stats": {
                "age_threshold": AGE_THRESHOLD,
                "traffic_threshold": TRAFFIC_THRESHOLD,
                "bridges_in_zone": int(len(in_zone)),
                "share_good_in_zone": float(y_train.loc[in_zone.index].mean()),
                "good_outside_zone": int(y_train.drop(in_zone.index).sum()),
            },
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        }
        MODEL_INFO_PATH.write_text(json.dumps(info, indent=2))
        X_train.to_csv(REFERENCE_PATH, index=False)  # reference data for drift monitoring
        logger.info("Saved %s, %s and %s", model_path, MODEL_INFO_PATH.name, REFERENCE_PATH.name)

    if not args.no_track:
        log_to_mlflow(args.run_name, params, metrics, model_path)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the bridge condition model.")
    parser.add_argument("--data", default=RAW_DATA_PATH, help="path to the raw CSV file")
    parser.add_argument("--model-out", default=MODEL_PATH, help="where to save the model")
    parser.add_argument("--max-depth", type=int, help="override the tree's max_depth")
    parser.add_argument("--min-samples-leaf", type=int, help="override min_samples_leaf")
    parser.add_argument("--run-name", default="decision-tree", help="name of the MLflow run")
    parser.add_argument("--no-save", action="store_true", help="do not overwrite the saved model")
    parser.add_argument("--no-track", action="store_true", help="do not log to MLflow")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
