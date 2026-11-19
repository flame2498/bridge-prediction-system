# Bridge Condition Prediction System

Predict whether a bridge is in **good** (1) or **poor** (0) condition from its age and traffic volume, and serve the model through a Streamlit web app with logging and monitoring.

## 1. Problem Statement

Bridges deteriorate over time because of age, traffic load, material quality and maintenance. Governments and civil engineering departments need to decide which bridges to inspect and repair first.

**Question:** can the condition of a bridge be predicted from its structural and usage characteristics?

**Task:** binary classification on `Bridge_Condition` (1 = Good, 0 = Poor).

**Business impact:** prioritize maintenance, reduce the risk of structural failure, optimize repair budgets and improve public safety.

## 2. Approach

| Sprint | Work | Notebook / code |
|---|---|---|
| 1. Data understanding & preprocessing | Loaded 720 rows, removed 128 duplicates (592 left), fixed names and a category typo, EDA, outlier check (none), encoding, scaling, 80/20 stratified split | `notebooks/sprint1.ipynb` |
| 2. Model building & evaluation | Trained Logistic Regression (baseline), Decision Tree, Random Forest, SVM and Naive Bayes; compared metrics and over/underfitting | `notebooks/sprint2.ipynb` |
| 3. Optimization & final model | Feature engineering, feature selection (correlation, importance, RFE), tuning with `GridSearchCV` / `RandomizedSearchCV`, final evaluation | `notebooks/sprint3.ipynb` |
| 4. Deployment & MLOps | Reusable pipeline, Streamlit app, prediction logging, monitoring, experiment tracking, documentation | `src/`, `app/` |

**Final pipeline** (`src/train.py`):

```
raw age + traffic  ->  feature engineering  ->  StandardScaler  ->  Decision Tree
                       (traffic_per_year,
                        old_and_busy flag)
```

Because feature engineering is part of the pipeline, the model only needs two raw inputs: `age_of_bridge` and `traffic_volume`.

## 3. Results

Evaluated once on the unseen 20% test set (119 bridges, 20 of them good):

| Model | Accuracy | Precision (Good) | Recall (Good) | F1 (Good) |
|---|---|---|---|---|
| Sprint 2 baseline: Logistic Regression | 0.832 | 0.500 | 0.350 | 0.412 |
| Sprint 2 best: Random Forest | 0.891 | 0.667 | 0.700 | 0.683 |
| **Sprint 3 final: Decision Tree** | **0.849** | **0.526** | **1.000** | **0.690** |

Final model: ROC-AUC 0.936, cross-validated F1 0.727 (±0.040), training F1 0.720 vs test F1 0.690, so there is no overfitting.

**Key findings**
- **Every good bridge in the data is both older than 50 years and carries more than 2,000 vehicles.** None of the 410 bridges outside that group is good; inside it, 55.5% are good.
- Inside that group, the available features cannot separate good from poor (cross-validated ROC-AUC 0.55). A one-line rule, "old and busy", scores the same test F1 (0.69) as the tuned model.
- Material type and maintenance level have almost no predictive power in this dataset.
- Better results need new information (inspection scores, structural details, repair history, load ratings), not more tuning.

**Limitations**
- The test set is small (20 good bridges), so scores are noisy.
- The final model finds all good bridges but also labels 18 of 99 poor bridges as good. If calling a poor bridge "good" is the more dangerous mistake, the Sprint 2 Random Forest makes fewer of those (7).
- Older, busier bridges being "good" is unusual. Confirm with the data owner that 1 really means Good, and how the label was produced.
- The `old_and_busy` thresholds (50 years, 2,000 vehicles) were found in the EDA of the full dataset.
- The model's `score_good` is a class-weighted score, **not** a calibrated probability (it reaches about 0.98 in a group where 56% of bridges are good). The app therefore shows the observed good rate instead.

## 4. Project Structure

```
bridge-condition-prediction/
├── data/
│   ├── bridge_data.csv            # raw data
│   └── bridge_data_clean.csv      # cleaned data (Sprint 1 output)
├── notebooks/
│   ├── sprint1.ipynb              # preprocessing and EDA
│   ├── sprint2.ipynb              # model comparison
│   └── sprint3.ipynb              # optimization and final model
├── src/
│   ├── preprocessing.py           # cleaning, feature engineering, split, preprocessing pipeline
│   ├── train.py                   # train, evaluate, save model, track experiment
│   ├── predict.py                 # load model, validate input, predict, log predictions
│   └── monitor.py                 # usage, input drift, live performance, feedback
├── models/
│   ├── model.pkl                  # trained end-to-end pipeline (used by the app)
│   ├── model_info.json            # version, parameters, metrics, training ranges
│   ├── reference_inputs.csv       # training inputs, used for drift checks
│   └── bridge_condition_model.joblib   # Sprint 3 notebook artifact (superseded by model.pkl)
├── app/
│   └── app.py                     # Streamlit UI
├── logs/                          # predictions.csv and app.log are created at run time
├── requirements.txt               # runtime dependencies
├── requirements-dev.txt           # + MLflow, DVC and notebook dependencies
├── params.yaml                    # hyperparameters for training pipeline
├── dvc.yaml                       # DVC reproducible pipeline stages
├── dvc.lock                       # DVC pipeline reproduction lock file
├── metrics.json                   # tracked evaluation metrics
├── Dockerfile                     # container image specification
├── docker-compose.yml             # Docker Compose specification with persistent log volumes
└── README.md
```

`src/monitor.py`, `dvc.yaml`, `params.yaml`, `docker-compose.yml`, `requirements-dev.txt`, the `Dockerfile` and `.gitignore` are additions to the standard layout.

## 5. How to Run

Use Python 3.11 or newer (tested with 3.12). Run all commands from the project root.

```bash
# 1. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # or requirements.txt for just the app

# 2. (Re)train the model. Writes models/model.pkl and logs the run to MLflow
python -m src.train

# 3. Predict from the command line
python -m src.predict --age 80 --traffic 3500      # -> Good
python -m src.predict --age 30 --traffic 1000      # -> Poor

# 4. Launch the web app
streamlit run app/app.py                           # opens http://localhost:8501

# 5. Check monitoring from the command line
python -m src.monitor
python -m src.monitor --feedback <prediction_id> 1   # record the real condition (1 = Good, 0 = Poor)

# 6. Compare experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db    # opens http://localhost:5000

# 7. Reproduce pipeline with DVC
dvc repro                                           # reproduces prepare and train stages
dvc metrics show                                    # view model performance metrics
```

Try other settings and compare the runs in MLflow without touching the saved model:

```bash
python -m src.train --max-depth 3 --run-name depth-3 --no-save
python -m src.train --min-samples-leaf 20 --run-name leaf-20 --no-save
```

To re-run the notebooks, open them from the `notebooks/` folder (`jupyter notebook`).

> Run modules with `python -m src.<name>` (not `python src/<name>.py`), so the `src` package is found. `models/model.pkl` was saved with scikit-learn 1.9.1, which is pinned in `requirements.txt`. If you use another version, run `python -m src.train` again.

## 6. MLOps Practices

**Version control.** The project is a Git repository with small, focused commits. `.gitignore` keeps logs, MLflow databases and caches out of the repository, while the small trained model and the data are tracked so the app works straight after cloning.

**Experiment tracking (MLflow).** Every call to `python -m src.train` records the model type, hyperparameters, features, metrics (test accuracy, precision, recall, F1, ROC-AUC, training F1, cross-validated F1) and the model file as an artifact. Metadata for the deployed model is also saved in `models/model_info.json`.

**Logging.**
- `logs/predictions.csv`: one row per prediction with ID, timestamp, inputs, prediction, score, model version and (later) the actual condition.
- `logs/app.log`: application messages.

**Monitoring** (Monitoring tab in the app, or `python -m src.monitor`):
- *Usage:* number of predictions and share predicted good.
- *Input drift:* logged inputs are compared with the training inputs using a Kolmogorov-Smirnov test (needs at least 30 predictions; flags p < 0.01).
- *Live performance:* once real conditions are recorded (via the app or `--feedback`), F1, precision, recall and accuracy are calculated and compared with the test-set F1. An alert is raised if live F1 is more than 0.15 below it (needs at least 20 confirmed outcomes).
- Retrain with `python -m src.train` when drift or an alert appears.

## 7. Deployment

**Local:** `streamlit run app/app.py` (see above).

**Streamlit Community Cloud (free):**
1. Push the repository to GitHub (see section 8).
2. Go to [share.streamlit.io](https://share.streamlit.io), choose **New app**, and select the repository and branch.
3. Set the main file path to `app/app.py` and deploy. The platform installs `requirements.txt` automatically.

On Streamlit Cloud the `logs/` folder is temporary storage, so prediction logs reset whenever the app restarts. For permanent logs, write to a database or cloud storage instead.

**Docker (with persistent logs):**
```bash
# Using Docker Compose (recommended - mounts ./logs for persistence)
docker compose up -d

# Or using plain Docker
docker build -t bridge-condition-app .
docker run -p 8501:8501 -v $(pwd)/logs:/app/logs bridge-condition-app
```
Access at `http://localhost:8501`.

## 8. GitHub Repository

Repository URL:
```bash
https://github.com/flame2498/bridge-prediction-system.git
```

To push changes to GitHub:
```bash
git add .
git commit -m "Your commit message"
git push origin main
```

## 9. Using the Model

| Input | Meaning |
|---|---|
| `age_of_bridge` | Age in years (training data: 1 to 99) |
| `traffic_volume` | Traffic volume (training data: 51 to about 5,000) |

Inputs outside the training range still produce a prediction, with a warning. The app is a screening aid and does not replace a structural inspection.
