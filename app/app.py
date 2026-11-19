"""Streamlit app for the Bridge Condition Prediction System.

Run from the project root:   streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import streamlit as st

from src import monitor
from src.predict import load_model_info, predict_bridge
from src.preprocessing import AGE_THRESHOLD, TRAFFIC_THRESHOLD

st.set_page_config(page_title="Bridge Condition Prediction", page_icon="🌉", layout="centered")

info = load_model_info()
zone = info.get("zone_stats", {})

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the model")
    if info:
        metrics = info.get("metrics", {})
        st.write(f"**Model:** {info.get('model_type')}")
        st.write(f"**Version:** {info.get('model_version')}")
        st.write(f"**Trained on:** {info.get('training_rows')} bridges")
        st.write("**Test-set results**")
        st.write(f"- F1 (Good class): {metrics.get('test_f1', 0):.2f}")
        st.write(f"- Recall (Good): {metrics.get('test_recall', 0):.2f}")
        st.write(f"- Precision (Good): {metrics.get('test_precision', 0):.2f}")
    else:
        st.warning("No model information found. Run `python -m src.train` first.")
    st.caption("This tool is a screening aid. It does not replace a structural inspection.")

st.title("🌉 Bridge Condition Prediction")
st.write("Predict whether a bridge is in **good** or **poor** condition from its age and traffic volume.")

tab_predict, tab_monitor = st.tabs(["Predict", "Monitoring"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    col1, col2 = st.columns(2)
    age = col1.slider("Age of bridge (years)", min_value=1, max_value=150, value=60)
    traffic = col2.number_input("Traffic volume (vehicles)", min_value=0, max_value=20000,
                                value=2500, step=50)

    if st.button("Predict condition", type="primary"):
        try:
            st.session_state["result"] = predict_bridge(float(age), float(traffic))
        except (ValueError, FileNotFoundError) as error:
            st.session_state.pop("result", None)
            st.error(str(error))

    result = st.session_state.get("result")
    if result:
        if result["prediction"] == 1:
            st.success("Predicted condition: **GOOD**")
        else:
            st.error("Predicted condition: **POOR**")

        for warning in result["warnings"]:
            st.warning(warning)

        if result["in_old_and_busy_zone"]:
            share = zone.get("share_good_in_zone")
            detail = f" In the training data, {share:.0%} of bridges in this group were in good condition." if share is not None else ""
            st.info(f"This bridge is older than {AGE_THRESHOLD} years and carries more than "
                    f"{TRAFFIC_THRESHOLD:,} vehicles. This is the only group in which good "
                    f"bridges appear.{detail} The available data cannot tell good from poor "
                    f"inside this group, so treat a GOOD result as a screening outcome, not a guarantee.")
        else:
            st.info(f"This bridge is not both older than {AGE_THRESHOLD} years and carrying more than "
                    f"{TRAFFIC_THRESHOLD:,} vehicles. None of the training bridges outside this group "
                    f"was in good condition.")

        st.caption(f"Prediction ID: `{result['prediction_id']}` · model {result['model_version']}. "
                   "Use the ID in the Monitoring tab to record the real condition after inspection.")

# ---------------------------------------------------------------- monitoring tab
with tab_monitor:
    st.button("Refresh")
    log = monitor.load_log()
    summary = monitor.summarize(log)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predictions logged", summary["total_predictions"])
    if summary["total_predictions"]:
        c2.metric("Predicted good", f"{summary['share_predicted_good']:.0%}")
        c3.metric("With confirmed outcome", summary["labelled_predictions"])

        st.subheader("Input drift")
        drift = monitor.drift_report(log)
        if drift is None:
            st.caption(f"Needs at least {monitor.MIN_ROWS_FOR_DRIFT} logged predictions.")
        else:
            st.dataframe(drift, hide_index=True)
            if drift["drift_detected"].any():
                st.warning("Inputs differ from the training data. Check whether the model still applies.")
            else:
                st.success("No input drift detected.")

        st.subheader("Live performance")
        perf = monitor.performance_report(log)
        if perf is None:
            st.caption(f"Needs at least {monitor.MIN_LABELLED_FOR_PERF} predictions with a confirmed outcome.")
        else:
            st.write(f"F1: **{perf['f1']}** (test-set baseline {perf['baseline_test_f1']}) · "
                     f"recall {perf['recall']} · precision {perf['precision']} · "
                     f"on {perf['labelled_rows']} bridges")
            if perf["alert"]:
                st.error("Live F1 is well below the test-set F1. Consider retraining.")

        st.subheader("Recent predictions")
        st.dataframe(log.tail(20).iloc[::-1], hide_index=True)

        st.subheader("Record the real condition")
        open_ids = log.loc[log["actual"].isna(), "prediction_id"].tolist()[::-1]
        if open_ids:
            chosen = st.selectbox("Prediction ID", open_ids)
            actual = st.radio("Condition found at inspection", ["Good", "Poor"], horizontal=True)
            if st.button("Save outcome"):
                monitor.record_actual(chosen, 1 if actual == "Good" else 0)
                st.success("Saved. Refresh to update the performance numbers.")
        else:
            st.caption("All logged predictions have a confirmed outcome.")
    else:
        st.info("No predictions yet. Make one in the Predict tab.")
