from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app_core import available_runs, load_run, prepare_features, run_inference
from Combined_lara import adapt_model


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"

st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
st.title("Industrial Anomaly Detection Dashboard")
st.caption("Upload process data to score it with a trained VAE, with optional LARA adaptation.")

runs = available_runs(OUTPUT_ROOT)
if not runs:
    st.error("No trained model run was found in the output directory.")
    st.stop()
run_name = st.sidebar.selectbox("Trained model run", runs, index=len(runs) - 1)
use_lara = st.sidebar.checkbox("Apply LARA adaptation", help="Adapts lightweight layers to this upload before scoring.")
lara_epochs = st.sidebar.slider("LARA epochs", 1, 20, 5, disabled=not use_lara)

@st.cache_resource(show_spinner=False)
def get_model(run: str):
    return load_run(OUTPUT_ROOT / run)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is None:
    st.info("The CSV must contain the model's sensor columns. `Timestamp` is optional.")
    st.stop()

try:
    source_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(source_df.head(), use_container_width=True)

if st.button("Detect anomalies", type="primary"):
    model, scaler, threshold, config = get_model(run_name)
    try:
        adapters = None
        if use_lara:
            with st.spinner("Adapting LARA layers to the uploaded data..."):
                _, scaled = prepare_features(source_df, scaler)
                adapters = adapt_model(model, scaled, epochs=lara_epochs)
        with st.spinner("Running anomaly detection..."):
            results, errors, _ = run_inference(source_df, model, scaler, threshold, adapters)
    except (ValueError, KeyError) as error:
        st.error(str(error))
        st.stop()

    anomalies = results[results["is_anomaly"] == 1]
    total, flagged, rate = st.columns(3)
    total.metric("Total records", len(results))
    flagged.metric("Anomalies detected", len(anomalies))
    rate.metric("Anomaly rate", f"{len(anomalies) / len(results):.2%}")
    st.caption(f"Run: {run_name} · threshold: {threshold:.6f} · features: {config['input_dim']}")

    st.subheader("Detected anomalies")
    st.dataframe(anomalies, use_container_width=True)
    st.download_button("Download scored CSV", results.to_csv(index=False).encode("utf-8"), "anomaly_results.csv", "text/csv")

    st.subheader("Reconstruction loss over the process sequence")
    figure, axis = plt.subplots(figsize=(10, 4))
    if "Timestamp" in results.columns:
        timestamps = pd.to_datetime(results["Timestamp"], errors="coerce")
        x_values = timestamps if timestamps.notna().all() else results.index
        x_label = "Timestamp" if timestamps.notna().all() else "Record index"
    else:
        x_values = results.index
        x_label = "Record index"

    above_threshold = errors > threshold
    axis.plot(x_values, errors, color="#2563eb", linewidth=1.25, label="Reconstruction loss")
    axis.axhline(threshold, color="#dc2626", linestyle="--", label="Anomaly threshold")
    axis.fill_between(
        x_values,
        threshold,
        errors,
        where=above_threshold,
        interpolate=True,
        color="#dc2626",
        alpha=0.25,
        label="Threshold exceeded",
    )
    axis.scatter(
        x_values[above_threshold],
        errors[above_threshold],
        color="#dc2626",
        s=16,
        zorder=3,
        label="Anomalous record",
    )
    axis.set(xlabel=x_label, ylabel="Reconstruction loss")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)

    st.subheader("Reconstruction-error distribution")
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.hist(errors, bins=50, color="#2563eb")
    axis.axvline(threshold, color="#dc2626", linestyle="--", label="Anomaly threshold")
    axis.set(xlabel="Reconstruction loss", ylabel="Records")
    axis.legend()
    st.pyplot(figure, clear_figure=True)
