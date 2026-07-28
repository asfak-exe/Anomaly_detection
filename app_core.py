"""Portable model loading, preprocessing, and anomaly scoring for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from vae import VAE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def available_runs(output_root: Path) -> list[str]:
    return sorted(
        (p.name for p in output_root.glob("run_*") if (p / "vae.pth").exists()),
        key=lambda name: int(name.split("_")[1]),
    )


def load_run(run_dir: Path):
    with (run_dir / "train_config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    with (run_dir / "threshold.json").open(encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])

    model = VAE(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(run_dir / "vae.pth", map_location=DEVICE))
    model.eval()
    return model, joblib.load(run_dir / "scaler.pkl"), threshold, config


def prepare_features(df: pd.DataFrame, scaler):
    features = df.drop(columns=["Timestamp"], errors="ignore").copy()
    expected = list(scaler.feature_names_in_)
    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError("Non-numeric feature columns: " + ", ".join(non_numeric))

    missing = [column for column in expected if column not in features]
    for column, mean in zip(expected, scaler.mean_):
        if column in missing:
            features[column] = mean
    features = features.reindex(columns=expected)
    if features.isna().any().any():
        features = features.fillna(pd.Series(dict(zip(expected, scaler.mean_))))
    return features, scaler.transform(features)


def huber_loss(reconstruction, values):
    error = reconstruction - values
    absolute_error = torch.abs(error)
    return torch.where(absolute_error <= 1, 0.5 * error.square(), absolute_error - 0.5).mean(dim=1)


def run_inference(df: pd.DataFrame, model, scaler, threshold: float, adapters=None):
    features, scaled = prepare_features(df, scaler)
    values = torch.tensor(scaled, dtype=torch.float32)
    losses, sensor_errors = [], []
    model.eval()
    if adapters is not None:
        adapters.eval()

    with torch.no_grad():
        for batch in values.split(256):
            batch = batch.to(DEVICE)
            mu, _ = model.encode(batch)
            reconstruction = model.decode(mu)
            if adapters is not None:
                _, reconstruction = adapters(mu, reconstruction)
            losses.append(huber_loss(reconstruction, batch).cpu().numpy())
            sensor_errors.append(torch.abs(batch - reconstruction).cpu().numpy())

    losses = np.concatenate(losses)
    sensor_errors = np.concatenate(sensor_errors)
    result = df.copy().reset_index(drop=True)
    result["Reconstruction_Loss"] = losses
    result["is_anomaly"] = (losses > threshold).astype(int)
    result["Top_Contributing_Sensor"] = features.columns[sensor_errors.argmax(axis=1)]
    return result, losses, sensor_errors
