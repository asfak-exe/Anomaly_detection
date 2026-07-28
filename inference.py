import os
import json
import joblib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from main import huber_reconstruction_loss
from vae import VAE

# Config

DATA_PATH = r"C:\Users\6000_J1I\Desktop\asfak_project\test_FINAL.csv"

OUTPUT_ROOT = r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\FINAL\output"

BATCH_SIZE = 128

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {device}")

#used to select the run u want to check on or load the model from

def get_available_runs(output_root):

    runs = sorted(
        [
            d for d in os.listdir(output_root)
            if d.startswith("run_")
        ],
        key=lambda x: int(x.split("_")[1])
    )

    if len(runs) == 0:
        raise RuntimeError("No trained models found.")

    print("\nAvailable Runs")

    for r in runs:
        print(r)

    return runs



def load_run(output_root):
    """
    Loads a trained run (model, scaler and threshold).
    """

    runs = get_available_runs(output_root)

    run_number = input("\nEnter run number: ").strip()

    run_name = f"run_{run_number}"

    if run_name not in runs:
        raise ValueError(f"{run_name} does not exist.")

    run_dir = os.path.join(
        output_root,
        run_name
    )

    
    # Load Training Config
    

    with open(
        os.path.join(run_dir, "train_config.json"),
        "r"
    ) as f:

        config = json.load(f)

    
    # Load Threshold
    

    with open(
        os.path.join(run_dir, "threshold.json"),
        "r"
    ) as f:

        threshold = json.load(f)["threshold"]

    
    # Load Scaler
    

    scaler = joblib.load(
        os.path.join(
            run_dir,
            "scaler.pkl"
        )
    )

   
    # Load Model
   

    model = VAE(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(run_dir, "vae.pth"),
            map_location=device,
        )
    )

    model.eval()

    print("\nLoaded Successfully")
    print(f"Run        : {run_name}")
    print(f"Threshold  : {threshold:.6f}")

    return (
        model,
        scaler,
        threshold,
        run_dir,
    )


def prepare_test_data(
    data_path,
    scaler,
):
    """
    Loads test data and prepares it for inference.
    """

    print("\nLoading test dataset...")

    df = pd.read_csv(data_path)

    print(f"Dataset Shape : {df.shape}")

    # Keep Timestamp
    

    if "Timestamp" in df.columns:

        timestamps = pd.to_datetime(
            df["Timestamp"]
        ).reset_index(drop=True)

        X = df.drop(columns=["Timestamp"])

    else:

        timestamps = None

        X = df.copy()

    print(f"Feature Shape : {X.shape}")

    
# Align features with training


    train_features = list(scaler.feature_names_in_)
    train_means = dict(zip(train_features, scaler.mean_))

# Add missing features using training mean
    missing_cols = set(train_features) - set(X.columns)

    if missing_cols:
     print(f"Adding {len(missing_cols)} missing features...")
     for col in missing_cols:
        X[col] = train_means[col]

# Remove unexpected features
    extra_cols = set(X.columns) - set(train_features)

    if extra_cols:
     print(f"Removing {len(extra_cols)} extra features...")
     X = X.drop(columns=list(extra_cols))

# Reorder to match training
    X = X.reindex(columns=train_features)

    print(f"Final Feature Shape : {X.shape}")

    assert list(X.columns) == train_features
    
    # Scale using TRAINING scaler
   
    X_scaled = scaler.transform(X)
    
    # Tensor
    
    data_tensor = torch.FloatTensor(
        X_scaled
    )

    dataset = TensorDataset(
        data_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        df,
        timestamps,
        loader,
    )


def predict(
    model,
    loader,
    threshold,
):
    """
    Runs inference and detects anomalies.
    
    """

    model.eval()

    all_losses = []
    all_predictions = []
    all_reconstructions = []
    all_sensor_errors = []

    with torch.no_grad():

        for (batch_x,) in loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            recon_x, _, _ = model(batch_x)

            # Per-sensor reconstruction error (scaled space)
            sensor_errors = torch.abs(batch_x - recon_x)

            all_sensor_errors.extend(
                sensor_errors.cpu().numpy()
                )

            losses = huber_reconstruction_loss(
                recon_x,
                batch_x,
            )

            predictions = (
                losses > threshold
            ).int()

            all_losses.extend(
                losses.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_reconstructions.extend(
                recon_x.cpu().numpy()
            )

            

    all_losses = np.array(all_losses)
    all_predictions = np.array(all_predictions)
    all_reconstructions = np.array(all_reconstructions)
    all_sensor_errors = np.array(all_sensor_errors)

    print("\n========== INFERENCE ==========")

    print(f"Threshold          : {threshold:.6f}")
    print(f"Total Samples      : {len(all_losses)}")
    print(f"Detected Anomalies : {all_predictions.sum()}")

    print(
        f"Anomaly Percentage : "
        f"{100 * all_predictions.mean():.2f}%"
    )

    return (
        all_losses,
        all_predictions,
        all_reconstructions,
        all_sensor_errors
    )


def save_results(
    df,
    timestamps,
    losses,
    predictions,
    reconstructions,
    sensor_errors,
    scaler,
    run_dir,
):
    """
    Saves inference results.
    """

    results = df.copy()

    if timestamps is not None:
        results["Timestamp"] = timestamps

    feature_cols = [
        col
        for col in df.columns
        if col != "Timestamp"
    ]

    # Convert reconstructions back to original units
    reconstructions = scaler.inverse_transform(
    reconstructions
)

    recon_df = pd.DataFrame(
        reconstructions,
        columns=[
            f"{c}_Recon"
            for c in feature_cols
        ]
    )

    error_df = pd.DataFrame(
    sensor_errors,
    columns=[
        f"{c}_Error"
        for c in feature_cols
    ]
)

    results = pd.concat(
    [
        results.reset_index(drop=True),
        recon_df.reset_index(drop=True),
        error_df.reset_index(drop=True),
    ],
    axis=1,
)

    results["Reconstruction_Loss"] = losses
    results["is_anomaly"] = predictions

    results_path = os.path.join(
        run_dir,
        "predictions2.csv",
    )

    results.to_csv(
        results_path,
        index=False,
    )

    anomaly_df = results[
        results["is_anomaly"] == 1
    ]

    if "Timestamp" in anomaly_df.columns:

        anomaly_timestamps = anomaly_df[
            [
                "Timestamp",
                "Reconstruction_Loss",
            ]
        ]

        timestamps_path = os.path.join(
            run_dir,
            "anomaly_timestamps2.csv",
        )

        anomaly_timestamps.to_csv(
            timestamps_path,
            index=False,
        )

        print(f"Timestamp file     : {timestamps_path}")

    anomaly_path = os.path.join(
        run_dir,
        "detected_anomalies2.csv",
    )

    anomaly_df.to_csv(
        anomaly_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("Inference Complete")
    print("=" * 60)

    print(f"Results saved      : {results_path}")
    print(f"Anomalies detected : {len(anomaly_df)}")
    print(f"Anomaly file       : {anomaly_path}")



if __name__ == "__main__":

    model, scaler, threshold, run_dir = load_run(
        OUTPUT_ROOT
    )

    (
        df,
        timestamps,
        loader,
    ) = prepare_test_data(
        DATA_PATH,
        scaler,
    )

    losses, predictions, reconstructions,sensor_errors = predict(
    model,
    loader,
    threshold,
)

    save_results(
    df,
    timestamps,
    losses,
    predictions,
    reconstructions,
    sensor_errors,
    scaler,
    run_dir,
)
    









