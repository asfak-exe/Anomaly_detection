import os
import re
import json
import joblib
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)

from vae import VAE



# CONFIG


DATA_PATH = r"C:\Users\6000_J1I\Desktop\asfak_project\train_FINAL.csv"

OUTPUT_ROOT = r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\FINAL\output"

INPUT_DIM=944
LATENT_DIM = 32

BATCH_SIZE = 128
LEARNING_RATE = 1e-4

EPOCHS = 20
DROPOUT = 0.3

BETA = 0.001
THRESHOLD_PERCENTILE = 95

PATIENCE = 5


# gpu or cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {device}")

#creates a new folder for each run and its config

def get_next_run_dir(output_root):
    """
    Creates

    outputs/
        run_1/
        run_2/
        ...

    automatically.
    """

    os.makedirs(output_root, exist_ok=True)

    pattern = re.compile(r"run_(\d+)")

    runs = []

    for folder in os.listdir(output_root):

        match = pattern.fullmatch(folder)

        if match:
            runs.append(int(match.group(1)))

    next_run = 1 if len(runs) == 0 else max(runs) + 1

    run_dir = os.path.join(
        output_root,
        f"run_{next_run}"
    )

    os.makedirs(run_dir)

    print("=" * 60)
    print(f"Run Directory : {run_dir}")
    print("=" * 60)

    return run_dir

def save_sensor_thresholds(df, run_dir):
    """
    Saves the 95th percentile threshold for every sensor.
    """

    if "Timestamp" in df.columns:
        data = df.drop(columns=["Timestamp"])
    else:
        data = df.copy()

    threshold_df = pd.DataFrame({
        "Sensor": data.columns,
        "Threshold": data.quantile(0.95).values
    })

    output_path = os.path.join(
        run_dir,
        "sensor_thresholds.csv"
    )

    threshold_df.to_csv(
        output_path,
        index=False
    )

    print(f"Sensor thresholds saved to:\n{output_path}")


#Preprocessing of data (includes scaling and converting to tensors)(Not for cleaning data)

def prepare_data(data_path, run_dir):
    """
    Loads cleaned data, scales features and performs
    an 80:20 chronological train-validation split.
    """

    print("\nLoading dataset...")

    df = pd.read_csv(data_path)

    print(f"Original Shape : {df.shape}")

    # Save sensor thresholds
    
    save_sensor_thresholds(
    df,
    run_dir,)


    # Separate Timestamp (do NOT train on it)
    

    if "Timestamp" in df.columns:

        timestamps = df["Timestamp"].copy()

        X = df.drop(columns=["Timestamp"])

    else:

        timestamps = None
        X = df.copy()

    print(f"Feature Shape : {X.shape}")


    # Standard Scaling
    
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    scaler_path = os.path.join(run_dir, "scaler.pkl")

    joblib.dump(scaler, scaler_path)

    print(f"Scaler saved to:\n{scaler_path}")

    
    # Convert to Tensor


    data_tensor = torch.FloatTensor(X_scaled)

    split_idx = int(0.8 * len(data_tensor))

    train_tensor = data_tensor[:split_idx]
    val_tensor = data_tensor[split_idx:]

    train_timestamps = (
        timestamps.iloc[:split_idx].reset_index(drop=True)
        if timestamps is not None else None
    )

    val_timestamps = (
        timestamps.iloc[split_idx:].reset_index(drop=True)
        if timestamps is not None else None
    )

    print("\nDataset Summary")
    # print("-" * 40)
    print(f"Total Samples      : {len(data_tensor)}")
    print(f"Training Samples   : {len(train_tensor)}")
    print(f"Validation Samples : {len(val_tensor)}")

    return (
        train_tensor,
        val_tensor,
        train_timestamps,
        val_timestamps,
        scaler
    )



# Dataloader for loading into the model (torch)

def create_dataloaders(
    train_tensor,
    val_tensor,
    batch_size,
):
    """
    Creates train and validation dataloaders.
    """

    train_dataset = TensorDataset(
        train_tensor,
        train_tensor
    )

    val_dataset = TensorDataset(
        val_tensor,
        val_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,          # Time series
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader
    )

def huber_reconstruction_loss(
    recon_x,
    x,
    delta=1.0
):
    """
    Computes per-sample Huber reconstruction loss.

    Returns
    -------
    Tensor of shape (batch_size,)
    """

    error = recon_x - x

    abs_error = torch.abs(error)

    per_element_loss = torch.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta)
    )

    return per_element_loss.mean(dim=1)


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
):
    """
    Trains the model for one epoch.

    Returns
    -------
    train_loss : float
        Average training loss for the epoch.
    """

    model.train()

    running_loss = 0.0

    for batch_x, _ in train_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        recon_x, mu, logvar = model(batch_x)

        recon_loss = huber_reconstruction_loss(
            recon_x,
            batch_x
        ).mean()

        kl_div = -0.5 * torch.mean(
            torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=1
            )
        )

        loss = recon_loss + BETA * kl_div

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    return train_loss


def validate_epoch(
    model,
    val_loader,
    device,
):
    """
    Validates the model for one epoch.

    Returns
    -------
    val_loss : float
        Average validation loss for the epoch.
    """

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for batch_x, _ in val_loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True
            )

            recon_x, mu, logvar = model(batch_x)

            recon_loss = huber_reconstruction_loss(
                recon_x,
                batch_x
            ).mean()

            kl_div = -0.5 * torch.mean(
                torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(),
                    dim=1
                )
            )

            loss = recon_loss + BETA * kl_div

            running_loss += loss.item() * batch_x.size(0)

    val_loss = running_loss / len(val_loader.dataset)

    return val_loss


def compute_threshold(
    model,
    train_tensor,
    val_tensor,
    device,
    batch_size=BATCH_SIZE,
):
    """
    Computes the anomaly detection threshold using
    the entire clean dataset (train + validation).
    """

    model.eval()

    all_tensor = torch.cat(
        [train_tensor, val_tensor],
        dim=0
    )

    loader = DataLoader(
        TensorDataset(all_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    all_losses = []

    with torch.no_grad():

        for (batch_x,) in loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True
            )

            recon_x, _, _ = model(batch_x)

            losses = huber_reconstruction_loss(
                recon_x,
                batch_x
            )

            all_losses.extend(
                losses.cpu().numpy()
            )

    all_losses = np.array(all_losses)

    print("\n========== TRAIN LOSS STATS ==========")
    print(f"Min Loss  : {all_losses.min():.6f}")
    print(f"Mean Loss : {all_losses.mean():.6f}")
    print(f"Max Loss  : {all_losses.max():.6f}")

    threshold = np.percentile(
        all_losses,
        THRESHOLD_PERCENTILE
    )

    print(
        f"\nThreshold ({THRESHOLD_PERCENTILE}th percentile): "
        f"{threshold:.6f}"
    )

    return threshold



def train_vae(
    train_loader,
    val_loader,
    train_tensor,
    val_tensor,
    run_dir,
):
    """
    Trains the VAE and saves the best model.
    """

    model = VAE(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")

    best_model_path = os.path.join(
        run_dir,
        "vae.pth",
    )

    counter = 0

    history = []

    for epoch in range(EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss = validate_epoch(
            model,
            val_loader,
            device,
        )

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            counter = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print("✓ Best model saved.")

        else:

            counter += 1

            print(
                f"Validation not improved "
                f"({counter}/{PATIENCE})"
            )

            if counter >= PATIENCE:

                print("\nEarly stopping triggered.")

                break

    
    # Load Best Model
    

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
        )
    )

    print("\nBest model loaded successfully.")

    
    # Compute Threshold
    

    threshold = compute_threshold(
        model=model,
        train_tensor=train_tensor,
        val_tensor=val_tensor,
        device=device,
    )

    
    # Save Threshold
    
    threshold_path = os.path.join(
        run_dir,
        "threshold.json",
    )

    with open(threshold_path, "w") as f:

        json.dump(
            {
                "threshold": float(threshold),
                "percentile": THRESHOLD_PERCENTILE,
                "loss": "Huber",
            },
            f,
            indent=4,
        )

    
    # Save Training Config
   

    config = {

        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,

        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,

        "epochs": EPOCHS,

        "dropout": DROPOUT,

        "beta": BETA,

        "threshold_percentile": THRESHOLD_PERCENTILE,

        "best_validation_loss": float(best_val_loss),
    }

    config_path = os.path.join(
        run_dir,
        "train_config.json",
    )

    with open(config_path, "w") as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    
    # Save Training History
    

    history_path = os.path.join(
        run_dir,
        "history.csv",
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    
    # Save Summary
    

    summary = {

        "best_validation_loss": float(best_val_loss),

        "threshold": float(threshold),

        "train_samples": len(train_tensor),

        "validation_samples": len(val_tensor),

        "total_samples": len(train_tensor) + len(val_tensor),

        "epochs_completed": len(history),

        "early_stopped": counter >= PATIENCE,
    }

    summary_path = os.path.join(
        run_dir,
        "summary.json",
    )

    with open(summary_path, "w") as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

   
    # Final Summary
    

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    print(f"Best Validation Loss : {best_val_loss:.6f}")
    print(f"Threshold            : {threshold:.6f}")

    print(f"\nOutputs saved to:\n{run_dir}")

    print(f"Model          : {best_model_path}")
    print(f"Threshold      : {threshold_path}")
    print(f"Config         : {config_path}")
    print(f"History        : {history_path}")
    print(f"Summary        : {summary_path}")

    return model, threshold


if __name__ == "__main__":

    run_dir = get_next_run_dir(OUTPUT_ROOT)

    (
        train_tensor,
        val_tensor,
        train_timestamps,
        val_timestamps,
        scaler,
    ) = prepare_data(
        DATA_PATH,
        run_dir,
    )

    train_loader, val_loader = create_dataloaders(
        train_tensor,
        val_tensor,
        BATCH_SIZE,
    )

    model, threshold = train_vae(
        train_loader=train_loader,
        val_loader=val_loader,
        train_tensor=train_tensor,
        val_tensor=val_tensor,
        run_dir=run_dir,
    )


