import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
# from autoencoder import Autoencoder
from vae import VAE
from data.data_loader import prepare_data
import numpy as np

import matplotlib.pyplot as plt



df_clean = pd.read_csv(
    r"C:\Users\6000_J1I\Desktop\asfak_project\cleaned_data.csv")

df = df_clean.copy()

train_tensor, test_tensor, scaler = prepare_data(df)


def train_autoencoder_final(
    train_tensor,
    input_dim=944,
    latent_dim=32,
    batch_size=128,
    lr=1e-3,
    epochs=20,
    dropout=0.4,
    save_dir=r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\outputs\autoencoders_final",
    best_model_path=r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\outputs\autoencoders_final"
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)

  
    # Chronological Train / Validation Split
  

    split_idx = int(0.8 * len(train_tensor))

    train_data = train_tensor[:split_idx]
    val_data = train_tensor[split_idx:]

    train_dataset = TensorDataset(
        train_data,
        train_data
    )

    val_dataset = TensorDataset(
        val_data,
        val_data
    )

    
    # Validation Loader


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.HuberLoss()  

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode="min",
        factor=0.5,
        patience=3,        
    )





   
    # Best model now based on validation loss


    best_val_loss = float("inf")

    best_model_path = os.path.join(
        save_dir,
        "vae_final3.pth"
    )

  
    # Early Stopping


    patience = 5
    counter = 0

    for epoch in range(epochs):

      
        # TRAIN
        

        model.train()

        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            # outputs = model(batch_x)
            recon_x, mu, logvar = model(batch_x)

            recon_loss = criterion(recon_x, batch_x)
            kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1))

            loss = recon_loss + 0.001*kl_div  

            loss.backward()

            optimizer.step()

            running_train_loss += (
                loss.item() * batch_x.size(0)
            )

        train_loss = (
            running_train_loss /
            len(train_dataset)
        )

       

        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for batch_x, batch_y in val_loader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                recon_x, mu, logvar = model(batch_x)
                recon_loss = criterion(recon_x, batch_y)
                kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

                loss = recon_loss + 0.001*kl_div

                # outputs = model(batch_x)

                # loss = criterion(
                #     outputs,
                #     batch_y
                # )

                running_val_loss += (
                    loss.item() * batch_x.size(0)
                )

        val_loss = (
            running_val_loss /
            len(val_dataset)
        )
        
        scheduler.step(val_loss) 

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

       
        # Save best model using
        # validation loss
      

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            counter = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )

        else:

            counter += 1

            print(
                f"Validation not improved "
                f"({counter}/{patience})"
            )

            if counter >= patience:

                print(
                    "\nEarly stopping triggered."
                )
 
                break

   
    # Save Config
    

    config = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs
    }

    with open(
        os.path.join(
            save_dir,
            "train_config3.json"
        ),
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print(
        f"\nSaved best model to: "
        f"{best_model_path}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )


    # ==================================================
    # Load Best Model
    # ==================================================

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    model.eval()

    all_losses = []


    # Threshold Calculation
    
    with torch.no_grad():

     for batch_x, _ in train_loader:     #FOR huber loss

        batch_x = batch_x.to(device)

        recon_x, _, _ = model(batch_x)

        error = recon_x - batch_x

        delta = 1.0

        abs_error = torch.abs(error)

        per_element_huber = torch.where(
            abs_error <= delta,
            0.5 * error**2,
            delta * (abs_error - 0.5 * delta)
        )

        per_sample_huber = per_element_huber.mean(dim=1)

        all_losses.extend(
            per_sample_huber.cpu().numpy()
        )

        



    # with torch.no_grad():            #FOR MSE

    #     for batch_x, _ in train_loader:

    #         batch_x = batch_x.to(device)

    #         outputs = model(batch_x)

    #         per_sample_loss = torch.mean(
    #             (outputs - batch_x) ** 2,
    #             dim=1
    #         )

    #         all_losses.extend(
    #             per_sample_loss
    #             .cpu()
    #             .numpy()
    #         )

    all_losses = np.array(
        all_losses
    )
    print("\nTrain Loss Stats")
    print(f"Min Loss       : {np.min(all_losses):.6f}")
    print(f"Mean Loss      : {np.mean(all_losses):.6f}")
    print(f"Max Loss       : {np.max(all_losses):.6f}")

    threshold = np.percentile(
        all_losses,
        99
    )

    print(
        f"Threshold "
        f"(99th percentile): "
        f"{threshold:.6f}"
    )

    with open(
        os.path.join(
            save_dir,
            "threshold3.json"
        ),
        "w"
    ) as f:

        json.dump(
            {"threshold": float(threshold)},
            f,
            indent=4
        )

    print(
        f"Threshold saved to: "
        f"{os.path.join(save_dir, 'threshold_3.json')}"
    )


    test_dataset = TensorDataset(
    test_tensor,
    test_tensor
)

    test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Load Model


    model = VAE(
    input_dim=944,
    latent_dim=32,
    dropout=0.3
).to(device)

    model.load_state_dict(
    torch.load(
        r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\outputs\autoencoders_final\vae_final3.pth",
        map_location=device
    )
)

    model.eval()

    print("Model loaded successfully.")


# Evaluation


    criterion = nn.HuberLoss()

    running_loss_test = 0.0

    all_sample_losses_test = []

    with torch.no_grad():

     for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        recon_x,mu,logvar= model(batch_x)

        loss = criterion(
            recon_x,
            batch_y
        )

        running_loss_test += (
            loss.item() * batch_x.size(0)
        )
        error = recon_x - batch_x
        delta = 1.0
        abs_error = torch.abs(error)

        per_element_huber = torch.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta))
        
        per_sample_huber = per_element_huber.mean(dim=1)

        all_sample_losses_test.extend(
            per_sample_huber.cpu().numpy()
        )


# Results


    test_mse = (
    running_loss_test /
    len(test_dataset)
)

    test_rmse = test_mse ** 0.5

    avg_sample_loss = (
    sum(all_sample_losses_test) /
    len(all_sample_losses_test)
)
    all_sample_losses_test = np.array(
    all_sample_losses_test
)

    num_anomalies = np.sum(
    all_sample_losses_test > threshold
)
    # print(np.percentile(all_losses, 50))
    # print(np.percentile(all_losses, 75))
    # print(np.percentile(all_losses, 90))
    # print(np.percentile(all_losses, 95))
    # print(np.percentile(all_losses, 99))

    print("\n========== TEST RESULTS ==========")
    print(f"Test MSE       : {test_mse:.6f}")
    print(f"Test RMSE      : {test_rmse:.6f}")
    print(f"Avg Sample MSE : {avg_sample_loss:.6f}")
    print(f"Number of Anomalies : {num_anomalies}")
    print(f"Threshold           : {threshold:.6f}")
    print(f"Total Test Samples  : {len(all_sample_losses_test)}")
    
    print(f"Anomaly Percentage  : {(100*num_anomalies/len(all_sample_losses_test)):.2f}%")
    
  # Save reconstruction error for every timestamp
# ===========================================

    split_idx = int(0.8 * len(df))

    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    recon_df = pd.DataFrame({
    "Timestamp": test_df["Timestamp"],
    "Reconstruction_Error": all_sample_losses_test
})
    recon_df["Threshold"] = threshold

    recon_df["Is_Anomaly"] = (
    recon_df["Reconstruction_Error"] > threshold
).astype(int)

    recon_df.to_csv(
    "reconstruction_errors.csv",
    index=False
)

    print("Saved reconstruction_errors.csv")
    split_idx = int(0.8 * len(df))

    test_df = df.iloc[split_idx:].copy()
    anomaly_idx = np.where(all_sample_losses_test > threshold)[0]
    anomaly_report = test_df.iloc[anomaly_idx]

    anomaly_report.to_csv(
    "anomalies_2.csv",
    index=False
)
    


    return model, best_val_loss



model, best_loss = train_autoencoder_final(
    train_tensor=train_tensor,
    input_dim=944,
    latent_dim=32,
    batch_size=128,
    lr=1e-4,
    epochs=20,
    dropout=0.3
)
 

