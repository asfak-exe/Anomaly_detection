# Industrial Anomaly Detection

## Overview

This project is an unsupervised industrial anomaly detection framework designed to identify abnormal behavior in multivariate time-series sensor data. The system learns normal operating patterns using a Variational Autoencoder (VAE) and detects anomalies based on reconstruction error. An optional Latent Adaptive Rumination Architecture (LARA) module further improves the quality of latent representations.

The project also includes an interactive Streamlit dashboard for inference, visualization, and anomaly analysis.

---

## Problem Statement

Modern industrial plants generate large volumes of sensor data continuously. Detecting abnormal operating conditions manually is challenging due to the high dimensionality of the data and the complex relationships between sensors.

This project aims to:

* Learn normal operating behavior from historical sensor data.
* Detect anomalies without requiring labeled datasets.
* Provide interpretable visualizations of detected events.
* Enable easy deployment through an interactive dashboard.

---

## Features

* Variational Autoencoder (VAE) for unsupervised anomaly detection
* LARA-based latent representation refinement
* Automated preprocessing and feature scaling
* Reconstruction error-based anomaly scoring
* Dynamic threshold-based anomaly detection
* Sensor-level anomaly localization
* Interactive Streamlit dashboard
* Automatic anomaly report generation with visualizations

---

## Project Structure

```text
.
├── app_core.py
├── Combined_lara.py
├── evaluate.py
├── inference.py
├── inject_anomaly.py
├── main.py
├── plotting_final.py
├── streamlit_app.py
├── train_with_val.py
├── vae.py
├── requirements.txt
├── README.md
└── .streamlit/
```

---

## Model Pipeline

1. Load multivariate industrial sensor data.
2. Preprocess and normalize features.
3. Train the Variational Autoencoder on normal operating data.
4. Learn latent representations of healthy system behavior.
5. Compute reconstruction error for unseen samples.
6. Apply anomaly thresholding.
7. Generate anomaly reports and visualizations.
8. Optionally refine latent representations using LARA.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Anomaly_detection.git
cd Anomaly_detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python train_with_val.py
```

---

## Inference

Run:

```bash
python inference.py
```

The inference pipeline:

* Loads the trained model
* Computes reconstruction errors
* Detects anomalies
* Generates prediction outputs

---

## Evaluation

Run:

```bash
python evaluate.py
```

---

## Streamlit Dashboard

Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard allows users to:

* Upload industrial sensor data
* Run anomaly detection
* Visualize reconstruction errors
* Inspect detected anomaly events
* Explore generated reports

---

## Future Improvements

* Real-time streaming anomaly detection
* Online model adaptation
* Explainable AI for anomaly interpretation
* Integration with industrial IoT platforms
* Cloud deployment and monitoring

---

## Disclaimer

This repository contains an implementation of an industrial anomaly detection framework for research, educational, and demonstration purposes. Sensitive datasets, trained model weights, and generated outputs are not included.

---


