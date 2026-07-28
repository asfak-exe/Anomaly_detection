# Industrial Anomaly Detection Dashboard

Streamlit dashboard for scoring industrial sensor CSV files with the trained VAE in `output/run_10`. It also supports optional LARA adaptation for an uploaded data distribution.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep `output/run_10/vae.pth`, `scaler.pkl`, `train_config.json`, and `threshold.json` in the repository; they are needed at runtime.
2. In Streamlit Community Cloud, create an app from that repository and select `streamlit_app.py` as the entry point.
3. Deploy. No secrets are required for the current local-model workflow.

Large generated plot folders are excluded through `.gitignore`; they are not required by the dashboard.
