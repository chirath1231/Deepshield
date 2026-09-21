# 🛡️ DeepShield

DeepShield is a Gradio dashboard that combines two independent deep-learning
systems for network intelligence:

- **Network Intrusion Detection** — a deep neural network that classifies
  network flows into one of 10 categories (benign or attack type).
- **Network Traffic Forecasting** — an LSTM/GRU model that forecasts future
  hourly traffic volume from historical time-series data.

It also includes a **Live Network Monitor** tab that reports local OS-level
network statistics (interface throughput, packet counts, connection count)
for situational awareness — this is displayed separately and is **not**
fed into the intrusion model (see [Limitations](#limitations) below).

## Features

- 🔐 **Intrusion Detection** — pick an attack category, DeepShield samples a
  held-out test flow of that type and classifies it, showing predicted
  class, confidence, and per-class probabilities.
- 🎲 **Random Traffic Test** — classify a random unseen flow from the
  held-out set.
- 📡 **Traffic Forecasting** — forecast 1–24 hours of future traffic from the
  most recent 24 hours of history, with an interactive Plotly chart.
- 🌐 **Live Network Monitor** — read current machine network stats (upload/
  download rate, active interfaces, connection count).
- 🧠 **Models tab** — architecture overview and evaluation metrics for both
  models.

## Models

### Intrusion Detection (UNSW-NB15)

- **Architecture:** Deep neural network (dense layers) with a preprocessing
  pipeline for numeric + categorical flow features.
- **Classes (10):** Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic,
  Normal, Reconnaissance, Shellcode, Worms
- **Features:** 42 flow-level features
- **Test Accuracy:** ~0.73
- **Macro F1:** ~0.44

### Traffic Forecasting

- **Dataset:** [fedesoriano/traffic-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset) (hourly vehicle counts, Junction 1)
- **Architecture:** LSTM (selected over GRU on validation loss)
- **Lookback window:** 24 hours
- **Test MAE:** ~4.73
- **Test RMSE:** ~6.51
- **Test R²:** ~0.93

## Project structure

```
DeepShield/
├── app.py                             # Gradio app entry point
├── requirements.txt
├── deepshield_artifacts/              # Intrusion detection model + metadata
│   ├── deepshield_model.keras
│   ├── preprocessor.joblib
│   ├── label_encoder.joblib
│   ├── train_medians.joblib
│   ├── metadata.json
│   ├── per_class_metrics.csv
│   └── heldout_demo_flows.csv
└── deepshield_traffic_artifacts/      # Traffic forecasting model + metadata
    ├── traffic_forecast_model.keras
    ├── traffic_scaler.joblib
    ├── traffic_metadata.json
    ├── model_comparison.csv
    ├── recent_traffic_demo.csv
    └── example_future_forecast.csv
```

## Setup

Requires Python 3.10 (the `.keras` model artifacts were saved with a newer
Keras that writes fields older versions reject; `app.py` patches around this
automatically at load time, but scikit-learn is pinned for the same reason).

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

## Limitations

- The intrusion detection model is trained on the **UNSW-NB15** dataset's
  per-flow feature schema. The Live Network Monitor tab reports raw OS
  network counters (bytes/packets per second, interface state), which are
  **not** in that schema — there is no flow-feature extractor here to bridge
  live traffic into the model, so live traffic is never classified as an
  intrusion by this app. Intrusion detection is demonstrated only against
  held-out UNSW-NB15 test flows.
- Macro F1 (~0.44) is noticeably lower than accuracy (~0.73), reflecting
  class imbalance across the 10 attack categories in the training data.
