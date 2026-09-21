import os
import sys
import json
import joblib
import tempfile
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr
import plotly.graph_objects as go
import time
import socket
import psutil


# The status lines below use box-drawing and check characters, which
# the default Windows console codepage (cp1252) cannot encode.

if hasattr(sys.stdout, "reconfigure"):

    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

IDS_DIR = os.path.join(
    BASE_DIR,
    "deepshield_artifacts"
)

TRAFFIC_DIR = os.path.join(
    BASE_DIR,
    "deepshield_traffic_artifacts"
)

# ============================================================
# MODEL LOADING COMPATIBILITY
# ============================================================

# The .keras artifacts were saved with Keras 3.13, which writes a
# "quantization_config" key into every layer config. Older Keras
# versions (3.12 and below, the newest that supports Python 3.10)
# reject that key. Strip it into a temporary copy so the original
# artifacts stay untouched.


def load_keras_model(path):

    try:

        return tf.keras.models.load_model(path)

    except (TypeError, ValueError):

        pass

    def strip(node):

        if isinstance(node, dict):

            if node.get("quantization_config", "missing") is None:
                node.pop("quantization_config")

            for value in node.values():
                strip(value)

        elif isinstance(node, list):

            for value in node:
                strip(value)

    with zipfile.ZipFile(path) as archive:

        config = json.loads(
            archive.read("config.json")
        )

        strip(config)

        patched = os.path.join(
            tempfile.mkdtemp(),
            os.path.basename(path)
        )

        with zipfile.ZipFile(patched, "w") as out:

            for item in archive.infolist():

                if item.filename == "config.json":

                    out.writestr(
                        item,
                        json.dumps(config)
                    )

                else:

                    out.writestr(
                        item,
                        archive.read(item.filename)
                    )

    return tf.keras.models.load_model(patched)


# ============================================================
# LOAD INTRUSION DETECTION MODEL
# ============================================================

print("Loading DeepShield Intrusion Detection Model...")


ids_model = load_keras_model(
    os.path.join(
        IDS_DIR,
        "deepshield_model.keras"
    )
)


ids_preprocessor = joblib.load(
    os.path.join(
        IDS_DIR,
        "preprocessor.joblib"
    )
)


ids_label_encoder = joblib.load(
    os.path.join(
        IDS_DIR,
        "label_encoder.joblib"
    )
)


ids_medians = joblib.load(
    os.path.join(
        IDS_DIR,
        "train_medians.joblib"
    )
)


with open(
    os.path.join(
        IDS_DIR,
        "metadata.json"
    ),
    "r"
) as f:

    ids_metadata = json.load(f)


ids_demo = pd.read_csv(
    os.path.join(
        IDS_DIR,
        "heldout_demo_flows.csv"
    )
)


class_names = ids_metadata["classes"]

feature_columns = ids_metadata[
    "feature_columns"
]

numeric_cols = ids_metadata[
    "numeric_columns"
]

categorical_cols = ids_metadata[
    "categorical_columns"
]


print(
    "✓ Intrusion Detection Model Loaded"
)


# ============================================================
# LOAD TRAFFIC MODEL
# ============================================================

print(
    "Loading Traffic Forecasting Model..."
)


traffic_model = load_keras_model(
    os.path.join(
        TRAFFIC_DIR,
        "traffic_forecast_model.keras"
    )
)


traffic_scaler = joblib.load(
    os.path.join(
        TRAFFIC_DIR,
        "traffic_scaler.joblib"
    )
)


with open(
    os.path.join(
        TRAFFIC_DIR,
        "traffic_metadata.json"
    ),
    "r"
) as f:

    traffic_metadata = json.load(f)


recent_traffic = pd.read_csv(
    os.path.join(
        TRAFFIC_DIR,
        "recent_traffic_demo.csv"
    )
)


recent_traffic["DateTime"] = pd.to_datetime(
    recent_traffic["DateTime"]
)


LOOKBACK = int(
    traffic_metadata["lookback_hours"]
)


print(
    "✓ Traffic Forecasting Model Loaded"
)


# ============================================================
# INTRUSION PREDICTION ENGINE
# ============================================================

def predict_intrusion(row_df):

    row = row_df[
        feature_columns
    ].copy()


    row.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    row[numeric_cols] = (
        row[numeric_cols]
        .fillna(ids_medians)
    )


    for column in categorical_cols:

        row[column] = (
            row[column]
            .fillna("unknown")
            .astype(str)
        )


    processed = ids_preprocessor.transform(
        row
    )


    processed = np.asarray(
        processed,
        dtype=np.float32
    )


    probabilities = ids_model.predict(
        processed,
        verbose=0
    )[0]


    predicted_id = int(
        np.argmax(probabilities)
    )


    predicted_class = class_names[
        predicted_id
    ]


    confidence = float(
        probabilities[predicted_id]
    )


    probability_dict = {

        class_names[i]:
        float(probabilities[i])

        for i in range(
            len(class_names)
        )
    }


    return (
        predicted_class,
        confidence,
        probability_dict
    )


# ============================================================
# MANUAL ATTACK TEST
# ============================================================

def test_attack(
    attack_type
):

    available = ids_demo[
        ids_demo["attack_cat"]
        == attack_type
    ]


    if available.empty:

        return (
            "No sample available",
            "",
            "",
            {},
            pd.DataFrame()
        )


    sample = available.sample(
        n=1
    ).copy()


    actual_class = str(
        sample.iloc[0][
            "attack_cat"
        ]
    )


    model_input = sample[
        feature_columns
    ].copy()


    predicted_class, confidence, probabilities = (
        predict_intrusion(
            model_input
        )
    )


    if predicted_class == actual_class:

        status = (
            "✅ CORRECT PREDICTION"
        )

    else:

        status = (
            "❌ INCORRECT PREDICTION"
        )


    confidence_text = (
        f"{confidence * 100:.2f}%"
    )


    useful_columns = [

        "proto",
        "service",
        "state",
        "dur",
        "sbytes",
        "dbytes",
        "rate",
        "sttl",
        "dttl"

    ]


    useful_columns = [

        column

        for column
        in useful_columns

        if column
        in sample.columns
    ]


    preview = sample[
        useful_columns
    ]


    return (

        status,

        actual_class,

        f"{predicted_class} "
        f"({confidence_text})",

        probabilities,

        preview

    )


# ============================================================
# RANDOM INTRUSION TEST
# ============================================================

def random_intrusion_test():

    sample = ids_demo.sample(
        n=1
    ).copy()


    actual = str(
        sample.iloc[0][
            "attack_cat"
        ]
    )


    predicted, confidence, probabilities = (
        predict_intrusion(
            sample
        )
    )


    if predicted == actual:

        status = "✅ CORRECT"

    else:

        status = "❌ INCORRECT"


    return (

        status,

        actual,

        predicted,

        f"{confidence * 100:.2f}%",

        probabilities

    )


# ============================================================
# TRAFFIC FORECASTING
# ============================================================

def forecast_next(
    recent_values
):

    recent_values = np.asarray(
        recent_values,
        dtype=np.float32
    )


    scaled = traffic_scaler.transform(
        recent_values.reshape(-1, 1)
    )


    X = scaled.reshape(
        1,
        LOOKBACK,
        1
    )


    prediction_scaled = traffic_model.predict(
        X,
        verbose=0
    )[0][0]


    prediction = (
        traffic_scaler.inverse_transform(
            [[prediction_scaled]]
        )[0][0]
    )


    return max(
        0,
        float(prediction)
    )


# ============================================================
# MULTI-HOUR FORECAST
# ============================================================

def forecast_future(
    hours
):

    hours = int(hours)


    recent_values = (
        recent_traffic[
            "Vehicles"
        ]
        .tail(LOOKBACK)
        .to_numpy(
            dtype=np.float32
        )
    )


    window = list(
        recent_values
    )


    forecasts = []


    for _ in range(hours):

        prediction = forecast_next(
            window[-LOOKBACK:]
        )


        forecasts.append(
            prediction
        )


        window.append(
            prediction
        )


    last_time = recent_traffic[
        "DateTime"
    ].iloc[-1]


    future_times = pd.date_range(

        start=(
            last_time
            + pd.Timedelta(hours=1)
        ),

        periods=hours,

        freq="h"

    )


    forecast_df = pd.DataFrame({

        "DateTime":
        future_times,

        "Predicted Vehicles":
        np.round(
            forecasts,
            2
        )

    })


    # ========================================================
    # PLOT
    # ========================================================

    history = recent_traffic.tail(
        48
    )


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=history["DateTime"],

            y=history["Vehicles"],

            mode="lines+markers",

            name="Historical Traffic"

        )

    )


    fig.add_trace(

        go.Scatter(

            x=future_times,

            y=forecasts,

            mode="lines+markers",

            name="DeepShield Forecast"

        )

    )


    fig.update_layout(

        title=(
            "Network Traffic Forecast"
        ),

        xaxis_title="Time",

        yaxis_title=(
            "Vehicles per Hour"
        ),

        hovermode="x unified",

        template="plotly_dark"

    )


    next_hour = forecasts[0]


    average = np.mean(
        forecasts
    )


    peak = np.max(
        forecasts
    )


    peak_index = int(
        np.argmax(
            forecasts
        )
    )


    peak_time = future_times[
        peak_index
    ]


    summary = f"""
### 📡 Traffic Forecast

**Next Hour:** {next_hour:.2f} vehicles

**Forecast Average:** {average:.2f} vehicles/hour

**Forecast Peak:** {peak:.2f} vehicles

**Expected Peak Time:** {peak_time.strftime('%Y-%m-%d %H:%M')}
"""


    return (
        summary,
        fig,
        forecast_df
    )


# ============================================================
# LIVE NETWORK MONITOR
# ============================================================

def get_live_network_stats(interval=1.0):
    """Measure local interface counters over a short interval.

    This is monitoring only. These OS counters are not passed to the
    UNSW-NB15 classifier because they do not match its trained flow schema.
    """
    before = psutil.net_io_counters()
    time.sleep(float(interval))
    after = psutil.net_io_counters()

    upload_bps = (after.bytes_sent - before.bytes_sent) / float(interval)
    download_bps = (after.bytes_recv - before.bytes_recv) / float(interval)

    active_interfaces = []
    for name, stats in psutil.net_if_stats().items():
        if stats.isup:
            active_interfaces.append(name)

    try:
        connection_count = len(psutil.net_connections(kind="inet"))
    except (psutil.AccessDenied, PermissionError):
        connection_count = -1

    hostname = socket.gethostname()

    summary = f"""
### 🟢 Live Network Monitor

**Host:** {hostname}

**Active interfaces:** {", ".join(active_interfaces) if active_interfaces else "None detected"}

**Download rate:** {download_bps / 1024:.2f} KB/s

**Upload rate:** {upload_bps / 1024:.2f} KB/s

**Total received:** {after.bytes_recv / (1024**2):.2f} MB

**Total sent:** {after.bytes_sent / (1024**2):.2f} MB

**Packets received:** {after.packets_recv:,}

**Packets sent:** {after.packets_sent:,}

**Internet connections:** {connection_count if connection_count >= 0 else "Permission required"}

> Live OS statistics are displayed separately from IDS inference. The trained intrusion model expects UNSW-NB15-compatible per-flow features, so these aggregate counters are not falsely presented as model predictions.
"""

    rows = []
    addrs = psutil.net_if_addrs()
    stats_map = psutil.net_if_stats()
    for name, stats in stats_map.items():
        if not stats.isup:
            continue
        ipv4 = []
        for addr in addrs.get(name, []):
            if addr.family == socket.AF_INET:
                ipv4.append(addr.address)
        rows.append({
            "Interface": name,
            "Status": "UP",
            "Speed (Mbps)": stats.speed if stats.speed >= 0 else "Unknown",
            "IPv4": ", ".join(ipv4) if ipv4 else "—",
        })

    return summary, pd.DataFrame(rows)


# ============================================================
# MODEL INFORMATION
# ============================================================

ids_accuracy = ids_metadata.get(
    "test_accuracy",
    0
)

ids_f1 = ids_metadata.get(
    "macro_f1",
    0
)

traffic_mae = traffic_metadata.get(
    "test_mae",
    0
)

traffic_rmse = traffic_metadata.get(
    "test_rmse",
    0
)

traffic_r2 = traffic_metadata.get(
    "test_r2",
    0
)


# ============================================================
# GRADIO
# ============================================================

with gr.Blocks(
    title="DeepShield"
) as app:


    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        """
# 🛡️ DeepShield

## Deep Learning Network Security & Traffic Intelligence

DeepShield combines two independent deep-learning
systems:

**🔐 Network Intrusion Detection**

Detect and classify malicious network traffic.

**📡 Network Traffic Forecasting**

Forecast future network traffic from historical
time-series observations.
"""
    )


    # ========================================================
    # OVERVIEW
    # ========================================================

    with gr.Tab(
        "🏠 Overview"
    ):


        gr.Markdown(
            f"""
## DeepShield System

### 🔐 Security Intelligence

**Model:** Deep Neural Network

**Attack Classes:** {len(class_names)}

**Features:** {len(feature_columns)}

**Test Accuracy:** {ids_accuracy:.4f}

**Macro F1:** {ids_f1:.4f}

---

### 📡 Traffic Intelligence

**Model:** {traffic_metadata.get("best_model", "LSTM/GRU")}

**Historical Window:** {LOOKBACK} hours

**Forecast Target:** Vehicles per hour

**Test MAE:** {traffic_mae:.3f}

**Test RMSE:** {traffic_rmse:.3f}

**Test R²:** {traffic_r2:.3f}
"""
        )


    # ========================================================
    # INTRUSION DETECTION
    # ========================================================

    with gr.Tab(
        "🔐 Intrusion Detection"
    ):


        gr.Markdown(
            """
## Multiclass Network Intrusion Detection

Select a traffic category.

DeepShield selects a **held-out test flow**
belonging to that category.

The actual class is NOT passed to the neural
network.
"""
        )


        attack_dropdown = gr.Dropdown(

            choices=class_names,

            value=class_names[0],

            label="Traffic / Attack Type"

        )


        analyze_button = gr.Button(

            "🔍 Analyze Traffic",

            variant="primary"

        )


        ids_status = gr.Textbox(
            label="Result"
        )


        with gr.Row():


            ids_actual = gr.Textbox(
                label="Actual Class"
            )


            ids_prediction = gr.Textbox(
                label="DeepShield Prediction"
            )


        ids_probabilities = gr.Label(

            num_top_classes=
            len(class_names),

            label="Class Probabilities"

        )


        ids_flow = gr.Dataframe(

            label="Network Flow"

        )


        analyze_button.click(

            fn=test_attack,

            inputs=[
                attack_dropdown
            ],

            outputs=[

                ids_status,

                ids_actual,

                ids_prediction,

                ids_probabilities,

                ids_flow

            ]

        )


        gr.Markdown(
            "### Random Unseen Traffic"
        )


        random_button = gr.Button(
            "🎲 Random Traffic Test"
        )


        random_status = gr.Textbox(
            label="Result"
        )


        with gr.Row():


            random_actual = gr.Textbox(
                label="Actual"
            )


            random_prediction = gr.Textbox(
                label="Prediction"
            )


            random_confidence = gr.Textbox(
                label="Confidence"
            )


        random_probs = gr.Label(
            label="Probabilities"
        )


        random_button.click(

            fn=random_intrusion_test,

            inputs=[],

            outputs=[

                random_status,

                random_actual,

                random_prediction,

                random_confidence,

                random_probs

            ]

        )


    # ========================================================
    # TRAFFIC FORECASTING
    # ========================================================

    with gr.Tab(
        "📡 Traffic Forecasting"
    ):


        gr.Markdown(
            f"""
## Deep Learning Traffic Forecast

The model uses the previous **{LOOKBACK} hours**
of traffic to forecast future traffic.

The training model selected was:

### {traffic_metadata.get("best_model", "LSTM/GRU")}
"""
        )


        forecast_hours = gr.Slider(

            minimum=1,

            maximum=24,

            value=12,

            step=1,

            label="Forecast Hours"

        )


        forecast_button = gr.Button(

            "📈 Generate Forecast",

            variant="primary"

        )


        forecast_summary = gr.Markdown()


        forecast_plot = gr.Plot(
            label="Traffic Forecast"
        )


        forecast_table = gr.Dataframe(

            headers=[
                "DateTime",
                "Predicted Vehicles"
            ],

            label="Future Predictions"

        )


        forecast_button.click(

            fn=forecast_future,

            inputs=[
                forecast_hours
            ],

            outputs=[

                forecast_summary,

                forecast_plot,

                forecast_table

            ]

        )


    # ========================================================
    # LIVE NETWORK MONITOR
    # ========================================================

    with gr.Tab("🌐 Live Network Monitor"):

        gr.Markdown(
            """
## Current Computer Network Connection

This tab reads **local Windows network statistics** such as transfer rate,
packet counters, interfaces, and connection count.

It does **not** feed these aggregate counters directly into the intrusion DNN.
The intrusion model was trained on UNSW-NB15 per-flow features and requires a
compatible flow-feature extractor before true live IDS classification can be
performed.
"""
        )

        live_refresh = gr.Button("🔄 Check Current Network", variant="primary")
        live_summary = gr.Markdown()
        live_interfaces = gr.Dataframe(label="Active Network Interfaces")

        live_refresh.click(
            fn=get_live_network_stats,
            inputs=[],
            outputs=[live_summary, live_interfaces]
        )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with gr.Tab(
        "🧠 Models"
    ):


        gr.Markdown(
            """
# DeepShield Architecture

## Model 1 — Intrusion Detection

Raw Network Flow

↓

Feature Preprocessing

↓

Deep Neural Network

↓

10-Class Softmax

↓

Attack Classification


## Model 2 — Traffic Forecasting

Historical Traffic

↓

24-Hour Sequence

↓

LSTM / GRU

↓

Regression Layer

↓

Future Traffic Forecast
"""
        )


        gr.Markdown(
            f"""
### Intrusion Detection

Accuracy: **{ids_accuracy:.4f}**

Macro F1: **{ids_f1:.4f}**


### Traffic Forecasting

Model: **{traffic_metadata.get("best_model", "Unknown")}**

MAE: **{traffic_mae:.3f}**

RMSE: **{traffic_rmse:.3f}**

R²: **{traffic_r2:.3f}**
"""
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DEEPSHIELD")
    print("Deep Learning Network Intelligence")
    print("=" * 60)

    print(
        "Open your browser:"
    )

    print(
        "http://127.0.0.1:7860"
    )

    app.launch(

        server_name="127.0.0.1",

        server_port=7860,

        share=False

    )