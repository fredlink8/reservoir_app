#!/usr/bin/env python
# coding: utf-8

import os
import random
import re
import unicodedata
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib import rc

import streamlit as st
import streamlit.components.v1 as components

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from streamlit.runtime.uploaded_file_manager import UploadedFile


# ============================================================
# Page setup
# ============================================================
st.set_page_config(layout="wide")

st.title("Reservoir Operation Decision-Support Dashboard")
st.markdown(
    "Climate-driven reservoir prediction and optimization framework with "
    "interactive hydrograph visualization, storage analysis, and selectable "
    "4-panel reservoir operation animation for decision support."
)


# ============================================================
# Reproducibility
# ============================================================
def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds(42)
tf.keras.backend.set_floatx("float32")
rc("animation", html="jshtml")


# ============================================================
# Global parameters
# ============================================================
DELTA_T = 86400.0
FLOOD_CONTROL_LIMIT = 2130.0
CAP_DEAD = 280.0

TRAIN_STRIDE = 5
DEFAULT_PPO_EPOCHS = 12
LR_ACTOR = 2e-4
LR_CRITIC = 5e-4
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
SCALE_CAP = 0.50

SMOOTH_WIN = 3
FPS = 2


# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("📁 Data Source Setup")

input_method = st.sidebar.selectbox(
    "Choose Input Data Mode",
    ["📂 Upload Raw Excel Files", "📋 Copy & Paste / Manual Typing"]
)

st.sidebar.markdown("---")
st.sidebar.header("🏋️ Neural Layer Controls")

epochs = st.sidebar.slider(
    "CNN-LSTM Training Epochs",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

ppo_epochs = st.sidebar.slider(
    "PPO Training Epochs",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Boundary Constraints")

total_storage_capacity = st.sidebar.slider(
    "Total Reservoir Capacity (million m³)",
    min_value=500.0,
    max_value=5000.0,
    value=2900.0,
    step=50.0
)

max_discharge = st.sidebar.slider(
    "Maximum Outflow Limit (m³/s)",
    min_value=1000.0,
    max_value=10000.0,
    value=5750.0,
    step=250.0
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ PPO Reward Weight Tuning")

w_store_slider = st.sidebar.slider(
    "Storage Penalty (W_STORE)",
    min_value=0.1,
    max_value=5.0,
    value=1.5,
    step=0.1
)

w_dev_slider = st.sidebar.slider(
    "Outflow Deviation Penalty (W_DEV)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

w_excess_slider = st.sidebar.slider(
    "Excess Release Penalty (W_EXCESS)",
    min_value=0.1,
    max_value=5.0,
    value=0.6,
    step=0.1
)


# ============================================================
# Default data paths for Streamlit Cloud deployment
# ============================================================
BASE_HISTORICAL_PATH = "data/Cleaned_Reservoir_Data_v5_newV2.xlsx"
BASE_FUTURE_INPUTS_PATH = "data/precip_temp_newv2.xlsx"

# ============================================================
# Utility functions
# ============================================================
def normalize_col(name):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(name))).strip()


def normalize_df(df):
    out = df.copy()
    out.columns = [normalize_col(c) for c in out.columns]
    return out


def pick_col(df, aliases, required=True, name_hint=""):
    for a in aliases:
        if a in df.columns:
            return a

    if required:
        raise KeyError(
            f"Missing required column for {name_hint}. Tried: {aliases}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


@st.cache_data(hash_funcs={UploadedFile: lambda x: x.name if hasattr(x, "name") else str(x)})
def load_excel_source(filepath_or_buffer):
    return normalize_df(pd.read_excel(filepath_or_buffer))


def validate_required_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"{df_name} is missing required columns: {missing}")
        st.stop()


def clean_numeric_columns(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").ffill().bfill()
    return out


def roll_mean(x, w):
    x = np.asarray(x, float)
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def slice_season_df(df, season_name):
    months = df["date"].dt.month

    if season_name == "Annual":
        return df.copy()
    if season_name == "Winter":
        return df[months.isin([12, 1, 2])].copy()
    if season_name == "Spring":
        return df[months.isin([3, 4, 5])].copy()
    if season_name == "Summer":
        return df[months.isin([6, 7, 8])].copy()
    if season_name == "Fall":
        return df[months.isin([9, 10, 11])].copy()

    return df.copy()


def season_from_month(m):
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Fall"


def decade_bounds(decade_label):
    if decade_label == "2026–2035":
        return 2026, 2035, "a"
    if decade_label == "2036–2045":
        return 2036, 2045, "b"
    return 2046, 2055, "c"


def filter_by_group_and_decade(df, group, decade_label):
    y_min, y_max, panel_label = decade_bounds(decade_label)

    out = slice_season_df(df, group)
    out = out[
        (out["date"].dt.year >= y_min) &
        (out["date"].dt.year <= y_max)
    ].copy()

    years_subset = list(range(y_min, y_max + 1))
    return out, years_subset, panel_label, y_min, y_max


# ============================================================
# CNN-LSTM forecast engine
# ============================================================
@st.cache_data(show_spinner=False)
def run_cnn_lstm_forecast(df_historical, df_future_inputs, epochs_num=10):
    set_seeds(42)

    hist_clean = df_historical.drop(
        columns=[col for col in df_historical.columns if "Unnamed" in col],
        errors="ignore"
    ).copy()

    predictor_cols = [
        "remote_sensed_watershed_precipitation (mm)",
        "temperature (°C)"
    ]

    hist_clean = hist_clean.drop(columns=["date"], errors="ignore")

    X_train = hist_clean[predictor_cols].copy()
    y_train = hist_clean.drop(columns=predictor_cols).copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    y_train = y_train.apply(pd.to_numeric, errors="coerce").ffill().bfill()

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_train)
    Y_scaled = y_scaler.fit_transform(y_train)

    timesteps = 7
    X_seq, Y_seq = [], []

    for i in range(timesteps, len(X_scaled)):
        X_seq.append(X_scaled[i - timesteps:i])
        Y_seq.append(Y_scaled[i])

    X_seq = np.array(X_seq)
    Y_seq = np.array(Y_seq)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(timesteps, 2)),
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="causal"
        ),
        tf.keras.layers.LSTM(64, activation="relu"),
        tf.keras.layers.Dense(Y_seq.shape[1])
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, Y_seq, epochs=epochs_num, batch_size=32, verbose=0)

    X_future = df_future_inputs[predictor_cols].copy()
    X_future = X_future.apply(pd.to_numeric, errors="coerce").ffill().bfill()

    X_hist_tail = X_train.tail(timesteps).copy()
    X_combo = pd.concat([X_hist_tail, X_future], ignore_index=True)
    X_combo_scaled = x_scaler.transform(X_combo)

    X_pred_seq = []

    for i in range(timesteps, len(X_combo_scaled)):
        X_pred_seq.append(X_combo_scaled[i - timesteps:i])

    X_pred_seq = np.array(X_pred_seq)

    y_pred_scaled = model.predict(X_pred_seq, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    predicted_data = pd.DataFrame(y_pred, columns=y_train.columns)

    predicted_data.insert(
        0,
        "temperature (°C)",
        X_future["temperature (°C)"].values
    )

    predicted_data.insert(
        0,
        "remote_sensed_watershed_precipitation (mm)",
        X_future["remote_sensed_watershed_precipitation (mm)"].values
    )

    predicted_data.insert(
        0,
        "date",
        df_future_inputs["date"].values
    )

    return predicted_data


# ============================================================
# PPO engine
# ============================================================
def build_models():
    actor = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="tanh")
    ])

    critic = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    opt_actor = tf.keras.optimizers.Adam(LR_ACTOR)
    opt_critic = tf.keras.optimizers.Adam(LR_CRITIC)

    return actor, critic, opt_actor, opt_critic


def clamp_outflow(q_baseline, a_tanh, max_discharge):
    scale = 1.0 + SCALE_CAP * float(a_tanh)
    return float(np.clip(float(q_baseline) * scale, 0.0, max_discharge))


def compute_reward(
    storage,
    inflow,
    q_release,
    q_baseline,
    w_store,
    w_dev,
    w_excess
):
    attenuation = max(0.0, float(inflow) - float(q_release))
    excess = max(0.0, float(q_release) - float(inflow))
    store_violation = max(0.0, float(storage) - FLOOD_CONTROL_LIMIT)
    deviation = abs(float(q_release) - float(q_baseline))

    reward = (
        +0.02 * attenuation
        -w_excess * excess
        -w_dev * deviation
        -w_store * (store_violation ** 2) / 1000.0
        -0.0005 * (q_release ** 2)
    )

    return float(reward)


def gae_advantages(rewards, values):
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)

    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)

    last_gae = 0.0
    next_value = 0.0

    for tt in reversed(range(T)):
        delta = rewards[tt] + GAMMA * next_value - values[tt]
        last_gae = delta + GAMMA * LAMBDA_GAE * last_gae
        adv[tt] = last_gae
        next_value = values[tt]

    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return adv.astype(np.float32), returns.astype(np.float32)


def ppo_update(actor, critic, opt_actor, opt_critic, states, old_logps, adv, returns):
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    old_logps = tf.convert_to_tensor(old_logps, dtype=tf.float32)
    adv = tf.convert_to_tensor(adv, dtype=tf.float32)
    returns = tf.convert_to_tensor(returns, dtype=tf.float32)

    with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
        a_new = actor(states, training=True)[:, 0]
        v_new = critic(states, training=True)[:, 0]

        logp_new = -0.5 * tf.square(a_new)
        ratio = tf.exp(logp_new - old_logps)

        unclipped = ratio * adv
        clipped = tf.clip_by_value(
            ratio,
            1.0 - CLIP_EPS,
            1.0 + CLIP_EPS
        ) * adv

        actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
        actor_loss -= ENT_COEF * tf.reduce_mean(-tf.abs(a_new))

        critic_loss = tf.reduce_mean(tf.square(returns - v_new))

    grads_actor = tape_actor.gradient(actor_loss, actor.trainable_variables)
    grads_critic = tape_critic.gradient(critic_loss, critic.trainable_variables)

    opt_actor.apply_gradients(zip(grads_actor, actor.trainable_variables))
    opt_critic.apply_gradients(zip(grads_critic, critic.trainable_variables))


@st.cache_data(show_spinner=False)
def optimize_ppo_outflow(
    df_target,
    s0_init,
    total_storage_capacity,
    max_discharge,
    w_store,
    w_dev,
    w_excess,
    ppo_epochs_num
):
    set_seeds(42)

    qin = np.maximum(df_target["Inflow (m^3/s)"].to_numpy(float), 0.0)
    qbase = np.maximum(df_target["Total discharge (m^3/s)"].to_numpy(float), 0.0)
    stor = df_target["Water storage (million m^3)"].to_numpy(float)

    eps = 1e-6

    qin_mu, qin_sd = float(qin.mean()), float(qin.std() + eps)
    qbas_mu, qbas_sd = float(qbase.mean()), float(qbase.std() + eps)
    stor_mu, stor_sd = float(stor.mean()), float(stor.std() + eps)

    def make_state(storage, inflow, q_baseline):
        return np.array([
            (storage - stor_mu) / stor_sd,
            (inflow - qin_mu) / qin_sd,
            (q_baseline - qbas_mu) / qbas_sd
        ], dtype=np.float32)

    actor, critic, opt_actor, opt_critic = build_models()
    train_idx = np.arange(0, len(qin), TRAIN_STRIDE)

    for _ in range(ppo_epochs_num):
        states, rewards, values, logps = [], [], [], []
        storage = float(s0_init)

        for i in train_idx:
            state = make_state(storage, qin[i], qbase[i])
            state_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)

            action = actor(state_tf)[0, 0]
            value = critic(state_tf)[0, 0]

            action_value = float(action.numpy())
            q_release = clamp_outflow(qbase[i], action_value, max_discharge)

            storage = np.clip(
                storage + (qin[i] - q_release) * DELTA_T / 1e6,
                0.0,
                total_storage_capacity
            )

            reward = compute_reward(
                storage,
                qin[i],
                q_release,
                qbase[i],
                w_store,
                w_dev,
                w_excess
            )

            states.append(state)
            rewards.append(reward)
            values.append(float(value.numpy()))
            logps.append(-0.5 * action_value ** 2)

        adv, returns = gae_advantages(rewards, values)

        ppo_update(
            actor,
            critic,
            opt_actor,
            opt_critic,
            np.array(states, dtype=np.float32),
            np.array(logps, dtype=np.float32),
            adv,
            returns
        )

    q_opt_out = []
    s_opt_out = []

    storage = float(s0_init)

    for idx in range(len(df_target)):
        state = make_state(storage, qin[idx], qbase[idx])
        state_tf = tf.convert_to_tensor(state[None, :], dtype=tf.float32)

        action = actor(state_tf)[0, 0]
        q_release = clamp_outflow(qbase[idx], float(action.numpy()), max_discharge)

        storage = float(np.clip(
            storage + (qin[idx] - q_release) * DELTA_T / 1e6,
            0.0,
            total_storage_capacity
        ))

        q_opt_out.append(q_release)
        s_opt_out.append(storage)

    return q_opt_out, s_opt_out


# ============================================================
# Violation table
# ============================================================
def build_violation_table(df, storage_col, limit, label):
    d = df[["date", storage_col]].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["year"] = d["date"].dt.year
    d[storage_col] = pd.to_numeric(d[storage_col], errors="coerce")
    d = d.dropna(subset=[storage_col])

    d["exceed"] = d[storage_col] - float(limit)
    d = d[d["exceed"] > 0.0].copy()

    if d.empty:
        return pd.DataFrame(
            columns=[
                "Violation",
                "Year",
                "Season",
                "Start_date",
                "End_date",
                "Days",
                "Max_exceed_Mm3",
                "Sum_exceed_Mm3_days"
            ]
        )

    d["month"] = d["date"].dt.month
    d["Season"] = d["month"].apply(season_from_month)
    d = d.sort_values(["year", "Season", "date"]).reset_index(drop=True)

    out_rows = []

    for (yr, seas), g in d.groupby(["year", "Season"], sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        gap = g["date"].diff().dt.days.fillna(1).astype(int)
        run_id = (gap != 1).cumsum()

        for _, r in g.groupby(run_id):
            out_rows.append({
                "Violation": label,
                "Year": int(yr),
                "Season": seas,
                "Start_date": str(r["date"].iloc[0].date()),
                "End_date": str(r["date"].iloc[-1].date()),
                "Days": int(len(r)),
                "Max_exceed_Mm3": round(float(r["exceed"].max()), 3),
                "Sum_exceed_Mm3_days": round(float(r["exceed"].sum()), 3),
            })

    return (
        pd.DataFrame(out_rows)
        .sort_values(["Year", "Season", "Start_date"])
        .reset_index(drop=True)
    )


# ============================================================
# Data loading
# ============================================================
required_historical_cols = [
    "date",
    "remote_sensed_watershed_precipitation (mm)",
    "temperature (°C)",
    "Water storage (million m^3)",
    "Inflow (m^3/s)",
    "Total discharge (m^3/s)"
]

required_future_cols = [
    "date",
    "remote_sensed_watershed_precipitation (mm)",
    "temperature (°C)"
]

if input_method == "📂 Upload Raw Excel Files":
    file_hist = st.sidebar.file_uploader(
        "Upload Observed Reservoir Data Sheet (.xlsx)",
        type=["xlsx"]
    )

    file_fut_in = st.sidebar.file_uploader(
        "Upload Future Weather Inputs Sheet (.xlsx)",
        type=["xlsx"]
    )

    try:
        df_historical_source = load_excel_source(
            file_hist if file_hist is not None else BASE_HISTORICAL_PATH
        )

        df_future_inputs_source = load_excel_source(
            file_fut_in if file_fut_in is not None else BASE_FUTURE_INPUTS_PATH
        )

    except Exception as e:
        st.error(
            "Could not load the Excel files. Upload both files manually, "
            "or check that the default paths exist on this computer."
        )
        st.exception(e)
        st.stop()

else:
    try:
        base_hist = load_excel_source(BASE_HISTORICAL_PATH)
        base_future = load_excel_source(BASE_FUTURE_INPUTS_PATH)

    except Exception as e:
        st.error(
            "Manual editing mode needs the default Excel files to load first. "
            "Please check the file paths or use upload mode."
        )
        st.exception(e)
        st.stop()

    st.info("💡 Select a cell below and press Ctrl+V to paste tables copied from Excel.")

    df_historical_source = normalize_df(
        st.data_editor(
            base_hist,
            num_rows="dynamic",
            height=220,
            key="historical_editor"
        )
    )

    df_future_inputs_source = normalize_df(
        st.data_editor(
            base_future,
            num_rows="dynamic",
            height=220,
            key="future_editor"
        )
    )

validate_required_columns(
    df_historical_source,
    required_historical_cols,
    "Observed reservoir data"
)

validate_required_columns(
    df_future_inputs_source,
    required_future_cols,
    "Future weather input data"
)

df_historical_source["date"] = pd.to_datetime(
    df_historical_source["date"],
    errors="coerce"
).dt.normalize()

df_future_inputs_source["date"] = pd.to_datetime(
    df_future_inputs_source["date"],
    errors="coerce"
).dt.normalize()

df_historical_source = df_historical_source.dropna(subset=["date"]).copy()
df_future_inputs_source = df_future_inputs_source.dropna(subset=["date"]).copy()

df_historical_source = clean_numeric_columns(
    df_historical_source,
    [
        "remote_sensed_watershed_precipitation (mm)",
        "temperature (°C)",
        "Water storage (million m^3)",
        "Inflow (m^3/s)",
        "Total discharge (m^3/s)"
    ]
)

df_future_inputs_source = clean_numeric_columns(
    df_future_inputs_source,
    [
        "remote_sensed_watershed_precipitation (mm)",
        "temperature (°C)"
    ]
)


# ============================================================
# Computation pipeline
# ============================================================
with st.status("🚀 Computing pipeline calculations...", expanded=True) as status:
    status.write(
    f"Training CNN-LSTM ({epochs} epochs) and generating future reservoir variables..."
    )

    df_fut = run_cnn_lstm_forecast(
        df_historical_source,
        df_future_inputs_source,
        epochs_num=epochs
    )

    df_fut["date"] = pd.to_datetime(
        df_fut["date"],
        errors="coerce"
    ).dt.normalize()

    df_fut = df_fut.dropna(subset=["date"]).copy()

    required_pred_cols = [
        "Water storage (million m^3)",
        "Inflow (m^3/s)",
        "Total discharge (m^3/s)"
    ]

    df_fut = clean_numeric_columns(df_fut, required_pred_cols)

    df_pred = df_fut[
        df_fut["date"] >= pd.Timestamp("2026-01-01")
    ].copy().sort_values("date").reset_index(drop=True)

    if df_pred.empty:
        st.error("No future data found from 2026-01-01 onward.")
        st.stop()

    last_hist = df_historical_source[
        df_historical_source["date"] < pd.Timestamp("2026-01-01")
    ].copy().sort_values("date")

    if last_hist.empty:
        st.error("No observed record before 2026-01-01 was found for initial storage.")
        st.stop()

    s0 = float(last_hist["Water storage (million m^3)"].iloc[-1])

    status.write(
    f"Running PPO reservoir operation optimization ({ppo_epochs} epochs)..."
    )

    q_opt, s_opt = optimize_ppo_outflow(
        df_pred,
        s0,
        total_storage_capacity,
        max_discharge,
        w_store_slider,
        w_dev_slider,
        w_excess_slider,
        ppo_epochs
    )

    df_pred["PPO_Inflow"] = df_pred["Inflow (m^3/s)"]
    df_pred["PPO_Optimized_Outflow"] = q_opt
    df_pred["PPO_Optimized_Storage"] = s_opt
    df_pred["year"] = df_pred["date"].dt.year

    storage_panel_col = pick_col(
        df_pred,
        [
            "Water storage (million m^3) - PPO",
            "Water storage (million m³) - PPO",
            "Storage (million m^3) - PPO",
            "Storage_PPO (million m^3)",
            "Storage_PPO (Mm3)",
            "PPO Storage (million m^3)",
            "Storage (million m^3) - PPO (computed)",
            "Water storage (million m^3)",
            "Water storage (million m³)",
            "Storage (million m^3)",
            "Storage",
            "storage",
        ],
        required=True,
        name_hint="storage for storage panels"
    )

    df_pred[storage_panel_col] = pd.to_numeric(
        df_pred[storage_panel_col],
        errors="coerce"
    ).ffill().bfill()

    status.update(
        label="✅ Complete predictive system pipeline successfully compiled.",
        state="complete"
    )

st.info(f"Storage panels are using: `{storage_panel_col}`")


# ============================================================
# Hydrograph controls and panels
# ============================================================
st.markdown("---")
st.subheader("🌊 Hydrograph Display Controls")

hydro_col1, hydro_col2 = st.columns(2)

with hydro_col1:
    hydro_group = st.selectbox(
        "Select Hydrograph Group Target",
        ["Annual", "Winter", "Spring", "Summer", "Fall"],
        key="hydro_group"
    )

with hydro_col2:
    hydro_decade = st.selectbox(
        "Select Hydrograph Projection Decade Horizon",
        ["2026–2035", "2036–2045", "2046–2055"],
        key="hydro_decade"
    )

df_hydro, hydro_years, hydro_panel_label, _, _ = filter_by_group_and_decade(
    df_pred,
    hydro_group,
    hydro_decade
)

st.subheader("🌊 Inflow and PPO Optimized Outflow Hydrograph Panels")

if df_hydro.empty:
    st.warning("No hydrograph data found for the current filter combination.")
else:
    years = sorted(df_hydro["date"].dt.year.unique())

    fig_h, axes_h = plt.subplots(5, 2, figsize=(16, 22), sharex=False)
    axes_flat = axes_h.flatten()

    global_ymax = max(
        df_pred["PPO_Inflow"].max(),
        df_pred["PPO_Optimized_Outflow"].max()
    ) * 1.1

    for idx, yr in enumerate(years[:10]):
        ax = axes_flat[idx]

        df_year = df_hydro[
            df_hydro["date"].dt.year == yr
        ].copy().sort_values("date").reset_index(drop=True)

        if df_year.empty:
            continue

        dates = df_year["date"]
        y_in = df_year["PPO_Inflow"]
        y_out = df_year["PPO_Optimized_Outflow"]

        ax.plot(dates, y_in, color="#0072B2", linewidth=1.3, label="Generated Inflow (m³/s)")
        ax.plot(dates, y_out, color="#E69F00", linewidth=1.3, label="PPO Optimized Outflow (m³/s)")

        ax.fill_between(
            dates.to_numpy(),
            y_in.to_numpy(),
            y_out.to_numpy(),
            where=(y_in.to_numpy() >= y_out.to_numpy()),
            color="#0072B2",
            alpha=0.22,
            interpolate=True,
            label="Peak Attenuation Window"
        )

        ax.fill_between(
            dates.to_numpy(),
            y_out.to_numpy(),
            y_in.to_numpy(),
            where=(y_out.to_numpy() > y_in.to_numpy()),
            color="#E69F00",
            alpha=0.16,
            interpolate=True,
            label="Excess Flood Release Window"
        )

        ax.set_title(f"Year: {yr}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, global_ymax)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        if hydro_group == "Annual":
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))

        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.set_ylabel("Discharge (m³/s)")

    for i in range(len(years[:10]), 10):
        fig_h.delaxes(axes_flat[i])

    handles, labels = axes_flat[0].get_legend_handles_labels()

    fig_h.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=12,
        frameon=True
    )

    fig_h.tight_layout(rect=[0, 0, 1, 0.98])
    st.pyplot(fig_h)


# ============================================================
# Storage controls and panels
# ============================================================
st.markdown("---")
st.subheader("💧 Storage Display Controls")

storage_col1, storage_col2 = st.columns(2)

with storage_col1:
    storage_group = st.selectbox(
        "Select Storage Group Target",
        ["Annual", "Winter", "Spring", "Summer", "Fall"],
        key="storage_group"
    )

with storage_col2:
    storage_decade = st.selectbox(
        "Select Storage Projection Decade Horizon",
        ["2026–2035", "2036–2045", "2046–2055"],
        key="storage_decade"
    )

df_storage, storage_years, storage_panel_label, _, _ = filter_by_group_and_decade(
    df_pred,
    storage_group,
    storage_decade
)

st.subheader("💧 PPO Storage Dynamics Panels")


def get_storage_ylim(df, storage_column):
    smax = np.nanmax(df[storage_column].to_numpy(dtype=float))
    smin = np.nanmin(df[storage_column].to_numpy(dtype=float))

    if not np.isfinite(smax):
        smax = total_storage_capacity

    if not np.isfinite(smin):
        smin = 0.0

    y0 = max(0.0, math.floor(smin / 100.0) * 100.0)
    y1 = math.ceil(smax / 100.0) * 100.0

    if y1 <= y0:
        y0, y1 = 0.0, max(500.0, smax)

    return y0, y1


def plot_storage_grid_streamlit(df, storage_column, years_subset, panel_label, group_name):
    y0, y1 = get_storage_ylim(df, storage_column)

    fig_s, axes_s = plt.subplots(
        5,
        2,
        figsize=(14, 18),
        sharey=True
    )

    axes_s = np.atleast_1d(axes_s).ravel()

    for i, yr in enumerate(years_subset[:10]):
        ax = axes_s[i]
        sub = df[df["date"].dt.year == yr].copy()

        ax.set_title(
            str(yr),
            fontsize=18,
            fontweight="bold",
            pad=12
        )

        ax.axhline(
            y=FLOOD_CONTROL_LIMIT,
            color="#FF8C00",
            linestyle=":",
            linewidth=1.8,
            zorder=1
        )

        ax.plot(
            sub["date"],
            sub[storage_column],
            color="tab:purple",
            linewidth=2.2,
            alpha=0.95,
            zorder=2
        )

        ax.set_ylim(y0, y1)
        ax.grid(True, color="#eeeeee")

        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=13)

        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")

        if i % 2 == 0:
            ax.set_ylabel(
                "Storage (Mm³)",
                fontsize=15,
                fontweight="bold"
            )

        if len(sub) > 0:
            d0 = sub["date"].min()
            d1 = sub["date"].max()
            mid = d0 + (d1 - d0) / 2

            ax.set_xticks([d0, mid, d1])
            ax.set_xticklabels(
                [
                    d0.strftime("%Y-%m-%d"),
                    mid.strftime("%Y-%m-%d"),
                    d1.strftime("%Y-%m-%d")
                ],
                rotation=0,
                fontsize=11,
                fontweight="bold"
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold"
            )

    for j in range(len(years_subset[:10]), len(axes_s)):
        axes_s[j].axis("off")

    fig_s.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#FF8C00",
                lw=1.8,
                ls=":",
                label=f"Flood-control line ({FLOOD_CONTROL_LIMIT:.0f} Mm³)"
            )
        ],
        loc="lower center",
        ncol=1,
        fontsize=16,
        frameon=False
    )

    fig_s.suptitle(
        f"({panel_label}) {group_name} PPO Storage Dynamics: "
        f"{years_subset[0]}–{years_subset[-1]}",
        fontsize=22,
        fontweight="bold",
        y=0.992
    )

    fig_s.tight_layout(rect=[0, 0.055, 1, 0.965])
    st.pyplot(fig_s)


if df_storage.empty:
    st.warning("No storage data found for the current filter combination.")
else:
    plot_storage_grid_streamlit(
        df_storage,
        storage_panel_col,
        storage_years,
        storage_panel_label,
        storage_group
    )


# ============================================================
# Violation report
# ============================================================
st.markdown("---")
st.subheader("⚠️ Storage Violation Report")

viol_fc = build_violation_table(
    df_pred,
    storage_panel_col,
    FLOOD_CONTROL_LIMIT,
    f"Flood-control limit > {FLOOD_CONTROL_LIMIT:.0f} Mm³"
)

viol_cap = build_violation_table(
    df_pred,
    storage_panel_col,
    total_storage_capacity,
    f"Total capacity > {total_storage_capacity:.0f} Mm³"
)

non_empty_violations = [
    df for df in [viol_fc, viol_cap]
    if df is not None and not df.empty
]

if len(non_empty_violations) == 0:
    viol_all = pd.DataFrame()
    st.success("No violations found for flood-control limit or total capacity.")
else:
    viol_all = pd.concat(non_empty_violations, ignore_index=True)
    st.dataframe(viol_all, use_container_width=True)

# ============================================================
# 4-panel animation with selectable date window
# ============================================================
st.markdown("---")
st.subheader("🎞️ 4-Panel Reservoir Operation Animation")

anim_col1, anim_col2, anim_col3 = st.columns(3)

min_anim_date = df_pred["date"].min().date()
max_anim_date = df_pred["date"].max().date()

with anim_col1:
    anim_start_date = st.date_input(
        "Animation Start Date",
        value=pd.Timestamp("2026-07-14").date(),
        min_value=min_anim_date,
        max_value=max_anim_date,
        key="anim_start_date"
    )

with anim_col2:
    anim_days = st.slider(
        "Animation Length (days)",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        key="anim_days"
    )

with anim_col3:
    show_animation = st.checkbox(
        "Show animation",
        value=True,
        key="show_animation"
    )

if show_animation:
    anim_start = pd.Timestamp(anim_start_date)
    anim_end = anim_start + pd.Timedelta(days=anim_days - 1)

    sub_anim = df_pred[
        (df_pred["date"] >= anim_start) &
        (df_pred["date"] <= anim_end)
    ].copy().sort_values("date").reset_index(drop=True)

    if len(sub_anim) < 2:
        st.warning("Animation range is too small or has insufficient data.")
    else:
        st.info(f"Animation window: {anim_start.date()} → {anim_end.date()}")

        dates = pd.DatetimeIndex(sub_anim["date"])

        inflow_anim = roll_mean(
            sub_anim["PPO_Inflow"].to_numpy(float),
            SMOOTH_WIN
        ).astype(np.float32)

        outflow_anim = roll_mean(
            sub_anim["PPO_Optimized_Outflow"].to_numpy(float),
            SMOOTH_WIN
        ).astype(np.float32)

        stor_anim_Mm3 = sub_anim["PPO_Optimized_Storage"].to_numpy(float).astype(np.float32)

        V_anim_m3 = (stor_anim_Mm3 * 1e6).astype(np.float32)
        V_anim_Mm3 = stor_anim_Mm3.astype(np.float32)

        nd = len(dates)

        Smax_m3 = 2_900e6
        hmax_m = 200.0
        Nshape = 0.29
        ln2 = np.log(2.0)

        def S_of_h(h):
            h = np.asarray(h, float)
            h = np.clip(h, 0, hmax_m)
            return Smax_m3 * np.power(
                np.exp(ln2 * (h / hmax_m)) - 1.0,
                1.0 / Nshape
            )

        def h_of_S(S):
            S = np.asarray(S, float)
            S = np.clip(S, 0, Smax_m3)
            return hmax_m * np.log1p((S / Smax_m3) ** Nshape) / ln2

        H_anim_m = h_of_S(V_anim_m3).astype(np.float32)

        S_fc_m3 = float(FLOOD_CONTROL_LIMIT) * 1e6
        H_fc_m = float(h_of_S(S_fc_m3))

        fig_anim, axs = plt.subplots(
            2,
            2,
            figsize=(10.5, 6.5),
            constrained_layout=True
        )

        (ax_cd, ax_res), (ax_acc, ax_hg) = axs

        cap_h = np.linspace(0, hmax_m, 240)
        cap_S_Mm3 = S_of_h(cap_h) / 1e6

        ax_cd.plot(
            cap_S_Mm3,
            cap_h,
            lw=2.5,
            label="Capacity–Depth Curve"
        )

        dot_cd, = ax_cd.plot(
            [V_anim_Mm3[0]],
            [H_anim_m[0]],
            "o",
            ms=8,
            label="Water Depth"
        )

        ax_cd.set_title("(a) Capacity–Depth Relationship", fontsize=13, fontweight="bold")
        ax_cd.set_xlabel("Reservoir Capacity (Mm³)")
        ax_cd.set_ylabel("Reservoir Depth (m)")
        ax_cd.legend(loc="lower right")

        L_valley = 65.8
        dam_height = hmax_m

        x_bank = np.array([0.0, L_valley])
        z_bank = np.array([0.0, 1.5])
        bed_slope = (z_bank[1] - z_bank[0]) / (x_bank[1] - x_bank[0])

        crest_z, crest_w, base_w = dam_height, 4.0, 22.0
        up_x = 0.0

        x_center = -base_w / 2.0
        top_left = x_center - crest_w / 2.0
        top_right = x_center + crest_w / 2.0

        def water_polygon_xy(h):
            h = float(np.clip(h, 0.0, dam_height))
            x_top = min(L_valley, h / bed_slope) if bed_slope > 0 else L_valley
            x_bed = np.linspace(0.0, x_top, 90)
            z_bed = z_bank[0] + bed_slope * x_bed

            x_on_dam = (
                up_x + (top_right - up_x) * (h / crest_z)
                if crest_z > 0 else up_x
            )

            wx = np.concatenate([x_bed, [x_top, x_on_dam]])
            wy = np.concatenate([z_bed, [h, h]])

            return wx, wy

        wx0, wy0 = water_polygon_xy(H_anim_m[0])
        water_poly = ax_res.fill(wx0, wy0, alpha=0.55, zorder=1)[0]

        ax_res.plot(x_bank, z_bank, lw=2.2, zorder=3)

        dam_poly_x = np.array([-base_w, up_x, top_right, top_left], dtype=float)
        dam_poly_y = np.array([0.0, 0.0, crest_z, crest_z], dtype=float)

        ax_res.fill(
            dam_poly_x,
            dam_poly_y,
            color="saddlebrown",
            zorder=4
        )

        date_text = ax_res.text(
            0.98,
            0.99,
            f"Day: {dates[0].date()}",
            transform=ax_res.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="yellow", ec="black"),
            zorder=7
        )

        ax_res.set_title("(b) Reservoir Water Level and Dam Profile", fontsize=13, fontweight="bold")
        ax_res.set_xlabel("Horizontal Distance (m)")
        ax_res.set_ylabel("Height (m)")
        ax_res.set_xlim(-base_w - 3, L_valley + 5)
        ax_res.set_ylim(0, dam_height + 7)

        ax_res.axhline(
            y=H_fc_m,
            linestyle=":",
            linewidth=2.0,
            color="#FF8C00",
            zorder=2
        )

        ax_res.text(
            1.2,
            H_fc_m,
            f"Flood control level (~{H_fc_m:.1f} m)",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=6
        )

        acc_line, = ax_acc.plot(
            dates,
            V_anim_Mm3,
            lw=2.5,
            label="Water Volume up to t"
        )

        acc_dot, = ax_acc.plot(
            [dates[0]],
            [V_anim_Mm3[0]],
            "o",
            ms=8,
            label="Volume at time t"
        )

        ax_acc.set_title("(c) Accumulated Reservoir Storage", fontsize=13, fontweight="bold")
        ax_acc.set_ylabel("Water Volume (Mm³)")
        ax_acc.set_xlabel("Time (days)")
        ax_acc.legend(loc="lower right")

        hg_in_line, = ax_hg.plot(
            dates,
            inflow_anim,
            lw=2.5,
            label="Inflow (m³/s)"
        )

        hg_out_line, = ax_hg.plot(
            dates,
            outflow_anim,
            lw=2.5,
            label="Outflow (m³/s)"
        )

        ax_hg.fill_between(
            dates,
            outflow_anim,
            inflow_anim,
            where=(inflow_anim > outflow_anim),
            alpha=0.35,
            interpolate=True,
            label="Water stored"
        )

        ax_hg.fill_between(
            dates,
            inflow_anim,
            outflow_anim,
            where=(outflow_anim > inflow_anim),
            alpha=0.30,
            interpolate=True,
            label="Release"
        )

        hg_in_mark = ax_hg.plot(
            [dates[0]],
            [inflow_anim[0]],
            marker="D",
            ms=7,
            label="Inflow at t"
        )[0]

        hg_out_mark = ax_hg.plot(
            [dates[0]],
            [outflow_anim[0]],
            marker="D",
            ms=7,
            label="Outflow at t"
        )[0]

        ax_hg.set_title("(d) Hydrographs and Attenuation", fontsize=13, fontweight="bold")
        ax_hg.set_ylabel("Water Discharge (m³/s)")
        ax_hg.set_xlabel("Time (days)")
        ax_hg.legend(loc="upper right")

        if len(inflow_anim) > 1 and len(outflow_anim) > 1:
            peak_in_idx = int(np.nanargmax(inflow_anim))
            peak_out_idx = int(np.nanargmax(outflow_anim))

            peak_in_date = dates[peak_in_idx]
            peak_out_date = dates[peak_out_idx]
            peak_in_val = float(inflow_anim[peak_in_idx])
            peak_out_val = float(outflow_anim[peak_out_idx])

            shift_days = 2
            left_idx = max(0, min(peak_in_idx, peak_out_idx) - shift_days)
            left_date = dates[left_idx]

            ax_hg.plot(
                [left_date, peak_in_date],
                [peak_in_val, peak_in_val],
                ls=":",
                lw=2,
                color="red"
            )

            ax_hg.plot(
                [left_date, peak_out_date],
                [peak_out_val, peak_out_val],
                ls=":",
                lw=2,
                color="red"
            )

            y0_arrow, y1_arrow = sorted([peak_out_val, peak_in_val])

            ax_hg.annotate(
                "",
                xy=(left_date, y0_arrow),
                xytext=(left_date, y1_arrow),
                arrowprops=dict(arrowstyle="<->", color="red", lw=2)
            )

            ax_hg.text(
                dates[max(0, left_idx - 1)],
                (y0_arrow + y1_arrow) / 2,
                "Flood peak\nattenuation",
                color="red",
                fontsize=10,
                ha="right",
                va="center"
            )

        locator = AutoDateLocator(minticks=3, maxticks=5)

        for axx in (ax_acc, ax_hg):
            axx.xaxis.set_major_locator(locator)
            axx.xaxis.set_major_formatter(ConciseDateFormatter(locator))

        def init():
            dot_cd.set_data([V_anim_Mm3[0]], [H_anim_m[0]])
            acc_dot.set_data([dates[0]], [V_anim_Mm3[0]])
            hg_in_mark.set_data([dates[0]], [inflow_anim[0]])
            hg_out_mark.set_data([dates[0]], [outflow_anim[0]])

            wx, wy = water_polygon_xy(H_anim_m[0])
            water_poly.set_xy(np.c_[wx, wy])
            date_text.set_text(f"Day: {dates[0].date()}")

            return (
                dot_cd,
                acc_dot,
                hg_in_mark,
                hg_out_mark,
                water_poly,
                date_text
            )

        def update(i):
            dot_cd.set_data([V_anim_Mm3[i]], [H_anim_m[i]])
            acc_dot.set_data([dates[i]], [V_anim_Mm3[i]])
            hg_in_mark.set_data([dates[i]], [inflow_anim[i]])
            hg_out_mark.set_data([dates[i]], [outflow_anim[i]])

            wx, wy = water_polygon_xy(H_anim_m[i])
            water_poly.set_xy(np.c_[wx, wy])
            date_text.set_text(f"Day: {dates[i].date()}")

            return (
                dot_cd,
                acc_dot,
                hg_in_mark,
                hg_out_mark,
                water_poly,
                date_text
            )

        anim = FuncAnimation(
            fig_anim,
            update,
            frames=nd,
            init_func=init,
            blit=True,
            interval=int(1000 / FPS)
        )

        html_anim = anim.to_jshtml()
        components.html(html_anim, height=760, scrolling=True)

        plt.close(fig_anim)


# ============================================================
# Output preview and download
# ============================================================
st.markdown("---")
st.subheader("📋 Output Matrix Preview Database")

export_df = df_pred[
    [
        "date",
        "PPO_Inflow",
        "PPO_Optimized_Outflow",
        "PPO_Optimized_Storage",
        storage_panel_col
    ]
].copy()

st.dataframe(export_df.head(100), use_container_width=True)

st.download_button(
    label="📥 Download Complete Predictions (.CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="optimized_seasonal_hydrographs.csv",
    mime="text/csv"
)
