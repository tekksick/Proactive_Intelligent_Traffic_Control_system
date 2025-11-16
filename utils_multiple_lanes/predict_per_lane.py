import numpy as np
import pandas as pd
from joblib import load
from utils.preprocessing import TrafficPreprocessor

INPUT_WIDTH = 6   # number of past time steps
OUT_STEPS = 1     # number of steps to predict ahead

# Load your pre-fitted scalers once (at program start)
scaler_features = load('scalers/scaler.joblib')  # input features
scaler_tv = load('scalers/scaler_tv.joblib')     # target variable


def predict_per_lane(csv_path, model):
    """
    Predict next OUT_STEPS traffic volumes for a single lane using the last INPUT_WIDTH samples.
    Prints both predicted and true (unscaled) traffic volume.
    Returns a float for OUT_STEPS=1.
    """
    # ------------------------------
    # 1. Load lane CSV
    # ------------------------------
    df_raw = pd.read_csv(
        csv_path,
        parse_dates=['timestamp'],
        index_col='timestamp'
    )
    lane = df_raw['lane_no'].iloc[0]
    # Drop lane_no if present
    if 'lane_no' in df_raw.columns:
        df_raw = df_raw.drop(columns=['lane_no'])

    # ------------------------------
    # 2. Add time features
    # ------------------------------
    df = TrafficPreprocessor.add_time_features(df_raw)

    # ------------------------------
    # 3. Take last INPUT_WIDTH rows
    # ------------------------------
    last_window = df.tail(INPUT_WIDTH)

    # ------------------------------
    # 4. Scale input features using pre-fitted scaler
    # ------------------------------
    scaled_window = scaler_features.transform(last_window)

    # Reshape for model: (batch_size=1, INPUT_WIDTH, num_features)
    model_input = scaled_window.reshape(1, INPUT_WIDTH, -1)

    # ------------------------------
    # 5. Predict scaled traffic volume
    # ------------------------------
    scaled_pred = model.predict(model_input, verbose=0)

    # Ensure shape (OUT_STEPS, 1)
    scaled_pred = np.asarray(scaled_pred).reshape(OUT_STEPS, 1)

    # ------------------------------
    # 6. Inverse transform to real traffic volume
    # ------------------------------
    pred_unscaled = scaler_tv.inverse_transform(scaled_pred).ravel()

    # ------------------------------
    # 7. Get true value (last traffic_volume)
    # ------------------------------
    # last step(s)
    true_value = df_raw['traffic_volume'].values[-OUT_STEPS:]

    # ------------------------------
    # 8. Print predicted vs true
    # ------------------------------
    print(
        f"lane: {lane} Predicted: {pred_unscaled[0]:.2f}, True: {true_value[0]:.2f}")

    # ------------------------------
    # 9. Return predicted value as float
    # ------------------------------
    return float(pred_unscaled[0])
