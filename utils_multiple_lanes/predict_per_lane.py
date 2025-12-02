import numpy as np
import pandas as pd
from joblib import load
from utils.preprocessing import TrafficPreprocessor

INPUT_WIDTH = 6
OUT_STEPS = 1

scaler_features = load('scalers/scaler.joblib')
scaler_tv = load('scalers/scaler_tv.joblib')


def predict_per_lane(csv_path, model, N_PREDICTIONS=8):
    df_raw = pd.read_csv(
        csv_path,
        parse_dates=['timestamp'],
        index_col='timestamp'
    )
    lane = df_raw['lane_no'].iloc[0]
    if 'lane_no' in df_raw.columns:
        df_raw = df_raw.drop(columns=['lane_no'])

    df = TrafficPreprocessor.add_time_features(df_raw)

    current_window_features = df.tail(INPUT_WIDTH).copy()

    unscaled_predictions = np.zeros(N_PREDICTIONS)
    true_values = df_raw['traffic_volume'].values[-N_PREDICTIONS:]

    for i in range(N_PREDICTIONS):
        scaled_input_features = scaler_features.transform(
            current_window_features)
        model_input = scaled_input_features.reshape(1, INPUT_WIDTH, -1)
        scaled_tv_pred = model.predict(model_input, verbose=0)
        pred_unscaled = scaler_tv.inverse_transform(scaled_tv_pred).ravel()[0]
        unscaled_predictions[i] = pred_unscaled

        last_time = current_window_features.index[-1]
        time_delta = current_window_features.index[1] - \
            current_window_features.index[0]
        next_time = last_time + time_delta

        next_step_df = pd.DataFrame(
            {'traffic_volume': [pred_unscaled]},
            index=[next_time]
        )
        next_step_features = TrafficPreprocessor.add_time_features(
            next_step_df).iloc[0]

        new_row_data = current_window_features.iloc[-1].copy()
        new_row_data.name = next_time

        new_row_data['traffic_volume'] = pred_unscaled

        for col in new_row_data.index:
            if col in next_step_features.index and col != 'traffic_volume':
                new_row_data[col] = next_step_features[col]

        current_window_features = current_window_features.iloc[1:].copy()
        current_window_features.loc[next_time] = new_row_data.values

    # ------------------------------
    # Print Predicted and True Values
    # ------------------------------
    # Format the predicted values to two decimal places
    predicted_str = " ".join([f"{v:.2f}" for v in unscaled_predictions])

    # Format the true values (only prints what's available up to N_PREDICTIONS)
    true_str = " ".join([f"{v:.2f}" for v in true_values])

    '''print(f"Lane: {lane}")
    print(f"Predicted Np Steps: {predicted_str}")
    print(f"True Np Steps:      {true_str}")'''

    return unscaled_predictions
