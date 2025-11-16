import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Utility functions
# Special Characters for Output Formating
StartBold = "\033[1m"
EndBold = "\033[0m"

def evaluate_predictions(y_true, y_pred, preprocessor, plot_start_index=-500):
    print(f'\n\n{StartBold}Predictions Evaluation:{EndBold}\n')
    y_p = np.asarray(preprocessor.scaler_tv.inverse_transform(
        y_pred.reshape(-1, 1))).ravel()
    n_predictions = len(y_p)
    y = np.asarray(y_true[-n_predictions:]).ravel()
    print('Predictions:', n_predictions)
    # MAE
    mae_fn = tf.keras.losses.MeanAbsoluteError()
    mae = float(mae_fn(y, y_p))
    #mae_scaled = float(mae_fn(y_scaled, y_p_scaled))
    mae_scaled = float(preprocessor.scaler_tv.transform(np.array([[mae]])))
    print(f'MAE: {mae:.2f} ({mae_scaled:.4f})')
    # RMSE
    rmse_fn = tf.keras.metrics.RootMeanSquaredError()
    rmse = float(rmse_fn(y, y_p))
    #rmse_scaled = float(rmse_fn(y_scaled, y_p_scaled))
    rmse_scaled = float(preprocessor.scaler_tv.transform(np.array([[rmse]])))
    print(f'RMSE: {rmse:.2f} ({rmse_scaled:.4f})')

    plt.subplots(figsize=(15, 2))
    plt.plot(y[plot_start_index:], marker='.', label='true')
    plt.plot(y_p[plot_start_index:], marker='.', label='predicted')
    plt.legend()
    plt.show()
