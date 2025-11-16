import tensorflow as tf
import IPython
from utils.plotting import plot_train_validation, TrainingPlot
from utils.evaluation import evaluate_predictions
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Utility functions
# Special Characters for Output Formating
StartBold = "\033[1m"
EndBold = "\033[0m"

def compile_and_fit(model, window, val_df, preprocessor, patience=5, max_epochs=100,
                     log_dict=None):

    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            min_delta=0.001, patience=patience, mode='min', verbose=1)

    # mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)

    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    history = model.fit(window.train, epochs=max_epochs, validation_data=window.val, callbacks=[
                         early_stopping, rlr])


    if log_dict:
        IPython.display.clear_output()

        print(f'\n\n{StartBold}Training vs. Validation:{EndBold}\n')
        plot_train_validation(history, window.val)

        log_dict['multi_val_performance'] = model.evaluate(
            window.val, verbose=0)
        log_dict['multi_performance'] = model.evaluate(
            window.train, verbose=0)

    predictions = model.predict(window.val)
    evaluate_predictions(val_df.traffic_volume, predictions, preprocessor)
    return history
