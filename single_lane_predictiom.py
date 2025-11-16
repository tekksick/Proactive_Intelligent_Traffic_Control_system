import pandas as pd
import tensorflow as tf
from utils.evaluation import evaluate_predictions
from utils.preprocessing import TrafficPreprocessor
from utils.windowing import WindowGenerator
from utils.prediction import TrafficPredictor
from utils.training import compile_and_fit
from joblib import dump



def describe(df):
    return pd.concat([df.describe().T, df.skew().rename('skew'),], axis=1)


def main():
    df_raw = pd.read_csv(r'data/2x2_junction_dataset/junction_0_road_0.csv',
                         parse_dates=['timestamp'], index_col='timestamp')
    df_raw = df_raw.drop(columns=['lane_no'])

    print("------------------------------------------------------------------")
    print(describe(df_raw))
    print("------------------------------------------------------------------")

    # add time features for raw datset
    df = TrafficPreprocessor.add_time_features(df_raw)
    total_rows = len(df)
    # Calculate split indices
    train_end = int(0.7 * total_rows)        # 70%
    val_end = train_end + int(0.2 * total_rows)  # 70% + 20% = 90%

    # Split the dataset sequentially
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Print shapes to verify
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("------------------------------------------------------------------")

    preprocessor = TrafficPreprocessor()
    scaled_train_df = preprocessor.fit_transform(train_df)
    scaled_val_df = preprocessor.transform(val_df)
    scaled_test_df = preprocessor.transform(test_df)
    #dump(preprocessor.scaler_tv, 'scaler_tv.joblib')
    #dump(preprocessor.scaler, 'scaler.joblib')



    INPUT_WIDTH = 6
    OUT_STEPS = 1
    SHIFT = 2
    window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=OUT_STEPS,
        shift=SHIFT,
        train_df=scaled_train_df,
        val_df=scaled_val_df,
        test_df=scaled_test_df,
        label_columns=['traffic_volume'],
        batch_size=32,
    )

    model = tf.keras.models.load_model(
        "prediction_models_single_lane/MyLSTM_2.keras")
    multi_val_performance = {}
    multi_performance = {}

    my_log = {
        'multi_val_performance': multi_val_performance,
        'multi_performance': multi_performance,
    }

    compile_and_fit(model, window, val_df,
                    preprocessor, log_dict=my_log)
    
    #model.save("MyLSTM_4.keras")


if __name__ == "__main__":
    main()
