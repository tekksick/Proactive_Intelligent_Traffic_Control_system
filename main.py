import time
from concurrent.futures import ThreadPoolExecutor
from utils_multiple_lanes.predict_per_lane import predict_per_lane
from utils_multiple_lanes.webster import compute_webster_all
import tensorflow as tf

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "prediction_models_single_lane/MyLSTM_2.keras"
LANE_PATHS = [
    f"data/2x2_junction_dataset/junction_{j}_road_{r}.csv"
    for j in range(4)
    for r in range(4)
]
DELTA_T_SEC = 5  # update interval of 5 seconds
#DELTA_T_MIN = 5  # update interval in minutes
INPUT_WIDTH = 6  # must match your model


def main():
    # ------------------------------
    # LOAD MODEL
    # ------------------------------
    model = tf.keras.models.load_model(MODEL_PATH)

    # ------------------------------
    # REAL-TIME LOOP
    # ------------------------------
    while True:
        # 1. Parallel prediction for all lanes
        with ThreadPoolExecutor(max_workers=len(LANE_PATHS)) as executor:
            predictions = list(executor.map(
                lambda p: predict_per_lane(p, model), LANE_PATHS))

        # 2. Compute Webster timings
        timings = compute_webster_all(predictions)

        print("\nWebster Timings for All Junctions:")
        for j, t in timings.items():
            print(f"Junction {j}: {t}")

        # 3. Wait for next cycle
        print(
            f"\nWaiting for {DELTA_T_SEC} seconds until next prediction...\n")
        time.sleep(DELTA_T_SEC)


# ------------------------------
# RUN SCRIPT
# ------------------------------
if __name__ == "__main__":
    main()
