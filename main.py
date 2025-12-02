import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tensorflow as tf
import json
import os
from utils_multiple_lanes.predict_per_lane import predict_per_lane
from mpc.grid_mpc import GridTrafficEnv, GridTrafficMpc, simulate_fixed_time_control
from mpc.grid_mpc import compute_mpc_control, predictions_to_disturbance_vector

def simulate_env_step(x_current, u_executed, d_in_first_step, env):
    """
    Approximates the environment's state transition using NumPy arrays.
    
    x_current: current queue lengths (np array)
    u_executed: the binary phase vector executed (np array)
    d_in_first_step: predicted input demand for the next step (np array)
    """
    T_hr = float(env.T) / 3600.0

    # B_nominal is a CasADi DM, must convert to NumPy for mtimes equivalent
    # Ensure B_nominal is accessible and is converted correctly
    try:
        B_nominal_np = np.array(env.B_nominal)
    except TypeError:
        # Fallback if env.B_nominal is already a numpy array or simpler structure
        B_nominal_np = env.B_nominal

    # q_out_rate = B_nominal * u
    # Use np.dot for matrix multiplication
    q_out_rate = np.dot(B_nominal_np, u_executed)

    # q_in_vol = d_in * T_hr
    q_in_vol = d_in_first_step * T_hr

    # q_out_vol = q_out_rate * T_hr
    q_out_vol = q_out_rate * T_hr

    x_next = x_current.flatten() + q_in_vol.flatten() - q_out_vol.flatten()

    return np.maximum(0, x_next)


def main():
    MODEL_PATH = "prediction_models_single_lane/MyLSTM_2.keras"
    JUNCTIONS = ["1_1", "1_2", "2_1", "2_2"]
    ROAD_INDICES = ["0", "1", "2", "3"]
    LANE_PATHS_TO_PREDICT = [f"data/2x2_junction_dataset/road_{i}_{k}.csv"
                             for i in JUNCTIONS for k in ROAD_INDICES]
    DELTA_T_SEC = 30
    OUTPUT_FILE = "signal_log.jsonl"

    try:
        # Check if the model file exists before trying to load it
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print(
            f"LSTM model not found at {MODEL_PATH}, using dummy predictions.")
        model = None

    env = GridTrafficEnv()
    mpc_controller = GridTrafficMpc(env)
    current_state_x = np.random.rand(env.N_STATE) * env.X_CAP / 4

    # CRITICAL: Initialize u_last_executed for the first MPC call
    u_last_executed = np.zeros((env.N_CONTROL, 1))
    u_last_executed[0, 0] = 1.0  # Default starting phase for Junction 1

    print("MPC controller initialized. Starting real-time loop...")
    print(f"Signal details will be logged to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'a') as f:
        while True:
            start_time = time.time()

            # --- Prediction ---
            if model:
                with ThreadPoolExecutor(max_workers=len(LANE_PATHS_TO_PREDICT)) as executor:
                    predictions = list(executor.map(
                        lambda p: predict_per_lane(p, model), LANE_PATHS_TO_PREDICT))
            else:
                predictions = [predict_per_lane(p, model)
                               for p in LANE_PATHS_TO_PREDICT]

            d_in_matrix = predictions_to_disturbance_vector(predictions, env)

            # --- MPC Control Calculation (Passing u_last_executed) ---
            optimal_phases_data = compute_mpc_control(
                current_state_x, d_in_matrix, mpc_controller, u_last_executed=u_last_executed)

            # --- Output and State Update ---
            if optimal_phases_data:

                # 1. Extract the chosen phase vector for simulation
                current_phases_dict = optimal_phases_data["current_phases"]
                u_executed_vector = np.zeros(env.N_CONTROL)
                # Ensure keys are sorted for consistent mapping (J1, J2, J3, J4 -> 0, 1, 2, 3)
                for j_idx, key in enumerate(sorted(current_phases_dict.keys())):
                    phase_index = current_phases_dict[key]
                    u_idx = j_idx * env.N_PHASES + phase_index
                    u_executed_vector[u_idx] = 1.0

                u_executed_vector = u_executed_vector.reshape(-1, 1)

                # 2. Log the output
                log_entry = {
                    "timestamp": time.time(),
                    "time_str": time.strftime('%H:%M:%S'),
                    "current_phases": current_phases_dict,
                    "predicted_timings": optimal_phases_data["predicted_timings"],
                    "state_before_step": current_state_x.tolist()
                }

                print(
                    f"Time: {log_entry['time_str']}, Phases: {log_entry['current_phases']}")

                f.write(json.dumps(log_entry) + '\n')
                f.flush()
                fixed_result = simulate_fixed_time_control(current_state_x, d_in_matrix, env)
                print(f"Fixed-time control result (for comparison): {fixed_result}")


                # 3. CRITICAL FIX: Simulate the environment step to get the next state
                # First column of disturbance matrix
                d_in_first_step = d_in_matrix[:, 0].reshape(-1, 1)

                current_state_x = simulate_env_step(
                    current_state_x,
                    u_executed_vector,
                    d_in_first_step,
                    env
                )

                # 4. Update u_last_executed for the next MPC call
                u_last_executed = u_executed_vector

            else:
                print("Warning: MPC solver failed. Maintaining previous signal phase.")
                # State and u_last are not updated if the solver fails.

            # --- Loop Timing ---
            elapsed_time = time.time() - start_time
            time.sleep(max(0, DELTA_T_SEC - elapsed_time))


if __name__ == "__main__":
    main()
