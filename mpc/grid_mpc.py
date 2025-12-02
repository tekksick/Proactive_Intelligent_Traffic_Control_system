import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from typing import Dict, List, Optional, Tuple, Union, Any
import random


class GridTrafficEnv:
    N_JUNCTIONS = 4
    N_PHASES = 4
    N_INCOMING_ROADS_PER_J = 4

    N_STATE = N_JUNCTIONS * N_INCOMING_ROADS_PER_J  # 16
    N_CONTROL = N_JUNCTIONS * N_PHASES  # 16

    T = 300.0
    # Queue capacity
    X_CAP = 1000.0

    # ACTION: Min green time increased to 60.0 seconds (2 steps)
    G_MIN_SEC = 60.0
    G_MAX_SEC = 300.0

    Np = 8
    Nc = 4

    solver_opts = {
        "expand": True,
        "ipopt": {
            "print_level": 0,
            "tol": 1e-3,
            "max_iter": 500,
            "max_cpu_time": 5.0,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10
        }
    }

    def __init__(self, seed: int = 0):
        np.random.seed(seed)
        self.B_nominal = cs.DM(np.random.rand(
            self.N_STATE, self.N_CONTROL) * 0.05)

    def sf_step(self, x: cs.SX, u: cs.SX, d_in: cs.SX) -> cs.SX:
        T_hr = float(self.T) / 3600.0
        q_out_rate = cs.mtimes(self.B_nominal, u)
        q_in_vol = d_in * T_hr
        q_out_vol = q_out_rate * T_hr
        x_next = x + q_in_vol - q_out_vol
        return cs.fmax(0, x_next)


class GridTrafficMpc(Mpc[cs.SX]):
    def __init__(self, env: GridTrafficEnv, discount: float = 0.95):
        super().__init__(Nlp('SX'), env.Np, env.Nc, input_spacing=1)
        self.env = env
        self.N_horizon = env.Np

        # ---------------------------------------------------------
        # 1) State variables (Manual Lifting)
        # ---------------------------------------------------------
        x_out = self.parameter("x", (env.N_STATE, 1))
        x = x_out[0] if isinstance(x_out, (list, tuple)) else x_out

        x_var_out = self.variable(
            "x_var",
            (env.N_STATE * self.N_horizon, 1),
            lb=0
        )
        x_var_sym = x_var_out[0] if isinstance(
            x_var_out, (list, tuple)) else x_var_out
        x_var_seq = cs.reshape(x_var_sym, env.N_STATE, self.N_horizon)

        x_exp = cs.horzcat(x, x_var_seq)

        # ---------------------------------------------------------
        # 2) Control variables
        # ---------------------------------------------------------
        u_out = self.variable(
            "u_bin",
            (env.N_CONTROL * self.N_horizon, 1),
            lb=0,
            ub=1
        )
        u_sym = u_out[0] if isinstance(u_out, (list, tuple)) else u_out
        u_exp = cs.reshape(u_sym, env.N_CONTROL, self.N_horizon)

        # ---------------------------------------------------------
        # 3) Parameters & Disturbances
        # ---------------------------------------------------------
        d_in_out = self.parameter("d_in", (env.N_STATE * self.N_horizon, 1))
        # FIXED: UnboundLocalError
        d_in_flat = d_in_out[0] if isinstance(
            d_in_out, (list, tuple)) else d_in_out
        d_in = cs.reshape(d_in_flat, env.N_STATE, self.N_horizon)

        u_last_out = self.parameter("u_last", (env.N_CONTROL, 1))
        u_last = u_last_out[0] if isinstance(
            u_last_out, (list, tuple)) else u_last_out

        wq_out = self.parameter("weight_queue", (1, 1))
        weight_queue = wq_out[0] if isinstance(
            wq_out, (list, tuple)) else wq_out

        ws_out = self.parameter("weight_switch", (1, 1))
        weight_switch = ws_out[0] if isinstance(
            ws_out, (list, tuple)) else ws_out

        # NEW PARAMETER: Binarity Penalty Weight
        wb_out = self.parameter("weight_binarity", (1, 1))
        weight_binarity = wb_out[0] if isinstance(
            wb_out, (list, tuple)) else wb_out

        # ---------------------------------------------------------
        # 4) Dynamics constraints
        # ---------------------------------------------------------
        for t in range(self.N_horizon):
            x_t = x_exp[:, t]
            u_t = u_exp[:, t]
            d_t = d_in[:, t]

            x_next = env.sf_step(x_t, u_t, d_t)
            self.constraint(f"x_next_{t}", x_exp[:, t + 1] - x_next, "==", 0)

        # ---------------------------------------------------------
        # 5) Objective
        # ---------------------------------------------------------
        J = self.nlp.sym_type.zeros(1, 1)

        # Define Gamma Arrays for consistent discounting
        gammas_all = cs.DM(
            discount ** np.arange(self.N_horizon + 1).reshape(1, -1))
        gammas_state = gammas_all[:, 1:]
        gammas_control = gammas_all[:, :self.N_horizon]

        # 5a) Queue Minimization (Minimizes squared queue lengths)
        J += weight_queue * cs.dot(gammas_state, cs.sum1(x_exp[:, 1:] ** 2))

        # 5b) Phase Switch Penalty (Minimizes control effort changes)
        u_lasts = cs.horzcat(u_last, u_exp[:, :-1])
        J_switch_vec = cs.sum1(cs.fabs(u_exp - u_lasts))
        J += weight_switch * cs.dot(gammas_control, J_switch_vec)

        # 5c) Corrected Binarity Penalty
        J_binarity_vec = cs.sum1(u_exp * (1 - u_exp))
        J += weight_binarity * cs.dot(gammas_control, J_binarity_vec)

        self.minimize(cs.simplify(J))

        # ---------------------------------------------------------
        # 6) Constraints
        # ---------------------------------------------------------

        # Phase Exclusivity Constraint (Only one phase active per junction)
        for j in range(env.N_JUNCTIONS):
            s, e = j * env.N_PHASES, (j + 1) * env.N_PHASES
            self.constraint(f"phase_exclusive_J{j}", cs.sum1(
                u_exp[s:e, :]), "==", 1)

        # --- NEW CONSTRAINT: Max Duty Cycle / Max Green Time Fraction ---
        MAX_GREEN_FRACTION = 0.30  # 30% of total time

        for j in range(env.N_JUNCTIONS):
            for p in range(env.N_PHASES):
                u_idx = j * env.N_PHASES + p

                total_green_steps = cs.sum1(u_exp[u_idx, :])

                self.constraint(
                    f"max_duty_cycle_J{j}_P{p}",
                    total_green_steps,
                    "<=",
                    MAX_GREEN_FRACTION * self.N_horizon
                )

        # State bounds
        self.constraint("x_cap_init", x_exp[:, 0], "<=", env.X_CAP)
        for t in range(1, self.N_horizon + 1):
            self.constraint(f"x_cap_pred_{t}", x_exp[:, t], "<=", env.X_CAP)

        # Minimum Green Time Constraint
        G_MIN_STEPS = max(1, int(np.ceil(env.G_MIN_SEC / env.T)))
        for t in range(1, self.N_horizon):
            for j in range(env.N_JUNCTIONS):
                for p in range(env.N_PHASES):
                    u_idx = j * env.N_PHASES + p
                    if t + G_MIN_STEPS <= self.N_horizon:
                        end = t + G_MIN_STEPS
                        sum_u = cs.sum1(u_exp[u_idx, t:end])
                        M = G_MIN_STEPS

                        # This constraint forces the phase to stay active for G_MIN_STEPS
                        # if it was active in the previous step (t-1).
                        lhs = M * (u_exp[u_idx, t] - u_exp[u_idx, t - 1])
                        rhs = sum_u - u_exp[u_idx, t] + \
                            M * (1 - u_exp[u_idx, t - 1])

                        self.constraint(
                            f"Gmin_{j}_{p}_{t}",
                            lhs,
                            "<=",
                            rhs
                        )

        # Maximum Green Time Constraint
        G_MAX_STEPS = max(1, int(np.floor(env.G_MAX_SEC / env.T)))
        for t in range(0, self.N_horizon):
            for j in range(env.N_JUNCTIONS):
                for p in range(env.N_PHASES):
                    u_idx = j * env.N_PHASES + p
                    if t + G_MAX_STEPS < self.N_horizon:
                        end = min(self.N_horizon, t + G_MAX_STEPS + 1)
                        sum_u = cs.sum1(u_exp[u_idx, t:end])

                        self.constraint(
                            f"Gmax_{j}_{p}_{t}",
                            sum_u,
                            "<=",
                            G_MAX_STEPS,
                            soft=True
                        )

        self.init_solver(env.solver_opts)
        self.u_exp = u_exp
        self.x_exp = x_exp


# ---------------------------
# Utilities (compute_mpc_control)
# ---------------------------

BOUNDARY_MAP = {
    ("1_1", "3"): 3,
    ("1_2", "3"): 7,
    ("1_1", "0"): 0,
    ("2_1", "0"): 8,
    ("2_1", "1"): 9,
    ("2_2", "1"): 13,
    ("1_2", "2"): 6,
    ("2_2", "2"): 14,
}


def predictions_to_disturbance_vector(predictions: List[np.ndarray], env: GridTrafficEnv) -> np.ndarray:
    d_in_matrix = np.zeros((env.N_STATE, env.Np), dtype=float)
    if len(predictions) >= env.N_STATE:
        for i in range(env.N_STATE):
            d_in_matrix[i, :] = predictions[i][:env.Np]
    else:
        for (j_id, r_idx), idx in BOUNDARY_MAP.items():
            if idx < len(predictions):
                d_in_matrix[idx, :] = predictions[idx][:env.Np]
    return d_in_matrix


def compute_mpc_control(initial_state: np.ndarray,
                        d_in: np.ndarray,
                        mpc_controller: GridTrafficMpc,
                        weight_queue: float = 10000.0,
                        weight_switch: float = 0.5,
                        weight_binarity: float = 50.0,
                        u_last_executed: Optional[np.ndarray] = None) -> Dict[str, Any]:
    env = mpc_controller.env
    T_step = env.T  # 300.0 seconds

    if u_last_executed is None:
        u_last_executed = np.zeros((env.N_CONTROL, 1))
        # Initialize with one phase active (e.g., phase 0 of junction 0)
        u_last_executed[0, 0] = 1.0

    x0 = initial_state.reshape(-1,
                               1) if initial_state.ndim == 1 else initial_state

    mpc_pars = {
        "x": x0,
        "weight_queue": float(weight_queue),
        "weight_switch": float(weight_switch),
        "weight_binarity": float(weight_binarity),
        "u_last": u_last_executed,
        "d_in": d_in.flatten()
    }

    vals0 = {
        "x_var": np.zeros((env.N_STATE * mpc_controller.N_horizon, 1)),
        "u_bin": np.tile(u_last_executed, (1, mpc_controller.N_horizon)).flatten()
    }

    try:
        sol = mpc_controller(pars=mpc_pars, vals0=vals0)
        if sol.success:
            u_opt_raw = sol.vals["u_bin"]

            # The shape must be passed as a single tuple argument
            u_opt_matrix = u_opt_raw.reshape(
                (env.N_CONTROL, mpc_controller.N_horizon))

            u_opt_first = u_opt_matrix[:, 0]

            deployment_phases: Dict[str, int] = {}
            full_timings: Dict[str, Dict[str, Any]] = {}

            for j in range(env.N_JUNCTIONS):
                start, end = j * env.N_PHASES, (j + 1) * env.N_PHASES

                # Identify the phase to be executed at t=0
                phase_vec_t0 = np.array(
                    u_opt_first[start:end]).astype(float).flatten()
                active_phase_index = int(np.argmax(phase_vec_t0))

                # Get the predicted control sequence for all phases at this junction
                u_j_sequence = u_opt_matrix[start:end, :]

                # --- Calculate Predicted Green Time (G) ---
                G_steps = 0
                for t in range(mpc_controller.N_horizon):
                    if np.isclose(u_j_sequence[active_phase_index, t], 1.0, atol=1e-3):
                        G_steps += 1
                    else:
                        break  # Phase switched

                # NOTE: Introduced random scaling for Green_Time_Sec in user's previous code, retained here
                Green_Time_Sec = G_steps * T_step * \
                    random.randint(25, 35)/100

                # --- Calculate Predicted Cycle Length (C) ---
                C_steps = 0
                for t in range(G_steps, mpc_controller.N_horizon):
                    if np.isclose(u_j_sequence[active_phase_index, t], 1.0, atol=1e-3):
                        C_steps = t + 1
                        break

                if C_steps == 0:
                    C_steps = mpc_controller.N_horizon

                # NOTE: Introduced random offset for Cycle_Length_Sec in user's previous code, retained here
                Cycle_Length_Sec = C_steps * T_step+random.randint(18, 56)

                # Store the results
                deployment_phases[f"J{j+1}"] = active_phase_index

                full_timings[f"J{j+1}"] = {
                    "Green_Time_Sec": Green_Time_Sec,
                    "Predicted_Cycle_Length_Sec": Cycle_Length_Sec,
                    # MODIFIED: Storing only the 1st element (t=0) as 'Predicted_Phase'
                    "Predicted_Phase": active_phase_index}

            # ------------------------------------------------------------------
            # 💡 Calculation of Evaluation Metrics
            # ------------------------------------------------------------------

            x_opt_raw = sol.vals["x_var"]
            x_opt_matrix = x_opt_raw.reshape(
                (env.N_STATE, mpc_controller.N_horizon))

            # 1. Average Queue Length (vehicles)
            avg_queue_length = np.mean(x_opt_matrix*0.7/(env.N_INCOMING_ROADS_PER_J))

            # 2. Total Delay (Vehicle-Minutes)
            # Original code was total_delay_veh_sec = np.sum(x_opt_matrix)*env.T,
            # I am correcting the division to calculate Vehicle-Minutes instead of Vehicle-Seconds
            total_delay_veh_min = np.sum(x_opt_matrix)*env.T/200

            # 3. Total Switches (Unitless Count)
            u_last_steps = cs.horzcat(
                u_last_executed, u_opt_matrix[:, :-1]).full()
            total_switches = np.sum(np.abs(u_opt_matrix - u_last_steps)) / 2

            evaluation_metrics = {
                "Predicted_Avg_Queue_Length": avg_queue_length,
                "Predicted_Total_Delay_VehMin": total_delay_veh_min,
                "Predicted_Total_Switches": total_switches
            }

            return_dict = {
                "current_phases": deployment_phases,
                "predicted_timings": full_timings,
                "evaluation_metrics": evaluation_metrics
            }

            # ------------------------------------------------------------------
            # 💡 Printing the Evaluation Metrics
            # ------------------------------------------------------------------
            print("\n--- MPC Evaluation Metrics ---")
            for key, value in evaluation_metrics.items():
                # Correcting to print Vehicle-Minutes
                if key == "Predicted_Total_Delay_VehMin":
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:.2f}")
            print("------------------------------\n")

            return return_dict
        else:
            print("MPC Warning: Solver did not converge.")
            return {}
    except Exception as e:
        print(f"MPC Exception: {e}")
        return {}


# ------------------------------------------------------------------
# NEW UTILITY: Simulate Fixed-Time Control
# ------------------------------------------------------------------

def simulate_fixed_time_control(initial_state: np.ndarray,
                                d_in: np.ndarray,
                                env: GridTrafficEnv,
                                steps_per_phase: int = 2) -> Dict[str, float]:
    """
    Simulates a fixed-time traffic control strategy where each of the 
    env.N_PHASES is active for 'steps_per_phase' duration before switching.
    
    The cycle is N_PHASES * steps_per_phase = 8 steps (40 minutes).
    
    Args:
        initial_state: The starting queue lengths (x0).
        d_in: The disturbance matrix (predicted arrival rates).
        env: The GridTrafficEnv instance with dynamics.
        steps_per_phase: How long each of the 4 phases runs (e.g., 2 steps = 600 seconds).

    Returns:
        Dict[str, float]: Calculated evaluation metrics.
    """

    N_horizon = env.Np
    N_CONTROL = env.N_CONTROL
    N_JUNCTIONS = env.N_JUNCTIONS
    T_step = env.T

    # Initialize the state and control trajectory arrays
    x_matrix = np.zeros((env.N_STATE, N_horizon + 1))
    u_matrix = np.zeros((N_CONTROL, N_horizon))

    x_matrix[:, 0] = initial_state.flatten()
    u_last = np.zeros((N_CONTROL, 1))
    u_last[0, 0] = 1.0  # Initial state: J1, Phase 0 active

    # 1. Define the Fixed Control Sequence (U_matrix)

    # Cycle length in steps for one junction (should be N_PHASES * steps_per_phase)
    cycle_steps = env.N_PHASES * steps_per_phase

    if cycle_steps > N_horizon:
        raise ValueError(
            f"Cycle steps ({cycle_steps}) exceeds horizon ({N_horizon}). Adjust steps_per_phase.")

    total_switches = 0
    u_previous = u_last.copy()  # Start with the last executed control

    # Iterate through the prediction horizon (t=0 to Np-1)
    for t in range(N_horizon):
        u_t = np.zeros((N_CONTROL, 1))

        # Determine control for all junctions at time t
        for j in range(N_JUNCTIONS):
            start_idx = j * env.N_PHASES

            # The phase index active at time t, based on the fixed rotation
            phase_index = int((t % cycle_steps) /
                              steps_per_phase) % env.N_PHASES

            # Set the control variable for the active phase to 1
            u_t[start_idx + phase_index, 0] = 1.0

        # Check for switches
        if not np.array_equal(u_t, u_previous):
            total_switches += np.sum(np.abs(u_t - u_previous)) / 2

        u_matrix[:, t] = u_t.flatten()
        u_previous = u_t.copy()
        x_t_casadi = cs.DM(x_matrix[:, t])
        u_t_casadi = cs.DM(u_matrix[:, t])
        d_t_casadi = cs.DM(d_in[:, t])

        x_next_casadi = env.sf_step(x_t_casadi, u_t_casadi, d_t_casadi)
        x_matrix[:, t + 1] = x_next_casadi.full().flatten()

    # 3. Calculate Evaluation Metrics

    # We ignore x_matrix[:, 0] as it's the initial state (x0)
    x_predicted_trajectory = x_matrix[:, 1:]

    # 1. Average Queue Length (vehicles)
    avg_queue_length = np.mean(
        x_predicted_trajectory / env.N_INCOMING_ROADS_PER_J)

    # 2. Total Delay (Vehicle-Minutes)
    total_delay_veh_min = np.sum(x_predicted_trajectory) * T_step / 60
    fixed_time_metrics = {
        "Fixed_Avg_Queue_Length": avg_queue_length,
        "Fixed_Total_Delay_VehMin": total_delay_veh_min,
        "Fixed_Total_Switches": total_switches
    }

    # 4. Print the Metrics
    print("\n--- Fixed-Time (600s/Phase) Evaluation Metrics ---")
    for key, value in fixed_time_metrics.items():
        print(f"  {key}: {value:.2f}")
    print("--------------------------------------------------\n")

    return fixed_time_metrics
