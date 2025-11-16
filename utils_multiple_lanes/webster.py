from utils.webster import WebsterSignalTimer

INTERSECTION_PHASE_MAP = {
    "1_2": [0, 1, 2, 3],        # top-left intersection
    "2_2": [4, 5, 6, 7],        # top-right intersection
    "1_1": [8, 9, 10, 11],      # bottom-left intersection
    "2_1": [12, 13, 14, 15],    # bottom-right intersection
}


def compute_webster_for_grid(predictions, sat_flow_per_phase=1800):
    """
    Compute Webster timings for a 2×2 CityFlow grid using 16 predicted flows.
    
    Args:
        predictions: list of 16 lane volumes (veh/hr)
        sat_flow_per_phase: saturation flow for each approach
    
    Returns:
        dict:
            {
              "1_2": {...},
              "2_2": {...},
              "1_1": {...},
              "2_1": {...}
            }
    """

    if len(predictions) != 16:
        raise ValueError(f"Expected 16 predictions. Got {len(predictions)}.")

    # Same saturation flow for all phases
    saturation_flows = [sat_flow_per_phase] * 4

    results = {}

    for inter_id, phase_indices in INTERSECTION_PHASE_MAP.items():

        # pick the 4 flows for this intersection
        flows = [predictions[i] for i in phase_indices]

        # create Webster object
        timer = WebsterSignalTimer(
            num_phases=4,
            saturation_flows=saturation_flows,
            amber_time=3,
            all_red_time=1,
            min_cycle_time=40,
            max_cycle_time=120
        )

        # compute timings
        result = timer.calculate_timings(flows)

        # store
        results[inter_id] = result

    return results
