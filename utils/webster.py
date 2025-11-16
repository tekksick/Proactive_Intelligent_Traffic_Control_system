import math

class WebsterSignalTimer:
    """
    Implements the Webster method to calculate traffic signal cycle times
    and green-light durations for a single junction.

    This class uses the standard Webster formulas:
    - C_0 = (1.5 * L + 5) / (1 - Y)
    - Y = sum(q_i / s_i)
    - G_i = (y_i / Y) * (C_0 - L)
    """
    
    def __init__(self, 
                 num_phases, 
                 saturation_flows, 
                 amber_time = 3, 
                 all_red_time = 1,
                 min_cycle_time = 40,
                 max_cycle_time = 120,
                 saturation_threshold = 0.95):
        """
        Initializes the Webster Method calculator.

        Args:
            num_phases (int): The number of conflicting phases (e.g., 4 for a 
                              standard 4-way intersection).
            saturation_flows (list[int]): A list of the saturation flow rate (s_i) 
                                          for each phase in [vehicles per hour].
                                          The length of this list must equal num_phases. 
                                         
            amber_time (int): Amber light duration in seconds (constant for all phases).
            all_red_time (int): All-red duration (clearance interval) between 
                                phases in seconds.
            min_cycle_time (int): Minimum practical cycle time in seconds.
            max_cycle_time (int): Maximum practical cycle time in seconds.
            saturation_threshold (float): The practical limit for Y (sum of flow ratios)
                                          to trigger max_cycle_time (e.g., 0.95).
        """
        if len(saturation_flows) != num_phases:
            raise ValueError("Length of 'saturation_flows' must equal 'num_phases'.")
        
        self.num_phases = num_phases
        self.saturation_flows = saturation_flows
        self.amber_time = amber_time
        
        # Calculate the total lost time for *one* phase (l_i)
        # This is the time in a phase that is not effective green (amber + all-red).
        self.lost_time_per_phase = self.amber_time + all_red_time
        
        # Calculate the Total Lost Time for the *entire cycle* (L)
        self.L_total_cycle_lost_time = self.num_phases * self.lost_time_per_phase
        
        self.min_cycle_time = min_cycle_time
        self.max_cycle_time = max_cycle_time
        self.saturation_threshold = saturation_threshold

    def calculate_timings(self, vehicle_flow_rates: list[int]) -> dict:
        """
        Calculates the optimal cycle time and green light duration for each phase.

        Args:
            vehicle_flow_rates (list[int]): A list of the current or predicted 
                                            vehicle flow rate (q_i) for each 
                                            phase in [vehicles per hour].
                                            Must have length == num_phases.

        Returns:
            dict: A dictionary containing the total cycle time and a list of
                  timings (green, amber, red) for each phase.
        """
        if len(vehicle_flow_rates) != self.num_phases:
            raise ValueError("Length of 'vehicle_flow_rates' must equal 'num_phases'.")
        
        # 1. Calculate Flow Ratios (y_i = q_i / s_i)
        y_ratios = []
        for i in range(self.num_phases):
            # Handle potential division by zero if saturation flow is 0
            if self.saturation_flows[i] == 0:
                y_ratios.append(0.0)
            else:
                y_ratios.append(vehicle_flow_rates[i] / self.saturation_flows[i])
        
        # 2. Calculate Sum of Critical Ratios (Y)
        Y = sum(y_ratios)
        
        # 3. Calculate Optimum Cycle Length (C_0)
        C0 = 0.0
        if Y == 0:
            # No traffic, use minimum cycle time
            C0 = self.min_cycle_time
        elif Y >= self.saturation_threshold:
            # Junction is saturated or oversaturated, use max cycle time
            C0 = self.max_cycle_time
        else:
            # Use Webster's formula
            C0 = (1.5 * self.L_total_cycle_lost_time + 5) / (1.0 - Y)
        
        # 4. Enforce min/max cycle time limits
        C0 = max(self.min_cycle_time, min(self.max_cycle_time, C0))
        
        # 5. Calculate Total Available Green Time
        # This is (C_0 - L) from the formula
        # This is the total time in the cycle that will be "green".
        total_green_time = C0 - self.L_total_cycle_lost_time
        
        # Ensure total green time is not negative (can happen if min_cycle_time
        # is less than total lost time)
        if total_green_time < 0:
            total_green_time = 0
            # Recalculate C0 if it was too small
            C0 = self.L_total_cycle_lost_time 
        
        # 6. Distribute Green Time to each phase (G_i)
        phase_timings = []
        
        if Y == 0:
            # No traffic, just split the minimum green time equally
            green_time = total_green_time / self.num_phases
            for _ in range(self.num_phases):
                phase_timings.append({
                    'green': round(green_time),
                    'amber': self.amber_time,
                    'red': round(C0 - green_time - self.amber_time)
                })
        else:
            # Distribute green time based on the flow ratio for each phase
            # This implements G_N = (Y_N / Y) * (C_0 - L)
            for y_i in y_ratios:
                green_time = (y_i / Y) * total_green_time
                
                # Calculate red time for this phase
                red_time = C0 - green_time - self.amber_time
                
                phase_timings.append({
                    'green': round(green_time),
                    'amber': self.amber_time,
                    'red': round(red_time)
                })

        # Return the results
        return {
            'total_cycle_time_s': round(C0),
            'phase_timings': phase_timings
        }
