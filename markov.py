import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


class MarkovTrafficPredictor:
    def __init__(self, csv_file_path):
        """
        Initialize the Markov Traffic Predictor
        
        Args:
            csv_file_path (str): Path to the CSV file containing traffic data
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.states = None
        self.transition_matrix = None
        self.state_counts = {'ff': 0, 'mf': 0, 'hf': 0}
        self.state_changes = 0

    def load_data(self):
        """Load traffic data from CSV file"""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Columns: {self.data.columns.tolist()}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def classify_traffic_state(self, vehicle_count):
        """
        Classify traffic state based on vehicle count
        
        Criteria:
        - Free Flow (ff): 0 ≤ count < 120
        - Moderate Flow (mf): 120 ≤ count < 220
        - High Flow (hf): count ≥ 220
        
        Args:
            vehicle_count (int): Number of vehicles
            
        Returns:
            str: Traffic state ('ff', 'mf', or 'hf')
        """
        if 0 <= vehicle_count < 120:
            return 'ff'
        elif 120 <= vehicle_count < 220:
            return 'mf'
        else:
            return 'hf'

    def prepare_states(self, vehicle_column='No. of Vehicles'):
        """
        Convert vehicle counts to traffic states
        
        Args:
            vehicle_column (str): Column name containing vehicle counts
        """
        if self.data is None:
            print("Error: Data not loaded. Call load_data() first.")
            return

        # Handle different possible column names
        possible_columns = [vehicle_column,
                            'No_of_Vehicles', 'vehicles', 'count']
        column_to_use = None

        for col in possible_columns:
            if col in self.data.columns:
                column_to_use = col
                break

        if column_to_use is None:
            # If no matching column found, use the first numeric column
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                column_to_use = numeric_cols[0]
                print(f"Using column '{column_to_use}' as vehicle count")
            else:
                print("Error: No suitable numeric column found")
                return

        # Convert vehicle counts to states
        self.states = [self.classify_traffic_state(
            count) for count in self.data[column_to_use]]

        # Count states
        for state in self.states:
            self.state_counts[state] += 1

        print(f"States prepared. Total observations: {len(self.states)}")
        print(f"State distribution: {self.state_counts}")

    def calculate_state_changes(self):
        """Calculate the number of state changes in the data"""
        if self.states is None or len(self.states) < 2:
            print("Error: States not prepared or insufficient data")
            return 0

        changes = 0
        for i in range(1, len(self.states)):
            if self.states[i] != self.states[i-1]:
                changes += 1

        self.state_changes = changes
        print(f"Number of state changes: {self.state_changes}")
        return self.state_changes

    def build_transition_matrix(self):
        """Build the transition probability matrix"""
        if self.states is None:
            print("Error: States not prepared. Call prepare_states() first.")
            return

        # Initialize transition counts
        transitions = defaultdict(lambda: defaultdict(int))
        state_totals = defaultdict(int)

        # Count transitions
        for i in range(len(self.states) - 1):
            current_state = self.states[i]
            next_state = self.states[i + 1]
            transitions[current_state][next_state] += 1
            state_totals[current_state] += 1

        # Convert to probabilities
        state_names = ['ff', 'mf', 'hf']
        self.transition_matrix = np.zeros((3, 3))

        for i, from_state in enumerate(state_names):
            for j, to_state in enumerate(state_names):
                if state_totals[from_state] > 0:
                    self.transition_matrix[i, j] = transitions[from_state][to_state] / \
                        state_totals[from_state]

        print("Transition Matrix:")
        print("    ", "    ".join(state_names))
        for i, from_state in enumerate(state_names):
            row_str = f"{from_state}  "
            for j in range(3):
                row_str += f"{self.transition_matrix[i, j]:.3f}  "
            print(row_str)

    def predict_next_states(self, current_state, steps=5):
        """
        Predict next traffic states using the Markov model
        
        Args:
            current_state (str): Current traffic state ('ff', 'mf', or 'hf')
            steps (int): Number of future steps to predict
            
        Returns:
            list: Predicted states for the next steps
        """
        if self.transition_matrix is None:
            print(
                "Error: Transition matrix not built. Call build_transition_matrix() first.")
            return []

        state_names = ['ff', 'mf', 'hf']
        state_idx = {'ff': 0, 'mf': 1, 'hf': 2}

        predictions = []
        current_idx = state_idx[current_state]

        for step in range(steps):
            # Get probabilities for next state
            probs = self.transition_matrix[current_idx]

            # Choose next state based on highest probability
            next_idx = np.argmax(probs)
            next_state = state_names[next_idx]

            predictions.append(next_state)
            current_idx = next_idx

        return predictions

    def predict_probabilistic(self, current_state, steps=5, num_simulations=1000):
        """
        Predict next states using probabilistic sampling
        
        Args:
            current_state (str): Current traffic state
            steps (int): Number of future steps
            num_simulations (int): Number of Monte Carlo simulations
            
        Returns:
            dict: Probability distribution for each future step
        """
        if self.transition_matrix is None:
            print("Error: Transition matrix not built.")
            return {}

        state_names = ['ff', 'mf', 'hf']
        state_idx = {'ff': 0, 'mf': 1, 'hf': 2}

        results = []

        for step in range(steps):
            step_results = []

            for _ in range(num_simulations):
                current_idx = state_idx[current_state]
                path = []

                for s in range(step + 1):
                    probs = self.transition_matrix[current_idx]
                    next_idx = np.random.choice(3, p=probs)
                    path.append(state_names[next_idx])
                    current_idx = next_idx

                step_results.append(path[-1])

            # Calculate probabilities for this step
            step_probs = {}
            for state in state_names:
                step_probs[state] = step_results.count(state) / num_simulations

            results.append(step_probs)

        return results

    def visualize_results(self):
        """Visualize the analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. State distribution
        axes[0, 0].bar(self.state_counts.keys(), self.state_counts.values())
        axes[0, 0].set_title('Traffic State Distribution')
        axes[0, 0].set_ylabel('Count')

        # 2. Time series of states
        # Only plot if not too many points
        if self.states and len(self.states) <= 100:
            state_numeric = [{'ff': 0, 'mf': 1, 'hf': 2}[s]
                             for s in self.states]
            axes[0, 1].plot(state_numeric, 'o-')
            axes[0, 1].set_title('Traffic States Over Time')
            axes[0, 1].set_ylabel('State (0=ff, 1=mf, 2=hf)')
            axes[0, 1].set_xlabel('Time')
        else:
            axes[0, 1].text(0.5, 0.5, 'Too many data points\nfor visualization',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Traffic States Over Time')

        # 3. Transition matrix heatmap
        if self.transition_matrix is not None:
            im = axes[1, 0].imshow(self.transition_matrix, cmap='Blues')
            axes[1, 0].set_title('Transition Matrix')
            axes[1, 0].set_xticks(range(3))
            axes[1, 0].set_yticks(range(3))
            axes[1, 0].set_xticklabels(['ff', 'mf', 'hf'])
            axes[1, 0].set_yticklabels(['ff', 'mf', 'hf'])

            # Add text annotations
            for i in range(3):
                for j in range(3):
                    axes[1, 0].text(j, i, f'{self.transition_matrix[i, j]:.3f}',
                                    ha='center', va='center')

        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Summary Statistics:
        
        Total Observations: {len(self.states) if self.states else 0}
        State Changes: {self.state_changes}
        
        State Counts:
        Free Flow (ff): {self.state_counts['ff']}
        Moderate Flow (mf): {self.state_counts['mf']}
        High Flow (hf): {self.state_counts['hf']}
        
        Change Rate: {self.state_changes/(len(self.states)-1)*100:.1f}%
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

    def run_analysis(self, vehicle_column='No. of Vehicles'):
        """Run the complete analysis pipeline"""
        print("=" * 50)
        print("MARKOV TRAFFIC PREDICTION ANALYSIS")
        print("=" * 50)

        # Load data
        if not self.load_data():
            return

        # Prepare states
        self.prepare_states(vehicle_column)
        if self.states is None:
            return

        # Calculate state changes
        self.calculate_state_changes()

        # Build transition matrix
        self.build_transition_matrix()

        # Example predictions
        print("\n" + "=" * 30)
        print("PREDICTION EXAMPLES")
        print("=" * 30)

        for initial_state in ['ff', 'mf', 'hf']:
            predictions = self.predict_next_states(initial_state, steps=5)
            print(f"From {initial_state}: {' -> '.join(predictions)}")

        # Probabilistic predictions
        print("\nProbabilistic predictions from 'ff' state:")
        prob_predictions = self.predict_probabilistic('ff', steps=3)
        for i, step_probs in enumerate(prob_predictions):
            print(f"Step {i+1}: {step_probs}")

        # Visualize results
        self.visualize_results()


# Example usage
if __name__ == "__main__":
    # Initialize the predictor with your CSV file
    predictor = MarkovTrafficPredictor("traffic_data_table1.csv")

    # Run the complete analysis
    predictor.run_analysis()

    # You can also run individual components:
    # predictor.load_data()
    # predictor.prepare_states()
    # predictor.calculate_state_changes()
    # predictor.build_transition_matrix()
    # predictions = predictor.predict_next_states('ff', steps=10)
    # print(f"Predictions: {predictions}")
