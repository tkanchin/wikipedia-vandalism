"""
Anomaly Detection TPR Analysis Library

This module provides tools for analyzing true positive rates (TPR) in anomaly detection
systems, including:
- Probability calibration
- TPR analysis by probability deciles
- TPR decay modeling as k increases
- Extrapolation to higher k values

Usage:
    from anomaly_detection_analysis import *
    
    # Load data
    df = load_data('your_data.csv')
    
    # Analyze and plot
    analyzer = AnomalyDetectionAnalyzer(df)
    analyzer.calibrate_probabilities()
    analyzer.plot_tpr_by_decile()
    analyzer.plot_calibration_curve()
    analyzer.plot_tpr_decay(max_k=1200, resource_constraint=800)
    
    # Get extrapolated values
    results = analyzer.get_extrapolation_results()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression


class AnomalyDetectionAnalyzer:
    """
    Class for analyzing anomaly detection model performance and extrapolating
    to higher k values.
    """
    
    def __init__(self, data, prob_col='prob_score', tp_col='is_true_positive'):
        """
        Initialize the analyzer with data.
        
        Parameters:
        - data: DataFrame with probability scores and true positive indicators
        - prob_col: Name of the probability score column
        - tp_col: Name of the true positive indicator column (1=TP, 0=FP)
        """
        self.data = data.copy()
        self.prob_col = prob_col
        self.tp_col = tp_col
        
        # Sort by probability score in descending order
        self.data = self.data.sort_values(by=prob_col, ascending=False).reset_index(drop=True)
        
        # Initialize additional attributes
        self.calibrated = False
        self.calibrated_prob_col = 'calibrated_prob'
        self.calibrator = None
        self.model_params = None
        self.extrapolation_results = None
    
    def calibrate_probabilities(self):
        """
        Calibrate probability scores using isotonic regression.
        """
        print("Calibrating probabilities using isotonic regression...")
        
        # Apply isotonic regression
        ir = IsotonicRegression(out_of_bounds='clip')
        
        # Fit and transform
        self.data[self.calibrated_prob_col] = ir.fit_transform(
            self.data[self.prob_col], 
            self.data[self.tp_col]
        )
        
        # Store calibrator for future use
        self.calibrator = ir
        self.calibrated = True
        
        # Re-sort by calibrated probability
        self.data = self.data.sort_values(
            by=self.calibrated_prob_col, 
            ascending=False
        ).reset_index(drop=True)
        
        # Check if calibration improved monotonicity
        self._check_monotonicity()
        
        return self
    
    def _check_monotonicity(self, num_bins=10):
        """
        Check if calibration improved monotonicity in TPR by probability decile.
        """
        # Create deciles safely handling potential duplicates
        try:
            # For original probabilities
            self.data['orig_decile'] = pd.qcut(
                self.data[self.prob_col].rank(method='first'),
                num_bins, 
                labels=False
            )
            
            # For calibrated probabilities
            if self.calibrated:
                self.data['calib_decile'] = pd.qcut(
                    self.data[self.calibrated_prob_col].rank(method='first'),
                    num_bins, 
                    labels=False
                )
        except ValueError as e:
            print(f"Warning when creating deciles: {e}")
            # Fallback to equal-sized groups
            n = len(self.data)
            self.data['orig_decile'] = np.floor(np.arange(n) / (n/num_bins)).astype(int)
            if self.calibrated:
                self.data['calib_decile'] = self.data['orig_decile'].copy()
        
        # Calculate TPR by decile
        orig_tpr = self.data.groupby('orig_decile')[self.tp_col].mean()
        
        # Check monotonicity (should decrease as decile number increases)
        orig_monotonic = (orig_tpr.diff().dropna() <= 0).all()
        
        if self.calibrated:
            calib_tpr = self.data.groupby('calib_decile')[self.tp_col].mean()
            calib_monotonic = (calib_tpr.diff().dropna() <= 0).all()
            print(f"Original probabilities monotonic: {orig_monotonic}")
            print(f"Calibrated probabilities monotonic: {calib_monotonic}")
        else:
            print(f"Original probabilities monotonic: {orig_monotonic}")
    
    def plot_tpr_by_decile(self, num_bins=10, figsize=(12, 6)):
        """
        Plot TPR by probability decile for both original and calibrated probabilities.
        """
        # Ensure deciles exist
        if 'orig_decile' not in self.data.columns:
            self._check_monotonicity(num_bins)
        
        # Calculate TPR by decile
        orig_decile_tpr = self.data.groupby('orig_decile')[self.tp_col].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=figsize)
        
        if self.calibrated:
            calib_decile_tpr = self.data.groupby('calib_decile')[self.tp_col].mean().reset_index()
            
            # Original probabilities
            bars1 = plt.bar(orig_decile_tpr['orig_decile'] - 0.2, 
                          orig_decile_tpr[self.tp_col], 
                          width=0.4, label='Original Probabilities')
            
            # Calibrated probabilities
            bars2 = plt.bar(calib_decile_tpr['calib_decile'] + 0.2, 
                          calib_decile_tpr[self.tp_col], 
                          width=0.4, label='Calibrated Probabilities')
            
            # Add value labels on top of bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.title('TPR by Probability Decile - Original vs Calibrated')
            plt.legend()
            
        else:
            # Only original probabilities
            bars = plt.bar(orig_decile_tpr['orig_decile'], 
                         orig_decile_tpr[self.tp_col])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.title('TPR by Probability Decile')
        
        # Add labels
        plt.xlabel('Probability Decile (High to Low)')
        plt.ylabel('True Positive Rate')
        
        # Reverse x-axis to show highest probabilities first
        plt.gca().invert_xaxis()
        
        return plt
    
    def plot_calibration_curve(self, figsize=(8, 6)):
        """
        Plot the probability calibration curve.
        """
        if not self.calibrated:
            print("Warning: Probabilities not calibrated. Call calibrate_probabilities() first.")
            return None
        
        # Sort values by original probability for plotting
        calib_df = self.data[[self.prob_col, self.calibrated_prob_col]].sort_values(self.prob_col)
        
        plt.figure(figsize=figsize)
        plt.plot(calib_df[self.prob_col], calib_df[self.calibrated_prob_col], 
                'b-', label='Calibration function')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Add labels and title
        plt.xlabel('Original Probability')
        plt.ylabel('Calibrated Probability')
        plt.title('Probability Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def _power_law_model(self, k, a, b, c):
        """Simple power law model with offset."""
        return a * (k ** -b) + c
    
    def _two_phase_model(self, k, a, b, c, k_threshold):
        """Two-phase model with power law decay until k_threshold, then constant."""
        result = np.zeros_like(k, dtype=float)
        
        # Phase 1: Power law decay for k < k_threshold
        idx_decay = k < k_threshold
        result[idx_decay] = a * (k[idx_decay] ** -b) + c
        
        # Phase 2: Constant for k >= k_threshold
        idx_const = k >= k_threshold
        if np.any(idx_const):
            # Calculate the value at the threshold for continuity
            threshold_val = a * (k_threshold ** -b) + c
            result[idx_const] = threshold_val
            
        return result
    
    def _continuous_decay_model(self, k, a, b, c, d):
        """
        Model TPR decay with a continuous power law that gradually declines.
        
        Parameters:
        - a, b: Shape parameters for the initial decay
        - c: Background TPR level (primary component)
        - d: Decay rate for the tail (smaller = slower decay)
        """
        # Base power law component for initial decay
        power_decay = a * (k ** -b)
        
        # Gradual exponential decay after k=300
        exp_decay = c * np.exp(-d * np.maximum(0, (k - 300) / 700))
        
        return power_decay + exp_decay
    
    def plot_tpr_decay(self, current_k=300, max_k=1200, resource_constraint=None, 
                      model_type='continuous', figsize=(10, 6)):
        """
        Plot TPR decay as k increases.
        
        Parameters:
        - current_k: Current number of reviews
        - max_k: Maximum k to extrapolate to
        - resource_constraint: Resource constraint to mark on the plot
        - model_type: Type of model to use ('power', 'two_phase', or 'continuous')
        - figsize: Figure size
        
        Returns:
        - Matplotlib plot
        """
        # Use calibrated probabilities if available
        prob_col = self.calibrated_prob_col if self.calibrated else self.prob_col
        
        # Prepare data for k values
        max_k = min(max_k, len(self.data))
        
        # Calculate observed TPR at different k values up to current_k
        k_values_observed = np.arange(10, current_k + 1, 10)  # steps of 10
        tpr_observed = []
        
        for k in k_values_observed:
            tpr_observed.append(self.data.iloc[:k][self.tp_col].mean())
        
        # Choose model function based on model_type
        if model_type == 'power':
            model_func = self._power_law_model
            p0 = [0.5, 0.5, 0.05]  # Initial guesses
            bounds = ([0, 0, 0], [1, 3, 0.2])  # Parameter bounds
        elif model_type == 'two_phase':
            model_func = self._two_phase_model
            p0 = [0.5, 0.5, 0.05, 250]  # Initial guesses
            bounds = ([0, 0, 0, 100], [1, 3, 0.2, 400])  # Parameter bounds
        else:  # 'continuous' is default
            model_func = self._continuous_decay_model
            p0 = [0.3, 0.2, 0.08, 0.001]  # Initial guesses
            bounds = ([0, 0, 0, 0], [1, 1, 0.2, 0.01])  # Parameter bounds
        
        # Fit the model to observed data
        try:
            params, _ = curve_fit(model_func, k_values_observed, tpr_observed, 
                               p0=p0, bounds=bounds, maxfev=10000)
            
            # Store parameters for later use
            self.model_params = params
            self.model_type = model_type
            
            # Generate predictions for all k values
            k_values_all = np.arange(10, max_k + 1, 10)
            tpr_predicted = model_func(k_values_all, *params)
            
        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            # Fallback to simple model if fitting fails
            if model_type == 'power':
                params = [0.3, 0.2, 0.05]
            elif model_type == 'two_phase':
                params = [0.3, 0.2, 0.05, 250]
            else:  # continuous
                params = [0.3, 0.2, 0.07, 0.0005]
            
            # Store parameters for later use
            self.model_params = params
            self.model_type = model_type
            
            # Generate predictions
            k_values_all = np.arange(10, max_k + 1, 10)
            tpr_predicted = model_func(k_values_all, *params)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot observed and predicted values
        plt.scatter(k_values_observed, tpr_observed, color='blue', label='Observed TPR')
        plt.plot(k_values_all, tpr_predicted, 'r-', label='Predicted TPR')
        
        # Add vertical lines
        plt.axvline(x=current_k, color='green', linestyle='--', 
                   label=f'Current k={current_k}')
        
        if resource_constraint:
            plt.axvline(x=resource_constraint, color='orange', linestyle='--', 
                       label=f'Resource Constraint k={resource_constraint}')
        
        # Add labels and title
        plt.xlabel('k (Number of Reviews)')
        plt.ylabel('True Positive Rate')
        
        prob_type = "Calibrated" if self.calibrated else "Original"
        plt.title(f'TPR Decay as k Increases (Using {prob_type} Probabilities)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Store extrapolation results
        self._calculate_extrapolation_results(k_values_all, tpr_predicted, model_func)
        
        return plt
    
    def _calculate_extrapolation_results(self, k_values, tpr_predicted, model_func):
        """
        Calculate and store extrapolation results.
        """
        # Key k values to report
        k_values_key = [100, 300, 500, 800, 1000, 1200]
        
        # Filter to only include k values that are available
        k_values_key = [k for k in k_values_key if k <= len(self.data)]
        
        # Create results dictionary
        results = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'predictions': {}
        }
        
        # Generate model formula string
        if self.model_type == 'power':
            a, b, c = self.model_params
            results['model_formula'] = f"TPR = {a:.4f} * k^(-{b:.4f}) + {c:.4f}"
        elif self.model_type == 'two_phase':
            a, b, c, k_threshold = self.model_params
            results['model_formula'] = (
                f"TPR = {a:.4f} * k^(-{b:.4f}) + {c:.4f} for k < {k_threshold:.1f}\n"
                f"TPR = constant for k >= {k_threshold:.1f}"
            )
        else:  # continuous
            a, b, c, d = self.model_params
            results['model_formula'] = (
                f"TPR = {a:.4f} * k^(-{b:.4f}) + {c:.4f} * exp(-{d:.6f} * (k-300)/700)"
            )
        
        # Calculate predictions for key k values
        for k in k_values_key:
            # Find closest k in our predictions
            closest_idx = np.argmin(np.abs(k_values - k))
            pred_tpr = tpr_predicted[closest_idx]
            expected_tp = k * pred_tpr
            
            results['predictions'][k] = {
                'tpr': float(pred_tpr),
                'expected_tp': float(expected_tp)
            }
        
        # Store results
        self.extrapolation_results = results
        
        return results
    
    def get_extrapolation_results(self):
        """
        Get extrapolation results.
        
        Returns:
        - Dictionary with model parameters and predictions
        """
        if not self.extrapolation_results:
            print("No extrapolation results available. Call plot_tpr_decay() first.")
            return None
        
        return self.extrapolation_results
    
    def print_extrapolation_summary(self):
        """
        Print a summary of the extrapolation results.
        """
        if not self.extrapolation_results:
            print("No extrapolation results available. Call plot_tpr_decay() first.")
            return
        
        results = self.extrapolation_results
        
        print(f"TPR Decay Model Parameters ({results['model_type']} model):")
        print(results['model_formula'])
        
        print("\nPredicted TPR at key k values:")
        for k, prediction in results['predictions'].items():
            tpr = prediction['tpr']
            expected_tp = prediction['expected_tp']
            print(f"k={k}: TPR = {tpr:.4f}, Expected TP = {expected_tp:.1f}")


def load_data(file_path, prob_col='prob_score', tp_col='is_true_positive'):
    """
    Load data from CSV file.
    
    Parameters:
    - file_path: Path to CSV file
    - prob_col: Name of the probability score column
    - tp_col: Name of the true positive indicator column (1=TP, 0=FP)
    
    Returns:
    - DataFrame with probability scores and true positive indicators
    """
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if prob_col not in df.columns:
        raise ValueError(f"Probability column '{prob_col}' not found in the data")
    
    if tp_col not in df.columns:
        raise ValueError(f"True positive column '{tp_col}' not found in the data")
    
    return df


def run_complete_analysis(file_path, prob_col='prob_score', tp_col='is_true_positive',
                         current_k=300, max_k=1200, resource_constraint=800,
                         model_type='continuous', calibrate=True):
    """
    Run complete analysis pipeline and generate all plots.
    
    Parameters:
    - file_path: Path to CSV file
    - prob_col: Name of the probability score column
    - tp_col: Name of the true positive indicator column
    - current_k: Current k value (number of reviews)
    - max_k: Maximum k to extrapolate to
    - resource_constraint: Resource constraint to mark on plots
    - model_type: Type of model to use ('power', 'two_phase', or 'continuous')
    - calibrate: Whether to calibrate probabilities
    
    Returns:
    - AnomalyDetectionAnalyzer instance with all analysis completed
    """
    # Load data
    print(f"Loading data from {file_path}...")
    df = load_data(file_path, prob_col, tp_col)
    
    # Create analyzer
    analyzer = AnomalyDetectionAnalyzer(df, prob_col, tp_col)
    
    # Calibrate probabilities if requested
    if calibrate:
        analyzer.calibrate_probabilities()
    
    # Generate plots
    print("Generating TPR by decile plot...")
    analyzer.plot_tpr_by_decile().savefig('tpr_by_decile.png', dpi=300)
    
    if calibrate:
        print("Generating calibration curve plot...")
        analyzer.plot_calibration_curve().savefig('calibration_curve.png', dpi=300)
    
    print(f"Generating TPR decay plot with {model_type} model...")
    analyzer.plot_tpr_decay(
        current_k=current_k,
        max_k=max_k,
        resource_constraint=resource_constraint,
        model_type=model_type
    ).savefig('tpr_decay.png', dpi=300)
    
    # Print extrapolation summary
    print("\nExtrapolation Results:")
    analyzer.print_extrapolation_summary()
    
    return analyzer


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = 'anomaly_detection_data.csv'
    
    # Run complete analysis
    analyzer = run_complete_analysis(
        file_path=file_path,
        prob_col='prob_score',
        tp_col='is_true_positive',
        current_k=300,
        max_k=1200,
        resource_constraint=800,
        model_type='continuous',
        calibrate=True
    )
    
    # Access results programmatically
    results = analyzer.get_extrapolation_results()
    
    # Example: Get expected true positives at resource constraint
    expected_tp_at_constraint = results['predictions'][800]['expected_tp']
    print(f"\nExpected true positives at resource constraint (k=800): {expected_tp_at_constraint:.1f}")