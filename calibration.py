"""
Anomaly Detection Precision@k Analysis Library

This module provides tools for analyzing precision at different k thresholds (Precision@k) 
in anomaly detection systems, including:
- Probability calibration
- Precision analysis by probability deciles
- Precision@k curve modeling as k increases
- Extrapolation to higher k values
- Model monitoring thresholds (green/yellow/red zones)

Usage:
    from anomaly_detection_analysis import *
    
    # Load data
    df = load_data('your_data.csv')
    
    # Analyze and plot
    analyzer = AnomalyDetectionAnalyzer(df)
    analyzer.calibrate_probabilities()
    analyzer.plot_precision_by_decile()
    analyzer.plot_calibration_curve()
    analyzer.plot_precision_at_k(max_k=1200, resource_constraint=800)
    
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
        Check if calibration improved monotonicity in precision by probability decile.
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
        
        # Calculate precision by decile
        orig_precision = self.data.groupby('orig_decile')[self.tp_col].mean()
        
        # Check monotonicity (should decrease as decile number increases)
        orig_monotonic = (orig_precision.diff().dropna() <= 0).all()
        
        if self.calibrated:
            calib_precision = self.data.groupby('calib_decile')[self.tp_col].mean()
            calib_monotonic = (calib_precision.diff().dropna() <= 0).all()
            print(f"Original probabilities monotonic: {orig_monotonic}")
            print(f"Calibrated probabilities monotonic: {calib_monotonic}")
        else:
            print(f"Original probabilities monotonic: {orig_monotonic}")
    
    def plot_precision_by_decile(self, num_bins=10, figsize=(12, 6)):
        """
        Plot precision by probability decile for both original and calibrated probabilities.
        """
        # Ensure deciles exist
        if 'orig_decile' not in self.data.columns:
            self._check_monotonicity(num_bins)
        
        # Calculate precision by decile
        orig_decile_precision = self.data.groupby('orig_decile')[self.tp_col].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=figsize)
        
        if self.calibrated:
            calib_decile_precision = self.data.groupby('calib_decile')[self.tp_col].mean().reset_index()
            
            # Original probabilities
            bars1 = plt.bar(orig_decile_precision['orig_decile'] - 0.2, 
                          orig_decile_precision[self.tp_col], 
                          width=0.4, label='Original Probabilities')
            
            # Calibrated probabilities
            bars2 = plt.bar(calib_decile_precision['calib_decile'] + 0.2, 
                          calib_decile_precision[self.tp_col], 
                          width=0.4, label='Calibrated Probabilities')
            
            # Add value labels on top of bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.title('Precision by Probability Decile - Original vs Calibrated')
            plt.legend()
            
        else:
            # Only original probabilities
            bars = plt.bar(orig_decile_precision['orig_decile'], 
                         orig_decile_precision[self.tp_col])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.title('Precision by Probability Decile')
        
        # Add labels
        plt.xlabel('Probability Decile (High to Low)')
        plt.ylabel('Precision')
        
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
        Model precision decay with a continuous power law that gradually declines.
        
        Parameters:
        - a, b: Shape parameters for the initial decay
        - c: Background precision level (primary component)
        - d: Decay rate for the tail (smaller = slower decay)
        """
        # Base power law component for initial decay
        power_decay = a * (k ** -b)
        
        # Gradual exponential decay after k=300
        exp_decay = c * np.exp(-d * np.maximum(0, (k - 300) / 700))
        
        return power_decay + exp_decay
    
    def plot_precision_at_k(self, current_k=300, max_k=1200, resource_constraint=None, 
                         model_type='continuous', figsize=(10, 6),
                         show_thresholds=True, threshold_method='statistical'):
        """
        Plot Precision@k curve as k increases.
        
        Parameters:
        - current_k: Current number of reviews
        - max_k: Maximum k to extrapolate to
        - resource_constraint: Resource constraint to mark on the plot
        - model_type: Type of model to use ('power', 'two_phase', or 'continuous')
        - figsize: Figure size
        - show_thresholds: Whether to show monitoring thresholds (green/yellow/red zones)
        - threshold_method: Method to determine thresholds:
                          'statistical': Based on confidence intervals
                          'percentile': Based on percentiles of observed values
                          'rule_based': Fixed percentage drops from baseline
                          
        Returns:
        - Matplotlib plot
        """
        # Use calibrated probabilities if available
        prob_col = self.calibrated_prob_col if self.calibrated else self.prob_col
        
        # Ensure max_k doesn't exceed data length
        data_length = len(self.data)
        if max_k > data_length:
            print(f"Note: max_k ({max_k}) exceeds data length ({data_length}). " +
                  f"Extrapolation will be used for k > {data_length}.")
        
        # Calculate observed Precision@k at different k values up to current_k
        k_values_observed = np.arange(10, min(current_k, data_length) + 1, 10)  # steps of 10
        precision_observed = []
        
        for k in k_values_observed:
            precision_observed.append(self.data.iloc[:k][self.tp_col].mean())
        
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
            params, _ = curve_fit(model_func, k_values_observed, precision_observed, 
                               p0=p0, bounds=bounds, maxfev=10000)
            
            # Store parameters for later use
            self.model_params = params
            self.model_type = model_type
            
            # Generate predictions for ALL k values up to max_k (important!)
            k_values_all = np.arange(10, max_k + 1, 10)
            precision_predicted = model_func(k_values_all, *params)
            
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
            precision_predicted = model_func(k_values_all, *params)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot observed and predicted values
        plt.scatter(k_values_observed, precision_observed, color='blue', label='Observed Precision')
        
        # Plot FULL prediction line up to max_k
        plt.plot(k_values_all, precision_predicted, 'r-', label='Predicted Precision')
        
        # Calculate threshold values for monitoring if requested
        if show_thresholds:
            # Get the baseline value (precision at current_k)
            baseline_idx = np.argmin(np.abs(k_values_all - current_k))
            baseline_precision = precision_predicted[baseline_idx]
            
            # Determine thresholds based on selected method
            green_threshold, yellow_threshold, red_threshold = self._calculate_monitoring_thresholds(
                baseline_precision, precision_observed, threshold_method)
            
            # Add horizontal lines for thresholds
            plt.axhline(y=green_threshold, color='green', linestyle='-', alpha=0.3)
            plt.axhline(y=yellow_threshold, color='gold', linestyle='-', alpha=0.3)
            plt.axhline(y=red_threshold, color='red', linestyle='-', alpha=0.3)
            
            # Add annotations
            plt.text(max_k*0.95, green_threshold*1.02, f'Green: >{green_threshold:.4f}', 
                    ha='right', va='bottom', color='green')
            plt.text(max_k*0.95, yellow_threshold*1.02, f'Yellow: >{yellow_threshold:.4f}', 
                    ha='right', va='bottom', color='goldenrod')
            plt.text(max_k*0.95, red_threshold*1.02, f'Red: <{yellow_threshold:.4f}', 
                    ha='right', va='bottom', color='red')
            
            # Store threshold values
            self.monitoring_thresholds = {
                'green': green_threshold,
                'yellow': yellow_threshold,
                'red': red_threshold,
                'method': threshold_method
            }
        
        # Add vertical lines
        plt.axvline(x=current_k, color='green', linestyle='--', 
                   label=f'Current k={current_k}')
        
        if resource_constraint:
            plt.axvline(x=resource_constraint, color='orange', linestyle='--', 
                       label=f'Resource Constraint k={resource_constraint}')
        
        # Set x-axis limit to ensure it shows up to max_k
        plt.xlim(0, max_k)
        
        # Add labels and title
        plt.xlabel('k (Number of Reviews)')
        plt.ylabel('Precision@k')
        
        prob_type = "Calibrated" if self.calibrated else "Original"
        plt.title(f'Precision@k Curve (Using {prob_type} Probabilities)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Store extrapolation results
        self._calculate_extrapolation_results(k_values_all, precision_predicted, model_func)
        
        return plt
    
    def _calculate_monitoring_thresholds(self, baseline_precision, observed_precision, method='statistical'):
        """
        Calculate monitoring thresholds for Precision@k.
        
        Parameters:
        - baseline_precision: Baseline precision value (typically at current_k)
        - observed_precision: Array of observed precision values
        - method: Method to determine thresholds
        
        Returns:
        - green_threshold: Minimum precision for green zone
        - yellow_threshold: Minimum precision for yellow zone
        - red_threshold: Minimum precision for red zone
        """
        if method == 'statistical':
            # Use statistical confidence intervals
            # Calculate standard deviation of observed precision
            std_dev = np.std(observed_precision)
            
            # Define thresholds based on standard deviations from baseline
            green_threshold = baseline_precision - 1 * std_dev  # Within 1 std dev
            yellow_threshold = baseline_precision - 2 * std_dev  # Within 2 std dev
            red_threshold = baseline_precision - 3 * std_dev  # Beyond 3 std dev
            
        elif method == 'percentile':
            # Use percentiles of observed precision values
            green_threshold = np.percentile(observed_precision, 25)  # 25th percentile
            yellow_threshold = np.percentile(observed_precision, 10)  # 10th percentile
            red_threshold = np.percentile(observed_precision, 5)  # 5th percentile
            
        elif method == 'rule_based':
            # Use fixed percentage drops from baseline
            green_threshold = baseline_precision * 0.8  # 20% drop
            yellow_threshold = baseline_precision * 0.6  # 40% drop
            red_threshold = baseline_precision * 0.4  # 60% drop
            
        else:
            # Default to rule-based
            print(f"Unknown threshold method '{method}'. Using rule-based method.")
            green_threshold = baseline_precision * 0.8
            yellow_threshold = baseline_precision * 0.6
            red_threshold = baseline_precision * 0.4
        
        # Ensure thresholds are in descending order
        yellow_threshold = min(yellow_threshold, green_threshold * 0.95)
        red_threshold = min(red_threshold, yellow_threshold * 0.95)
        
        # Ensure all thresholds are positive
        green_threshold = max(0.001, green_threshold)
        yellow_threshold = max(0.0005, yellow_threshold)
        red_threshold = max(0.0001, red_threshold)
        
        return green_threshold, yellow_threshold, red_threshold
    
    def _calculate_extrapolation_results(self, k_values, precision_predicted, model_func):
        """
        Calculate and store extrapolation results.
        """
        # Key k values to report
        k_values_key = [100, 300, 500, 800, 1000, 1200]
        
        # Filter to only include k values that exist in our prediction range
        max_k = max(k_values)
        k_values_key = [k for k in k_values_key if k <= max_k]
        
        # Create results dictionary
        results = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'predictions': {}
        }
        
        # Generate model formula string
        if self.model_type == 'power':
            a, b, c = self.model_params
            results['model_formula'] = f"Precision = {a:.4f} * k^(-{b:.4f}) + {c:.4f}"
        elif self.model_type == 'two_phase':
            a, b, c, k_threshold = self.model_params
            results['model_formula'] = (
                f"Precision = {a:.4f} * k^(-{b:.4f}) + {c:.4f} for k < {k_threshold:.1f}\n"
                f"Precision = constant for k >= {k_threshold:.1f}"
            )
        else:  # continuous
            a, b, c, d = self.model_params
            results['model_formula'] = (
                f"Precision = {a:.4f} * k^(-{b:.4f}) + {c:.4f} * exp(-{d:.6f} * (k-300)/700)"
            )
        
        # Calculate predictions for key k values
        for k in k_values_key:
            # Find closest k in our predictions
            closest_idx = np.argmin(np.abs(k_values - k))
            pred_precision = precision_predicted[closest_idx]
            expected_tp = k * pred_precision
            
            results['predictions'][k] = {
                'precision': float(pred_precision),
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
            print("No extrapolation results available. Call plot_precision_at_k() first.")
            return None
        
        return self.extrapolation_results
    
    def print_extrapolation_summary(self):
        """
        Print a summary of the extrapolation results.
        """
        if not self.extrapolation_results:
            print("No extrapolation results available. Call plot_precision_at_k() first.")
            return
        
        results = self.extrapolation_results
        
        print(f"Precision@k Model Parameters ({results['model_type']} model):")
        print(results['model_formula'])
        
        print("\nPredicted Precision at key k values:")
        for k, prediction in results['predictions'].items():
            precision = prediction.get('precision', 0)
            expected_tp = prediction['expected_tp']
            print(f"k={k}: Precision = {precision:.4f}, Expected TP = {expected_tp:.1f}")
        
        # Print monitoring thresholds if available
        if hasattr(self, 'monitoring_thresholds'):
            print("\nModel Monitoring Thresholds:")
            print(f"Method: {self.monitoring_thresholds['method']}")
            print(f"Green zone: > {self.monitoring_thresholds['green']:.4f}")
            print(f"Yellow zone: {self.monitoring_thresholds['yellow']:.4f} to {self.monitoring_thresholds['green']:.4f}")
            print(f"Red zone: < {self.monitoring_thresholds['yellow']:.4f}")


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
                         model_type='continuous', calibrate=True, threshold_method='statistical'):
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
    - threshold_method: Method for determining monitoring thresholds
    
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
    print("Generating Precision by decile plot...")
    analyzer.plot_precision_by_decile().savefig('precision_by_decile.png', dpi=300)
    
    if calibrate:
        print("Generating calibration curve plot...")
        analyzer.plot_calibration_curve().savefig('calibration_curve.png', dpi=300)
    
    print(f"Generating Precision@k curve with {model_type} model...")
    analyzer.plot_precision_at_k(
        current_k=current_k,
        max_k=max_k,
        resource_constraint=resource_constraint,
        model_type=model_type,
        show_thresholds=True,
        threshold_method=threshold_method
    ).savefig('precision_at_k.png', dpi=300)
    
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
        prob_col='probability',
        tp_col='is_true_positive',
        current_k=300,
        max_k=1200,
        resource_constraint=800,
        model_type='continuous',
        calibrate=True,
        threshold_method='statistical'
    )
    
    # Access results programmatically
    results = analyzer.get_extrapolation_results()
    
    # Example: Get expected true positives at resource constraint
    expected_tp_at_constraint = results['predictions'][800]['expected_tp']
    print(f"\nExpected true positives at resource constraint (k=800): {expected_tp_at_constraint:.1f}")