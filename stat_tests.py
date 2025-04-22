import numpy as np
from scipy import stats

def binomial_confidence_interval(successes, trials, confidence=0.95, method='wilson'):
    """
    Calculate confidence intervals for binomial proportion with multiple methods.
    
    Parameters:
    - successes: number of successful events (true positives)
    - trials: number of trials (total predictions reviewed)
    - confidence: confidence level (default: 0.95 for 95% CI)
    - method: method to use for CI calculation:
              'wilson' (default): Wilson score interval (recommended for most cases)
              'wald': Simple approximation (only for large samples)
              'agresti-coull': Adjusted Wald interval (good general method)
              'clopper-pearson': Exact interval (conservative, good for very rare events)
    
    Returns:
    - lower_bound: lower bound of confidence interval
    - upper_bound: upper bound of confidence interval
    """
    # Input validation
    if successes < 0 or trials <= 0 or successes > trials:
        raise ValueError("Invalid input: successes must be between 0 and trials, and trials must be positive")
    
    # Check if we have a rare event scenario (warning threshold)
    rare_event = (successes < 5) or ((trials - successes) < 5)
    
    # Handle edge case of zero successes or 100% success
    edge_case = (successes == 0) or (successes == trials)
    
    # Alpha value (1 - confidence level)
    alpha = 1 - confidence
    
    # Z-score for the given confidence level
    z = stats.norm.ppf(1 - alpha/2)
    
    # Sample proportion
    p = successes / trials
    
    # Select method based on input or automatically choose based on data characteristics
    if method == 'auto':
        if edge_case:
            method = 'clopper-pearson'  # Most reliable for edge cases
        elif rare_event:
            method = 'wilson'  # Good for rare events
        elif trials >= 40:
            method = 'agresti-coull'  # Good for larger samples
        else:
            method = 'wilson'  # Good general purpose method
    
    # Calculate CI using Wilson score method (recommended for most cases)
    if method == 'wilson':
        # Wilson score interval calculation
        denominator = 1 + z**2/trials
        center = (p + z**2/(2*trials))/denominator
        halfwidth = z * np.sqrt(p*(1-p)/trials + z**2/(4*trials**2))/denominator
        
        lower_bound = max(0, center - halfwidth)
        upper_bound = min(1, center + halfwidth)
        
        # Warning for very small samples
        if trials < 10:
            print("Warning: Sample size < 10. Wilson interval may not be optimal.")
    
    # Simple Wald interval (only recommended for large samples)
    elif method == 'wald':
        # This is the simple textbook method, but it has poor properties
        # Only use when: trials >= 30, 5 <= successes <= trials-5
        
        if rare_event:
            print("Warning: Wald interval is not recommended for rare events. Consider using Wilson or Clopper-Pearson.")
        
        halfwidth = z * np.sqrt(p * (1 - p) / trials)
        lower_bound = max(0, p - halfwidth)
        upper_bound = min(1, p + halfwidth)
    
    # Agresti-Coull interval (adjusted Wald interval)
    elif method == 'agresti-coull':
        # Agresti-Coull uses adjusted counts to improve coverage
        n_tilde = trials + z**2
        p_tilde = (successes + z**2/2) / n_tilde
        
        halfwidth = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        lower_bound = max(0, p_tilde - halfwidth)
        upper_bound = min(1, p_tilde + halfwidth)
    
    # Clopper-Pearson (exact) interval
    elif method == 'clopper-pearson':
        # This method works well for edge cases (0 successes or 100% success)
        # It is based on the binomial CDF and is more conservative
        
        # Handle edge cases explicitly
        if successes == 0:
            lower_bound = 0
            upper_bound = 1 - (alpha/2)**(1/trials)
        elif successes == trials:
            lower_bound = (alpha/2)**(1/trials)
            upper_bound = 1
        else:
            lower_bound = stats.beta.ppf(alpha/2, successes, trials - successes + 1)
            upper_bound = stats.beta.ppf(1 - alpha/2, successes + 1, trials - successes)
    
    else:
        raise ValueError("Unknown method. Choose from 'wilson', 'wald', 'agresti-coull', 'clopper-pearson', or 'auto'")
    
    # Additional warnings and information
    if rare_event and method != 'clopper-pearson' and method != 'wilson':
        print(f"Notice: Dealing with rare events (successes={successes}). Wilson or Clopper-Pearson methods may be more appropriate.")
    
    if confidence > 0.99 and trials < 100:
        print(f"High confidence level ({confidence}) with relatively small sample size ({trials}). Interval may be wide.")
    
    return lower_bound, upper_bound


def proportion_test(count1, n1, count2, n2, correction=True):
    """
    Statistical test comparing two proportions (Z-test).
    
    Parameters:
    - count1: number of successes in first sample (e.g., true positives in model)
    - n1: size of first sample (e.g., number of reviewed items by model)
    - count2: number of successes in second sample (e.g., true positives in baseline)
    - n2: size of second sample (e.g., number of reviewed items by baseline)
    - correction: whether to apply continuity correction (default: True)
                 Recommended for small counts (<5) to improve accuracy
    
    Returns:
    - z_stat: z-statistic
    - p_value: p-value for two-sided test
    - summary: Dictionary with test details and recommendations
    """
    # Calculate proportions
    p1 = count1 / n1
    p2 = count2 / n2
    diff = p1 - p2
    
    # Check for small counts
    small_counts = (count1 < 5) or (count2 < 5) or ((n1 - count1) < 5) or ((n2 - count2) < 5)
    
    # Calculate pooled proportion (used for standard error)
    p_pooled = (count1 + count2) / (n1 + n2)
    
    # Apply continuity correction if requested
    if correction and small_counts:
        # Continuity correction factor
        # This makes the test more conservative for small samples
        ccf = 0.5 * (1/n1 + 1/n2)
        
        # Adjust the difference with the correction factor
        # Only apply correction if it doesn't change the sign of the difference
        if abs(diff) > ccf:
            if diff > 0:
                diff = diff - ccf
            else:
                diff = diff + ccf
    
    # Calculate standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Handle division by zero (can happen with extreme proportions)
    if se == 0:
        if p1 == p2:
            # If proportions are exactly equal, no difference
            z_stat = 0
            p_value = 1.0
        else:
            # If proportions differ but SE is 0, this is a significant difference
            z_stat = np.inf if diff > 0 else -np.inf
            p_value = 0.0
    else:
        # Calculate z-statistic
        z_stat = diff / se
        
        # Calculate two-sided p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Create summary with additional information
    summary = {
        "proportions": {
            "sample1": p1,
            "sample2": p2,
            "difference": diff,
            "pooled": p_pooled
        },
        "test_details": {
            "z_statistic": z_stat,
            "p_value": p_value,
            "standard_error": se,
            "continuity_correction_applied": correction and small_counts
        },
        "interpretation": {
            "significant": p_value < 0.05,
            "confidence": "high" if p_value < 0.01 else "moderate" if p_value < 0.05 else "low"
        },
        "warnings": []
    }
    
    # Add relevant warnings
    if small_counts:
        summary["warnings"].append("Small counts detected. Results should be interpreted with caution.")
        if not correction:
            summary["warnings"].append("Continuity correction recommended for small counts but was not applied.")
    
    if min(n1, n2) < 30:
        summary["warnings"].append("Small sample size. Consider using Fisher's exact test for more accurate results.")
    
    if p1 == 0 or p1 == 1 or p2 == 0 or p2 == 1:
        summary["warnings"].append("Extreme proportions (0 or 1) detected. Z-test may not be appropriate.")
    
    return z_stat, p_value, summary


def determine_performance_zone(current_precision, lower_95, lower_997):
    """
    Determine the performance zone based on confidence intervals.
    
    Parameters:
    - current_precision: current precision value
    - lower_95: lower bound of 95% confidence interval
    - lower_997: lower bound of 99.7% confidence interval
    
    Returns:
    - zone: string indicating performance zone ('GREEN', 'YELLOW', or 'RED')
    - action: recommended action based on zone
    """
    if current_precision >= lower_95:
        return "GREEN", "Continue normal monitoring"
    elif current_precision >= lower_997:
        return "YELLOW", "Investigate potential data drift or concept drift"
    else:
        return "RED", "Detailed investigation needed"


# Example usage for anomaly detection monitoring
def run_statistical_analysis():
    """
    Run the statistical analysis for the anomaly detection model
    and compare it with the baseline system.
    """
    # 1. Calculate confidence intervals for current performance
    baseline_tp = 6
    baseline_reviewed = 800
    model_tp = 35
    model_reviewed = 300
    
    lower_95, upper_95 = binomial_confidence_interval(model_tp, model_reviewed)
    lower_997, upper_997 = binomial_confidence_interval(model_tp, model_reviewed, confidence=0.997)
    
    print(f"Current Precision: {model_tp/model_reviewed:.3f}")
    print(f"95% Confidence Interval: [{lower_95:.3f}, {upper_95:.3f}]")
    print(f"99.7% Confidence Interval: [{lower_997:.3f}, {upper_997:.3f}]")
    
    # 2. Statistical significance test
    z_stat, p_value, summary = proportion_test(model_tp, model_reviewed, baseline_tp, baseline_reviewed)
    print(f"Z-statistic: {z_stat:.2f}, p-value: {p_value:.10f}")
    
    # Print any warnings from the test
    if summary["warnings"]:
        print("\nTest Warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    
    # 3. Performance zone determination
    zone, action = determine_performance_zone(model_tp/model_reviewed, lower_95, lower_997)
    print(f"\nPerformance Zone: {zone}")
    print(f"Recommended Action: {action}")
    
    # 4. Additional information
    improvement_factor = (model_tp/model_reviewed) / (baseline_tp/baseline_reviewed)
    print(f"\nImprovement factor over baseline: {improvement_factor:.1f}x")
    print(f"Statistical significance: {'High' if p_value < 0.001 else 'Moderate' if p_value < 0.05 else 'Low'}")


if __name__ == "__main__":
    run_statistical_analysis()