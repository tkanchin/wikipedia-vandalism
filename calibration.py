import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression

# Load the CSV file with probability scores and ground truth
# Assuming format: prob_score, is_true_positive (1 or 0)
# Replace 'your_data.csv' with your actual file path
df = pd.read_csv('your_data.csv')

# Apply isotonic regression to calibrate probabilities
print("Calibrating probabilities using isotonic regression...")
ir = IsotonicRegression(out_of_bounds='clip')
df['calibrated_prob'] = ir.fit_transform(df['prob_score'], df['is_true_positive'])

# Sort by calibrated probability score in descending order
df = df.sort_values(by='calibrated_prob', ascending=False).reset_index(drop=True)

# PLOT 1: TPR by Probability Deciles
# Create probability deciles for both original and calibrated probabilities
num_bins = 10  # for deciles, you can change to other granularity
df['orig_decile'] = pd.qcut(df['prob_score'], num_bins, labels=False)
df['calib_decile'] = pd.qcut(df['calibrated_prob'], num_bins, labels=False)

# Calculate TPR for each decile (both original and calibrated)
orig_decile_tpr = df.groupby('orig_decile')['is_true_positive'].mean().reset_index()
calib_decile_tpr = df.groupby('calib_decile')['is_true_positive'].mean().reset_index()

# Create plot 1: TPR by probability decile (comparing original vs calibrated)
plt.figure(figsize=(15, 10))
ax = plt.subplot(2, 2, 1)

# Original probabilities
bars1 = plt.bar(orig_decile_tpr['orig_decile'] - 0.2, 
               orig_decile_tpr['is_true_positive'], 
               width=0.4, label='Original Probabilities')

# Calibrated probabilities
bars2 = plt.bar(calib_decile_tpr['calib_decile'] + 0.2, 
               calib_decile_tpr['is_true_positive'], 
               width=0.4, label='Calibrated Probabilities')

# Add labels and title
plt.xlabel('Probability Decile (High to Low)')
plt.ylabel('True Positive Rate')
plt.title('TPR by Probability Decile - Original vs Calibrated')
plt.legend()

# Add value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Reverse x-axis to show highest probabilities first
plt.gca().invert_xaxis()

# PLOT 2: Probability Calibration Curve
ax = plt.subplot(2, 2, 2)

# Sort values by original probability for plotting
calib_df = df[['prob_score', 'calibrated_prob']].sort_values('prob_score')
plt.plot(calib_df['prob_score'], calib_df['calibrated_prob'], 'b-', label='Calibration function')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

# Add labels and title
plt.xlabel('Original Probability')
plt.ylabel('Calibrated Probability')
plt.title('Probability Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# PLOT 3: TPR Decay with k (using calibrated probabilities)
# Prepare data for k values
max_k = min(1200, len(df))
current_k = 300
resource_constraint = 800

# Calculate observed TPR at different k values up to current_k
k_values_observed = np.arange(10, current_k + 1, 10)  # steps of 10
tpr_observed = []

for k in k_values_observed:
    tpr_observed.append(df.iloc[:k]['is_true_positive'].mean())

# Function to model TPR decay
def tpr_decay_model(k, a, b, c, k_threshold):
    """Two-phase model with power law decay until k_threshold, then constant"""
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

# Fit the model to observed data
try:
    # Initial parameter guesses
    p0 = [0.5, 0.5, 0.05, 250]
    bounds = ([0, 0, 0, 100], [1, 3, 0.2, 400])
    
    params, _ = curve_fit(tpr_decay_model, k_values_observed, tpr_observed, 
                         p0=p0, bounds=bounds, maxfev=10000)
    a, b, c, k_threshold = params
    
    # Generate predictions for all k values
    k_values_all = np.arange(10, max_k + 1, 10)
    tpr_predicted = tpr_decay_model(k_values_all, a, b, c, k_threshold)
    
except RuntimeError:
    # Fallback to simpler logarithmic model if curve_fit fails
    from sklearn.linear_model import LinearRegression
    
    # Try a piecewise linear model on log scale
    log_k = np.log(k_values_observed).reshape(-1, 1)
    
    # Find elbow point in the data (where decay stabilizes)
    from sklearn.cluster import KMeans
    
    # Use 2 clusters to find the elbow
    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        np.array(list(zip(k_values_observed, tpr_observed))))
    
    # Sort cluster centers by k value
    centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
    k_threshold = (centers[0][0] + centers[1][0]) / 2
    
    # Fit two separate linear models
    early_indices = k_values_observed < k_threshold
    late_indices = k_values_observed >= k_threshold
    
    early_model = None
    if np.any(early_indices):
        early_model = LinearRegression().fit(
            log_k[early_indices], 
            np.array(tpr_observed)[early_indices])
    
    late_model = None
    if np.any(late_indices):
        late_model = LinearRegression().fit(
            log_k[late_indices], 
            np.array(tpr_observed)[late_indices])
    
    # Generate predictions for all k values
    k_values_all = np.arange(10, max_k + 1, 10)
    log_k_all = np.log(k_values_all).reshape(-1, 1)
    tpr_predicted = np.zeros(len(k_values_all))
    
    for i, k in enumerate(k_values_all):
        if k < k_threshold and early_model is not None:
            tpr_predicted[i] = early_model.predict(np.log(k).reshape(1, -1))[0]
        elif late_model is not None:
            tpr_predicted[i] = late_model.predict(np.log(k).reshape(1, -1))[0]
        else:
            # Use the last observed TPR as fallback
            tpr_predicted[i] = tpr_observed[-1]
    
    # Ensure predictions are non-negative and decreasing
    tpr_predicted = np.maximum(tpr_predicted, 0)
    for i in range(1, len(tpr_predicted)):
        tpr_predicted[i] = min(tpr_predicted[i], tpr_predicted[i-1])

# Create plot 3: TPR decay with k
ax = plt.subplot(2, 1, 2)

# Plot observed and predicted values
plt.scatter(k_values_observed, tpr_observed, color='blue', label='Observed TPR')
plt.plot(k_values_all, tpr_predicted, 'r-', label='Predicted TPR')

# Add vertical lines at current k and resource constraint
plt.axvline(x=current_k, color='green', linestyle='--', 
           label=f'Current k={current_k}')
plt.axvline(x=resource_constraint, color='orange', linestyle='--', 
           label=f'Resource Constraint k={resource_constraint}')

# Add labels and title
plt.xlabel('k (Number of Reviews)')
plt.ylabel('True Positive Rate')
plt.title('TPR Decay as k Increases (Using Calibrated Probabilities)')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust layout and show plots
plt.tight_layout()
plt.savefig('tpr_analysis_with_calibration.png', dpi=300)
plt.show()

# Print model parameters and predictions
try:
    print(f"TPR Decay Model Parameters:")
    print(f"a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, k_threshold = {k_threshold:.1f}")
    print(f"TPR = {a:.4f} * k^(-{b:.4f}) + {c:.4f} for k < {k_threshold:.1f}")
    print(f"TPR = constant for k >= {k_threshold:.1f}")
except NameError:
    # If curve_fit failed
    print("Using piecewise linear model for TPR decay")
    print(f"Threshold between phases: k = {k_threshold:.1f}")

# Calculate and print TPR at key k values
k_values_key = [100, 300, 500, 800, 1000, 1200]
print("\nPredicted TPR at key k values:")
for k_idx, k in enumerate(k_values_key):
    # Find closest k in our predictions
    closest_idx = np.argmin(np.abs(k_values_all - k))
    pred_tpr = tpr_predicted[closest_idx]
    print(f"k={k}: TPR = {pred_tpr:.4f}")

# Compare original vs. calibrated probabilities
print("\nCalibration Effect Summary:")
print(f"Mean original probability: {df['prob_score'].mean():.4f}")
print(f"Mean calibrated probability: {df['calibrated_prob'].mean():.4f}")

# Check monotonicity improvements
orig_monotonic = (orig_decile_tpr['is_true_positive'].diff().dropna() <= 0).all()
calib_monotonic = (calib_decile_tpr['is_true_positive'].diff().dropna() <= 0).all()
print(f"Original probabilities are monotonic: {orig_monotonic}")
print(f"Calibrated probabilities are monotonic: {calib_monotonic}")