import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the CSV file with probability scores and ground truth
# Assuming format: prob_score, is_true_positive (1 or 0)
# Replace 'your_data.csv' with your actual file path
df = pd.read_csv('your_data.csv')

# Sort by probability score in descending order
df = df.sort_values(by='prob_score', ascending=False).reset_index(drop=True)

# PLOT 1: TPR by Probability Deciles
# Create probability deciles
num_bins = 10  # for deciles, you can change to other granularity
df['decile'] = pd.qcut(df['prob_score'], num_bins, labels=False)

# Calculate TPR for each decile
decile_tpr = df.groupby('decile')['is_true_positive'].mean().reset_index()

# Create plot 1: TPR by probability decile
plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
bars = plt.bar(decile_tpr['decile'], decile_tpr['is_true_positive'])

# Add labels and title
plt.xlabel('Probability Decile (High to Low)')
plt.ylabel('True Positive Rate')
plt.title('TPR by Probability Decile')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom')

# Reverse x-axis to show highest probabilities first
plt.gca().invert_xaxis()

# PLOT 2: TPR Decay with k
# Prepare data for k values
max_k = min(1200, len(df))
current_k = 300
resource_constraint = 800

# Function to model TPR decay
def tpr_decay_model(k, a, b, c):
    """Model TPR decay as k increases (power law with offset)"""
    return a * (k ** -b) + c

# Calculate observed TPR at different k values up to current_k
k_values_observed = np.arange(10, current_k + 1, 10)  # steps of 10
tpr_observed = []

for k in k_values_observed:
    tpr_observed.append(df.iloc[:k]['is_true_positive'].mean())

# Fit the model to observed data
try:
    # Initial parameter guesses
    p0 = [0.5, 0.5, 0.05]
    bounds = ([0, 0, 0], [1, 3, 0.2])
    
    params, _ = curve_fit(tpr_decay_model, k_values_observed, tpr_observed, 
                         p0=p0, bounds=bounds)
    a, b, c = params
    
    # Generate predictions for all k values
    k_values_all = np.arange(100, max_k + 1, 20)
    tpr_predicted = tpr_decay_model(k_values_all, a, b, c)
    
except RuntimeError:
    # Fallback to simpler logarithmic model if curve_fit fails
    from sklearn.linear_model import LinearRegression
    
    log_k = np.log(k_values_observed).reshape(-1, 1)
    model = LinearRegression().fit(log_k, tpr_observed)
    
    k_values_all = np.arange(100, max_k + 1, 20)
    log_k_all = np.log(k_values_all).reshape(-1, 1)
    tpr_predicted = model.predict(log_k_all)

# Create plot 2: TPR decay with k
ax = plt.subplot(1, 2, 2)

# Plot observed and predicted values
plt.scatter(k_values_observed, tpr_observed, color='blue', label='Observed TPR')
plt.plot(k_values_all, tpr_predicted, 'r-', label='Predicted TPR')

# Add vertical line at current k and resource constraint
plt.axvline(x=current_k, color='green', linestyle='--', 
           label=f'Current k={current_k}')
plt.axvline(x=resource_constraint, color='orange', linestyle='--', 
           label=f'Resource Constraint k={resource_constraint}')

# Add labels and title
plt.xlabel('k (Number of Reviews)')
plt.ylabel('True Positive Rate')
plt.title('TPR Decay as k Increases')
plt.legend()

# Adjust layout and show plots
plt.tight_layout()
plt.savefig('tpr_analysis.png', dpi=300)
plt.show()

# Print model parameters for reference
try:
    print(f"TPR Decay Model: TPR = {a:.4f} * k^(-{b:.4f}) + {c:.4f}")
    
    # Calculate and print TPR at key k values
    k_values_key = [100, 300, 500, 800, 1000, 1200]
    print("\nPredicted TPR at key k values:")
    for k in k_values_key:
        pred_tpr = tpr_decay_model(k, a, b, c)
        print(f"k={k}: TPR = {pred_tpr:.4f}")
except NameError:
    # If curve_fit failed, print logarithmic model details
    print("Using logarithmic model for TPR decay")
    print(f"Slope: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")
    
    print("\nPredicted TPR at key k values:")
    k_values_key = [100, 300, 500, 800, 1000, 1200]
    for k in k_values_key:
        log_k = np.log(k).reshape(1, -1)
        pred_tpr = model.predict(log_k)[0]
        print(f"k={k}: TPR = {pred_tpr:.4f}")