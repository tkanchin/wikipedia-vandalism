# Statistical Monitoring Playbook for Text Anomaly Detection Models

## Introduction

This playbook provides a focused framework for monitoring text-based anomaly detection models with extremely low event rates where the total number of true positives is unknown.

**Performance Summary**

Your ML model has shown significant improvement over the baseline rule-based system:
- Baseline: 6 true positives in 800 reviews (0.75% precision)
- ML Model: 35 true positives in 300 reviews (11.7% precision)
- 583% increase in true positive detection with 62.5% reduction in review effort

## 1. Statistical Monitoring Framework

**Binomial Test for Performance Monitoring**

The binomial test provides statistical confidence intervals around your observed precision rate. For your current model (35 TP in 300 reviews):
- 95% Confidence Interval: Lower bound = 8.2%, Upper bound = 15.2% 
- 99.7% Confidence Interval: Lower bound = 6.0%, Upper bound = 17.4%

References: [Wilson Score Interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)

**Control Limits & Decision Zones**

These statistical limits provide objective evaluation criteria for model performance:

- **Green Zone**: Performance within expected variation (within 95% CI)
  - Current model: >8.2% precision
  - Action: Continue normal monitoring

- **Yellow Zone**: Performance degradation requiring investigation (below 95% CI but above 99.7% CI)
  - Current model: 6.0%-8.2% precision  
  - Action: Investigate potential data drift or concept drift

- **Red Zone**: Significant decline requiring immediate action (below 99.7% CI)
  - Current model: <6.0% precision
  - Action: Consult data science team

**Proportion Test for Model Comparison**

To validate improvement over baseline, calculate the Z-statistic comparing current model to baseline:
- For current data (35/300 vs 6/800): Z = 8.21, p < 0.0001
- Interpretation: Improvement is highly statistically significant

## 2. Optimal Threshold Determination

**Expected Value Optimization Framework**

This approach determines the optimal number of cases to review based on the value of true positives and the cost of reviews:

1. Define Value Parameters:
   - V = Value obtained from each true positive case
   - C = Cost of reviewing each case (including false positives)

2. Calculate Expected Net Value at Different Thresholds:
   ```
   For threshold k (e.g., top 100, 200, 300, 400...):
   - Precision(k) = Number of TP in top k / k
   - Expected_TP(k) = k × Precision(k)
   - Expected_Value(k) = (Expected_TP(k) × V) - (k × C)
   ```

3. Find the Optimal Threshold:
   - Select k that maximizes Expected_Value(k)
   - This represents the point where marginal value equals marginal cost

4. Consider Resource Constraints:
   - If optimal k exceeds available resources, use the resource limit as the threshold
   - If resources allow for higher k than calculated optimum, stay with the calculated optimum

**Worked Example**

Assume the following:
- Value of each true positive (V) = $10,000
- Cost of each review (C) = $100
- Historical precision at different thresholds:
  - Top 100: 20% precision (20 TP)
  - Top 200: 17.5% precision (35 TP)
  - Top 300: 11.7% precision (35 TP) 
  - Top 400: 9.5% precision (38 TP)
  - Top 500: 8.0% precision (40 TP)

Expected value calculations:
- At k=100: (20 × $10,000) - (100 × $100) = $190,000
- At k=200: (35 × $10,000) - (200 × $100) = $330,000
- At k=300: (35 × $10,000) - (300 × $100) = $320,000
- At k=400: (38 × $10,000) - (400 × $100) = $340,000
- At k=500: (40 × $10,000) - (500 × $100) = $350,000

In this example, the optimal threshold would be k=500, as it maximizes expected value. However, if resources only allow for reviewing 300 cases, then k=300 would be used.

**Optimizing with Unknown Precision for Text Anomaly Detection**

In your text anomaly detection scenario with extremely low event rates, estimating precision at different thresholds requires a carefully designed sampling approach:

1. Initial Threshold Evaluation:
   - Start with your current threshold of 300 cases (resource limit)
   - Measure actual precision (currently 35/300 = 11.7%)
   - Compare to baseline system (6/800 = 0.75%)

2. Precision Curve Estimation Experiment:
   
   Design a sampling experiment as follows:
   - Divide scored population into score bands:
     * Band A: Top 100 predictions
     * Band B: Predictions 101-300
     * Band C: Predictions 301-500
     * Band D: Predictions 501-1000
   
   - Sample randomly within each band:
     * 50 cases from Band A (expecting highest precision)
     * 50 cases from Band B
     * 30 cases from Band C
     * 20 cases from Band D

   - After review, calculate precision in each band:
     * Band A: e.g., 10/50 = 20% precision
     * Band B: e.g., 7/50 = 14% precision
     * Band C: e.g., 2/30 = 6.7% precision
     * Band D: e.g., 0/20 = 0% precision

3. Extrapolate Full Precision Curve:
   - Based on band results, estimate:
     * Top 100: 20% precision (20 true positives)
     * Top 300: 17.3% precision (52 true positives)
     * Top 500: 13.8% precision (69 true positives)
     * Top 1000: 6.9% precision (69 true positives)

4. Value-Based Threshold Calculation:
   - Assuming each true positive is worth $5,000 and review cost is $50 per case:
     * Top 100: (20 × $5,000) - (100 × $50) = $95,000
     * Top 300: (52 × $5,000) - (300 × $50) = $245,000
     * Top 500: (69 × $5,000) - (500 × $50) = $320,000
     * Top 1000: (69 × $5,000) - (1000 × $50) = $295,000
   
   - Result: The optimal threshold appears to be 500, but resource constraints limit you to 300
   - This analysis provides justification for potentially increasing resource allocation to review 500 cases, which would yield higher expected value

5. Ongoing Refinement:
   - Monitor precision at current threshold (300) over time
   - If resources allow, periodically sample beyond threshold to detect changes in precision curve
   - Re-evaluate optimal threshold quarterly based on updated precision data

## 3. Monthly Monitoring Protocol

**Collect Results**
Record number of true positives identified in top 300 predictions.

**Calculate Precision**
Current month precision = TP / 300

**Apply Statistical Tests**
- Calculate 95% and 99.7% confidence intervals
- Determine performance zone (green, yellow, red)

**Take Action Based on Zone**
- Green: Continue normal operations
- Yellow: Investigate and monitor closely
- Red: Detailed investigation needed

## 4. Defending the Approach

**Key Statistical Arguments**

1. Quantifiable Improvement with Statistical Significance:
   - Precision increased from 0.75% (baseline) to 11.7% (ML model)
   - Z-statistic: 8.21, p-value < 0.0001
   - 95% Confidence Interval for current precision: [8.2%, 15.2%]
   - Probability that improvement occurred by chance: less than 1 in 10,000

2. Optimal Resource Allocation:
   - Expected value analysis shows reviewing 300 cases yields optimal return given constraints
   - Each additional true positive detected prevents potentially significant harm/loss
   - Current threshold represents optimal balance between detection capability and resource utilization

3. Scientifically Sound Monitoring Framework:
   - Statistical control limits derived from binomial distribution theory
   - Objective performance zones with clear action triggers
   - Methodology aligns with established statistical process control practices

**Addressing Stakeholder Questions**

"How do we know the model is actually better than the baseline rules?"
The improvement is statistically significant with p<0.0001. We're finding nearly 6 times more anomalies while reviewing less than half the cases. The probability this improvement occurred by random chance is less than 0.01%.

"Why review exactly 300 cases? Why not more or fewer?"
Our expected value analysis shows that reviewing 300 cases balances the value of finding true anomalies against our resource constraints. We've calculated that each true positive is worth approximately $X, while each review costs $Y, making 300 the optimal number given our current model performance.

"How will we know if the model starts performing worse?"
We've established statistical control limits based on the binomial distribution. If performance drops below our 95% confidence interval (8.2%), we'll investigate potential causes. If it drops below our 99.7% confidence interval (6.0%), we'll take immediate action.

"Can we be confident in the model without knowing the true recall?"
While we can't measure absolute recall, our primary goal is to maximize the number of anomalies found within our resource constraints. The current model identifies 35 true positives versus 6 with the baseline approach, representing a clear and statistically significant improvement in effectiveness.

"What if we suspect the nature of anomalies is changing over time?"
Our monthly monitoring protocol includes precision tracking and periodic sampling beyond our threshold to detect potential shifts in the precision curve. This will help us identify changes in anomaly patterns and adapt accordingly.