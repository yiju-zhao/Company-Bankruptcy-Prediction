# Company-Bankruptcy-Prediction

## Table of Content
1. Project Overview
    - Business questions
    - Literature review
2. Data Description
3. Exploratory Data Analysis
    - Summary Statistics
    - Cleaning
    - Handling Missing Data
    - Outlier Detection
    - Visualization
    - Feature correlation
4. Feature Engineering and Selection
    - Feature Transformation
    - Feature Engineering
    - Feature Selection
5. Model Development
    - Baseline model
    - Rationale for model selection
    - Hyper-parameter tuning
    - Training
    - Evaluation and Model Selection
6. Final Model Explanation
    - Feature Importance
    - SHAP
    - Business Insights and Implications
7. Repository Structure
8. Getting Start
9. Code Sample
10. Usage
11. Results
12. Discussion
13. References

## Project Overview
Corporate bankruptcy can significantly impact stakeholders, employees, and broader financial markets. Accurately predicting the risk of bankruptcy helps lenders, investors, and policymakers make better decisions regarding credit risk and investment allocations.
- **Goal**: Predict the hazard rate (risk) of bankruptcy over a specified time horizon using financial and operational features.
- **Approach**: Experiment with the Cox Proportional Hazards model (a semi-parametric survival model) and PH, and logistic regression to see which one explains the pattern of bankruptcy more. The “time” aspect is crucial since not all firms will declare bankruptcy, and those that don’t can be considered right-censored data.
