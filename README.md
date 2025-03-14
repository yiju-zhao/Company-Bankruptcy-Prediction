# Company-Bankruptcy-Prediction
This project aims to predict the probability of corporate bankruptcy by analyzing financial indicators using machine learning and survival analysis techniques. By comparing different modeling approaches, we seek to derive actionable business insights regarding high-risk versus low-risk firms.

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
# Bankruptcy Prediction Model


---

## Literature Review

*Include here a summary of relevant academic papers, industry reports, and previous work in bankruptcy prediction, survival analysis, and financial risk modeling.*

---

## Business Questions

- **Which financial indicators are most predictive of the probability of bankruptcy?**
- **What business insights can be derived from the final model about high-risk vs. low-risk firms?**
- **What level of accuracy can we achieve in predicting the probability of bankruptcy?**

---

## Model Explanation

The project experiments with multiple modeling approaches including Logistic Regression and the Cox Proportional Hazards model. The Cox model is particularly useful as it handles time-to-event data and accounts for right-censored observations, while Logistic Regression provides a straightforward classification approach.

---

## Getting Started

### Prerequisites

- **Python 3.8+**: Ensure you have Python 3.8 or later installed.
- **Required Packages**: Install the necessary libraries (e.g., `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `lifelines`, etc.).  
  See `requirements.txt` for a complete list of dependencies.

### Dataset

#### Data Description

Two bankruptcy datasets were considered:

1. **Taiwan Bankruptcy Dataset**  
   - **Source**: [Kaggle: Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)  
   - **Details**: Contains 6,819 observations with approximately 220 bankruptcy events.

2. **U.S. Bankruptcy Dataset (Selected)**  
   - **Source**: [Kaggle: American Companies Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset)  
   - **Details**: Contains approximately 78,682 observations, offering a much larger and diverse sample.

#### Justification for Dataset Selection

We chose the U.S. dataset over the Taiwan dataset because:

- **Geographic and Economic Context**:  
  The U.S. market operates under the federal Bankruptcy Code (with clear distinctions such as Chapter 7 vs. Chapter 11), resulting in more uniform definitions and timestamps for bankruptcy events.
  
- **Sample Size and Time Period**:  
  The U.S. dataset is significantly larger, providing 78,682 observations versus 6,819 in the Taiwan dataset. A larger sample offers more robust training data.
  
- **Feature Differences**:  
  U.S. companies adhere to standardized reporting requirements (SEC, GAAP), ensuring consistency in financial ratios, which enhances comparability and model reliability.
  
- **Implication for ML**:  
  A broader, higher-quality dataset leads to more reliable model performance and more transferable insights across different financial contexts.

---

## Exploratory Data Analysis

- **Summary Statistics and Visualization**:  
  Visualize the distribution and relationships of key variables using histograms, boxplots, and scatter plots.
  
- **Key Objectives**:  
  Understand feature distributions, detect missing values, and examine correlations among financial indicators.

---

## Feature Engineering and Selection

- **Feature Transformation**:  
  Apply log-transformation to heavily skewed variables, and standardize/normalize features to improve model performance.
  
- **Feature Engineering**:  
  Create new variables (e.g., ratios such as debt-to-assets, operating cash flow measures) from existing data.
  
- **Feature Selection**:  
  Utilize techniques (e.g., Lasso, stepwise selection, domain expertise) to determine the most predictive features and justify which ones are ultimately included or excluded.

---

## Model Development

- **Hyper-Parameter Tuning**:  
  Use strategies such as grid search, random search, or Bayesian optimization to fin
