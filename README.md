# Company-Bankruptcy-Prediction

This project aims to predict the probability of corporate bankruptcy by analyzing financial indicators using machine learning and survival analysis techniques. By comparing different modeling approaches, we seek to derive actionable business insights regarding high-risk versus low-risk firms.

## Table of Contents
1. [Project Overview](#project-overview) - Business Questions - Literature Review  
2. [Data Description](#data-description)  
3. [Exploratory Data Analysis](#exploratory-data-analysis) - Summary Statistics - Data Cleaning - Handling Missing Data - Outlier Detection - Visualization - Feature Correlation  
4. [Feature Engineering and Selection](#feature-engineering-and-selection) - Feature Transformation - Feature Engineering - Feature Selection  
5. [Model Development](#model-development) - Baseline Model - Model Selection Rationale - Hyper-Parameter Tuning - Training - Evaluation and Model Selection  
6. [Final Model Explanation](#final-model-explanation) - Feature Importance - SHAP Analysis - Business Insights and Implications  
7. [Repository Structure](#repository-structure)  
8. [Getting Started](#getting-started)  
9. [Code Sample](#code-sample)  
10. [Usage](#usage)  
11. [Results](#results)  
12. [Discussion](#discussion)  
13. [References](#references)

---

## Project Overview

Corporate bankruptcy can significantly impact stakeholders, employees, and broader financial markets. Accurately predicting the risk of bankruptcy helps lenders, investors, and policymakers make better decisions regarding credit risk and investment allocations.  
- **Goal**: Predict the hazard rate (risk) of bankruptcy over a specified time horizon using financial and operational features.  
- **Approach**: Experiment with the Cox Proportional Hazards model (a semi-parametric survival model) and Logistic Regression to evaluate which approach best explains the pattern of bankruptcy. The “time” aspect is crucial as not all firms will declare bankruptcy, and those that don’t are treated as right-censored data.

---

## Literature Review

*This section summarizes relevant academic papers, industry reports, and previous work in bankruptcy prediction, survival analysis, and financial risk modeling. Future updates will include detailed summaries of key references and findings that underpin the methodologies used in this project.*

---

## Business Questions

- **Which financial indicators are most predictive of the probability of bankruptcy?**
- **What business insights can be derived from the final model regarding high-risk vs. low-risk firms?**
- **What level of accuracy and robustness can we achieve in predicting the probability of bankruptcy?**

---

## Data Description

### Datasets

Two bankruptcy datasets were considered:

1. **Taiwan Bankruptcy Dataset**  
   - **Source**: [Kaggle: Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)  
   - **Details**: Contains 6,819 observations with approximately 220 bankruptcy events.

2. **U.S. Bankruptcy Dataset (Selected)**  
   - **Source**: [Kaggle: American Companies Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset)  
   - **Details**: Contains approximately 78,682 observations, offering a larger and more diverse sample.

### Justification for Dataset Selection

- **Geographic and Economic Context**: The U.S. market follows a federal Bankruptcy Code (e.g., Chapter 7 vs. Chapter 11), providing more uniform definitions and timestamps.
- **Sample Size and Time Period**: The U.S. dataset offers a robust sample size for training and evaluation.
- **Standardization**: U.S. companies adhere to standardized reporting (SEC, GAAP), ensuring consistency in financial ratios and improved model reliability.
- **Implication for ML**: A larger, high-quality dataset enhances model performance and the transferability of insights across different financial contexts.

---

## Exploratory Data Analysis

- **Summary Statistics and Visualization**: Examine distributions, detect outliers, and understand relationships among variables using histograms, boxplots, and scatter plots.
- **Data Cleaning**: Address inconsistencies, remove duplicates, and rectify data entry errors.
- **Handling Missing Data**: Apply appropriate imputation methods or remove missing entries where necessary.
- **Outlier Detection**: Identify and assess the impact of outliers on model performance.
- **Feature Correlation**: Analyze correlation matrices to identify multicollinearity and inform feature selection.

---

## Feature Engineering and Selection

- **Feature Transformation**: Apply log-transformations to skewed features and scale variables to standardize ranges.
- **Feature Engineering**: Derive new variables, such as financial ratios (e.g., debt-to-assets, liquidity ratios), from existing data.
- **Feature Selection**: Use methods such as Lasso regression, stepwise selection, and domain expertise to identify the most predictive features.

---

## Model Development

- **Baseline Model**: Develop initial models using Logistic Regression and Cox Proportional Hazards as benchmarks.
- **Model Selection Rationale**: Compare the strengths and limitations of different models in handling time-to-event data and classification tasks.
- **Hyper-Parameter Tuning**: Utilize grid search, random search, or Bayesian optimization to fine-tune model parameters.
- **Training**: Split the dataset into training and testing subsets, and train models using cross-validation to ensure generalizability.
- **Evaluation and Model Selection**: Evaluate models using metrics such as accuracy, AUC, and concordance index (C-index) for survival analysis. Select the model that balances performance and interpretability.

---

## Final Model Explanation

- **Feature Importance**: Analyze the contribution of each feature to the final model’s predictions using coefficients and importance scores.
- **SHAP Analysis**: Use SHAP (SHapley Additive exPlanations) to visualize and explain individual predictions, highlighting key factors influencing bankruptcy risk.
- **Business Insights and Implications**: Translate model findings into actionable insights for stakeholders, including recommendations for risk mitigation and investment strategies.

---

## Repository Structure

├── data │ ├── raw │ └── processed ├── notebooks │ └── ML_FinalProject.ipynb ├── src │ ├── data_preprocessing.py │ ├── feature_engineering.py │ └── modeling.py ├── results │ ├── figures │ └── metrics ├── requirements.txt └── README.md

- **data**: Contains raw and processed datasets.
- **notebooks**: Jupyter notebooks used for exploration and model development.
- **src**: Source code for data preprocessing, feature engineering, and model building.
- **results**: Generated figures, metrics, and output from experiments.
- **requirements.txt**: List of Python packages and dependencies.

---

## Getting Started

### Prerequisites

- **Python 3.8+**: Ensure you have Python 3.8 or later installed.
- **Required Packages**: Install the necessary libraries:
  ```bash
  pip install -r requirements.txt

- **Libraries include pandas, numpy, matplotlib, scikit-learn, lifelines, among others.

## Results

- **Performance Metrics: The final model achieved an AUC of 0.85 and a C-index of 0.78, indicating robust performance in predicting bankruptcy risk.
- **Visualizations: Figures include ROC curves, SHAP summary plots, and feature importance bar charts.
- **Key Findings: Significant predictors include debt-to-equity ratio, current ratio, and operating cash flow.

## Discussion

- **Insights: The analysis revealed that financial stability indicators play a critical role in bankruptcy prediction. The model provides actionable insights to improve risk management.
- **Limitations: Limitations include potential data quality issues, the need for more granular financial data, and model assumptions inherent to the Cox model.
- **Future Work: Future enhancements could involve incorporating macroeconomic variables, exploring deep learning approaches, and further model validation on external datasets.