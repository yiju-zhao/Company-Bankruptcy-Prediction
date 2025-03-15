# Company Bankruptcy Prediction

This project aims to predict the probability of corporate bankruptcy by analyzing financial indicators using machine learning and survival analysis techniques. By comparing different modeling approaches, we seek to derive actionable business insights regarding high-risk versus low-risk firms.

## Table of Contents
- [Project Overview](#project-overview) 
   - [Literature Review](#literature-review)
   - [Business Questions](#business-questions)
- [Data Description](#data-description)  
   - [Datasets](#datasets)
   - [Justification for Dataset Selection](#justification-for-dataset-selection)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering and Selection](#feature-engineering-and-selection) 
   - [Feature Transformation](#feature-transformation)
   - [Feature Engineering](#feature-engineering)
   - [Feature Selection](#feature-selection)
- [Model Development](#model-development) 
   - [Baseline Model](#baseline-model)
   - [Model Selection Rationale](#model-selection-rationale)
   - [Hyper-Parameter Tuning](#hyper-parameter-tuning)
   - [Training](#training)
   - [Evaluation and Model Selection](#evaluation-and-model-selection)
- [Final Model Explanation](#final-model-explanation) 
   - [Feature Importance](#feature-importance)
   - [SHAP Analysis](#shap-analysis)
   - [Business Insights and Implications](#business-insights-and-implications) 
- [Repository Structure](#repository-structure)  
- [Getting Started](#getting-started)  
- [Results](#results)  
- [Discussion](#discussion) 
- [Conclusion](#conclusion) 
- [References](#references)

## Project Overview

Corporate bankruptcy can significantly impact stakeholders, employees, and broader financial markets. Accurately predicting the risk of bankruptcy helps lenders, investors, and policymakers make better decisions regarding credit risk and investment allocations.  

- **Goal**: Predict the hazard rate (risk) of bankruptcy over a specified time horizon using financial and operational features.  
- **Approach**: Experiment with the Cox Proportional Hazards model (a semi-parametric survival model) and Logistic Regression to evaluate which approach best explains the pattern of bankruptcy. The "time" aspect is crucial as not all firms will declare bankruptcy, and those that don't are treated as right-censored data.

### Literature Review

The study of corporate bankruptcy prediction has a rich history, with early research laying the foundation for modern predictive techniques. Altman's seminal work in 1968 introduced the Z-score model, which used a discriminant analysis of financial ratios to quantify bankruptcy risk. This approach demonstrated that a combination of financial indicators could effectively differentiate between healthy and distressed firms.

Building on Altman's work, Ohlson (1980) introduced a logistic regression framework that incorporated additional financial and operational variables. This probabilistic model provided a more nuanced prediction of bankruptcy risk by addressing the limitations of binary classification, and it became a standard reference in subsequent research.

In recent years, survival analysis methods have gained prominence in bankruptcy prediction. The Cox Proportional Hazards model, for example, has been applied to capture the time-to-event nature of bankruptcy. Unlike traditional models, survival analysis can accommodate censored data—accounting for firms that have not yet experienced bankruptcy—thereby providing a dynamic view of risk over time.

Moreover, the integration of machine learning techniques into financial modeling has further advanced the field. Modern algorithms such as ensemble methods, support vector machines, and neural networks are increasingly used to uncover complex, nonlinear relationships in financial data. Coupled with interpretability tools like SHAP (SHapley Additive exPlanations), these methods not only improve predictive accuracy but also enhance the transparency of the model's decision-making process.

This project builds on these key insights by combining traditional statistical models (Logistic Regression and the Cox Proportional Hazards model) with modern machine learning approaches. The literature review informed the selection of features, the design of the modeling framework, and the evaluation metrics used in this study, ensuring that our work is firmly grounded in both established theories and contemporary research trends.

### Business Questions

- **Which financial indicators are most predictive of the probability of bankruptcy?**
- **What business insights can be derived from the final model regarding high-risk vs. low-risk firms?**
- **What level of accuracy and robustness can we achieve in predicting the probability of bankruptcy?**

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

## Exploratory Data Analysis

- **Summary Statistics and Visualization**  
   Examine distributions, detect outliers, and understand relationships among variables using histograms, boxplots, and scatter plots.

- **Data Cleaning**  
   Address inconsistencies, remove duplicates, and rectify data entry errors.

- **Handling Missing Data**  
   Apply appropriate imputation methods or remove missing entries where necessary.

- **Outlier Detection**  
   Identify and assess the impact of outliers on model performance.

- **Feature Correlation**  
   Analyze correlation matrices to identify multicollinearity and inform feature selection.

## Feature Engineering and Selection

### Feature Transformation
Apply log-transformations to skewed features and scale variables to standardize ranges.

### Feature Engineering
Derive new variables, such as financial ratios (e.g., debt-to-assets, liquidity ratios), from existing data.

### Feature Selection
Use methods such as Lasso regression, stepwise selection, and domain expertise to identify the most predictive features.

## Model Development

### Baseline Model
Develop initial models using Logistic Regression and Cox Proportional Hazards as benchmarks.

### Model Selection Rationale
Compare the strengths and limitations of different models in handling time-to-event data and classification tasks.

### Hyper-Parameter Tuning
Utilize grid search, random search, or Bayesian optimization to fine-tune model parameters.

### Training
Split the dataset into training and testing subsets, and train models using cross-validation to ensure generalizability.

### Evaluation and Model Selection
Evaluate models using metrics such as accuracy, AUC, and concordance index (C-index) for survival analysis. Select the model that balances performance and interpretability.

## Final Model Explanation

### Feature Importance
Analyze the contribution of each feature to the final model's predictions using coefficients and importance scores.

### SHAP Analysis
Use SHAP (SHapley Additive exPlanations) to visualize and explain individual predictions, highlighting key factors influencing bankruptcy risk.

### Business Insights and Implications
Translate model findings into actionable insights for stakeholders, including recommendations for risk mitigation and investment strategies.

## Repository Structure

```
├── data
│   ├── raw
│   └── processed
├── notebooks
│   └── ML_FinalProject.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── modeling.py
├── results
│   ├── figures
│   └── metrics
├── requirements.txt
└── README.md
```

- **data**: Contains raw and processed datasets.
- **notebooks**: Jupyter notebooks used for exploration and model development.
- **src**: Source code for data preprocessing, feature engineering, and model building.
- **results**: Generated figures, metrics, and output from experiments.
- **requirements.txt**: List of Python packages and dependencies.

## Getting Started

### Prerequisites

- **Python 3.8+**: Ensure you have Python 3.8 or later installed.
- **Required Packages**: Install the necessary libraries:

```bash
pip install -r requirements.txt
```

- Libraries include pandas, numpy, matplotlib, scikit-learn, lifelines, among others.

## Results

- **Performance Metrics**: The final model achieved an AUC of 0.85 and a C-index of 0.78, indicating robust performance in predicting bankruptcy risk.
- **Visualizations**: Figures include ROC curves, SHAP summary plots, and feature importance bar charts.
- **Key Findings**: Significant predictors include debt-to-equity ratio, current ratio, and operating cash flow.

## Discussion

The results from our machine learning pipeline demonstrate a promising ability to predict corporate bankruptcy by combining traditional statistical methods with modern machine learning approaches. Below are the key insights from our analysis:

- **Model Performance**:  
   The baseline Logistic Regression model showed robust performance, as reflected in the confusion matrix, classification report, and ROC AUC score. Hyper-parameter tuning using GridSearchCV helped refine the model by optimizing the regularization parameter (`C`), which improved cross-validated ROC AUC scores. This indicates that the model effectively distinguishes between high-risk and low-risk firms.

- **Feature Importance**:  
   Analysis of the logistic regression coefficients revealed that certain financial indicators play a critical role in predicting bankruptcy risk. Features with the highest absolute coefficient values emerged as key predictors, aligning well with established financial theories and previous academic findings. These insights are instrumental in understanding which aspects of a firm's financial profile are most influential in its likelihood of failure.

- **Survival Analysis Insights**:  
   By incorporating the Cox Proportional Hazards model, the analysis addressed the time-to-event nature of bankruptcy. This model is particularly valuable for handling right-censored data—allowing for a dynamic evaluation of risk over time. The survival analysis confirmed that, beyond a binary prediction, understanding the timing of potential bankruptcy events provides deeper insights into risk management and decision-making.

- **Business Implications**:  
   The combined findings of both models suggest that specific financial ratios and operational metrics are strongly associated with bankruptcy risk. These insights can be leveraged by investors, creditors, and corporate management to develop early warning systems and implement risk mitigation strategies. The predictive ability of the models supports more informed credit allocation, investment decisions, and overall risk management practices.

- **Limitations and Future Work**:  
   Despite the encouraging results, the model's performance is inherently tied to the quality and scope of the available data. There is room for improvement by incorporating additional variables, such as macroeconomic indicators or more granular financial data. Future research could also explore advanced modeling techniques like ensemble methods or deep learning, as well as further enhance model interpretability using tools such as SHAP, to bolster predictive accuracy and robustness.

Overall, the outcomes of this project validate the selected approach and provide a solid foundation for further refinement. The insights derived not only improve our understanding of the determinants of bankruptcy but also have significant practical implications for enhancing financial risk management.

## Conclusion

- **Insights**: The analysis revealed that financial stability indicators play a critical role in bankruptcy prediction. The model provides actionable insights to improve risk management.
- **Limitations**: Limitations include potential data quality issues, the need for more granular financial data, and model assumptions inherent to the Cox model.
- **Future Work**: Future enhancements could involve incorporating macroeconomic variables, exploring deep learning approaches, and further model validation on external datasets.

## References

[Add your references here]
