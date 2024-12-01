# CODTECH_task1
NAME:HITESH KUMAR PATEL
COMPANY:CODTECH IT SOLUTIONS
ID:CT08DS9771
DOMAIN:MACHINE LEARNING
DURATION:NOVEMBER TO DECEMBER 2024
MENTOR:

Overview: Boston Housing Price Prediction using Linear Regression
This project uses the Boston Housing dataset to predict median house prices based on various features of the houses and neighborhoods. The project employs Linear Regression for prediction and evaluation.

Project Objectives:
Understand the relationships between housing features and prices.
Build a predictive model using Linear Regression.
Evaluate model performance using metrics such as R-squared, MSE, and RMSE.
Gain insights into the most important factors influencing housing prices.
Libraries Used:
NumPy: For numerical computations.
Pandas: For data manipulation and analysis.
Matplotlib and Seaborn: For data visualization.
Scikit-learn: For preprocessing, model building, and evaluation.
Statsmodels: For detailed statistical summaries of the linear regression model.
Key Steps:
Data Loading:

Dataset: Boston Housing dataset (available in Scikit-learn's datasets module or via external sources).
Load the dataset and convert it into a Pandas DataFrame.
Exploratory Data Analysis (EDA):

Understand the dataset structure (e.g., features, target variable).
Visualize relationships using scatter plots, correlation heatmaps, and pair plots.
Check for missing values or anomalies.
Feature Selection:

Identify features highly correlated with the target variable (MEDV - Median value of owner-occupied homes).
Perform feature scaling if needed (e.g., Standardization or Normalization).
Model Building:

Use Statsmodels to fit an Ordinary Least Squares (OLS) regression model for detailed analysis.
Use Scikit-learn to fit a Linear Regression model for evaluation and predictions.
Model Evaluation:

Key Metrics:
R-squared and Adjusted R-squared: Measure the proportion of variance explained by the model.
Mean Squared Error (MSE) and Root Mean Squared Error (RMSE): Evaluate prediction accuracy.
Mean Absolute Percentage Error (MAPE): Interpret accuracy in percentage terms.
Visualize residuals to check for homoscedasticity and model assumptions.
Insights and Conclusion:

Analyze the importance of features based on coefficients.
Discuss the strengths and limitations of the model.
Suggest potential improvements (e.g., feature engineering, using advanced models like Ridge/Lasso Regression).
