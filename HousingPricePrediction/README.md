This is a Python project that predicts the prices of homes in Ames, Iowa using a dataset consisting of 79 variables. 

Data provided via https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

---

## 🔍 Findings

Looking at the correlation matrix, we see that the Overall Quality of the home has the greatest impact on its price. Additionally, there are numerous variables that factor into the Overall Quality of the home, mainly the year it was build / remodelled, total basement area, total above ground area, and garage area / size. As for the Linear Regression model I created, the metrics calculated in the end were:

- RMSE: 36836.91
- R2 Score: 0.8231

This means that the average prediction error by the model is $36836.91 and that 82.31% of the variation in the home's price can be explained by the model.

---

## 🧮 Steps Taken

1. **Data Preprocessing**
   - Handled missing values and filled them using the median or 0
   - Removed outliers

2. **Exploratory Data Analysis**
   - Correlation heatmaps and pairplots
   - Distribution and Q-Q plots

4. **Modeling**
   - Trained Linear Regression
   - Evaluated with MSE, RMSE, and R²

---
  
## 🛠️ Package Versions:

| Package        | Version |
|----------------|---------|
| Python         | 3.10.6  |
| notebook       | 7.4.2   |
| matplotlib     | 3.10.3  |
| numpy          | 2.2.5   |
| pandas         | 2.2.3   |
| scikit-learn   | 1.6.1   |
| scipy          | 1.15.3  |
| seaborn        | 0.13.2  |

---
