import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import pylab
import scipy.stats as stats
from scipy.stats import norm


# read files

df_test = pd.read_csv('C:/Users/Lam/Desktop/Code/ML/HousingPrediction/housepricesdata/test.csv')
df_train = pd.read_csv('C:/Users/Lam/Desktop/Code/ML/HousingPrediction/housepricesdata/train.csv')

print(df_train.info())
print(df_test.info())


# preprocessing

df_train.drop('Id', axis = 1, inplace=True)

# creating a linear regression model, so we only want numeric inputs
df_numerics_train = df_train.select_dtypes(include=np.number)
df_numerics_test = df_test.select_dtypes(include=np.number).interpolate()

# fill in missing values in training data
df_numerics_train['LotFrontage'].fillna(df_numerics_train['LotFrontage'].median(), inplace=True)
df_numerics_train['MasVnrArea'].fillna(0, inplace=True)
df_numerics_train['GarageYrBlt'].fillna(df_numerics_train['GarageYrBlt'].median(), inplace=True)


# data analysis

# correlation heatmap of all features

corrmat = df_numerics_train.corr()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(corrmat, vmax=.8, annot=True)
plt.show()

# correlation heatmap of features with a correlation > 0.5 with SalePrice

highest_corr = corrmat.index[abs(corrmat["SalePrice"])>0.5]
highest_corrmat = df_numerics_train[highest_corr].corr()
plt.figure(figsize=(10,8))
sns.heatmap(highest_corrmat, vmax=.8, annot=True)
plt.show()

# pair plot of features with a correlation > 0.5 with SalePrice
features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 
            'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
sns.pairplot(df_numerics_train[features], height = 2.5)
plt.show()

# distribution and probability plot of SalePrice

sns.distplot(df_numerics_train['SalePrice'], fit=norm)

fig = plt.figure()
res = stats.probplot(df_numerics_train['SalePrice'], plot=plt)
plt.show()


# modelling

X = df_numerics_train.drop(['SalePrice'], axis = 1)
y = df_numerics_train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

price_predictions = model.predict(X_test)

# scatter plot of actual price vs predicted price
plt.scatter(price_predictions, y_test, alpha = 0.5)
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')

# probability plot of model residuals
fig = plt.figure()
stats.probplot(y_test - price_predictions, dist="norm", plot=plt)
plt.show()

# regression metrics
mse = mean_squared_error(y_test, price_predictions)
rmse = math.sqrt(mean_squared_error(y_test, price_predictions))
r2 = r2_score(y_test, price_predictions)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")


# predictions using testing data
pred = model.predict(df_numerics_test.drop('Id', axis = 1))

submission = pd.DataFrame()
submission['Id'] = df_numerics_test['Id']
submission['SalePrice'] = pred

print(submission.head())