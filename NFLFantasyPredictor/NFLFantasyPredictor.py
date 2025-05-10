import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PowerTransformer
import math
import pylab
import scipy.stats as stats
from scipy.stats import norm, skew

# opponent_team - to be added after schedule for 2025 for further analysis based on matchups
# cols = ['player_id', 'player_name', 'position', 'season', 'season_type',
#         'completions', 'attempts', 'passing_yards', 'passing_tds', 
#         'interceptions', 'carries', 'rushing_yards', 'rushing_tds',
#         'rushing_fumbles', 'receptions', 'targets', 'receiving_yards',
#         'receiving_tds', 'receiving_fumbles', 'fantasy_points']

# currently testing WRs only
cols = ['player_id', 'player_name', 'position', 'season', 'season_type',
        'receptions', 'targets', 'receiving_yards',
        'receiving_tds', 'receiving_fumbles', 'fantasy_points']

df = pd.DataFrame(nfl.import_weekly_data([2020, 2021, 2022, 2023, 2024], cols))

df = df[(df['season_type'] == 'REG') & (df['position'] == 'WR')]

df_numerics = df.select_dtypes(include=np.number)

print(df_numerics.head(20))
print(df_numerics.info())


# EDA

# Correlation matrix between all the numeric variables
# corrmat = df_numerics.corr()
# f, ax = plt.subplots(figsize=(20, 10))
# sns.heatmap(corrmat, vmax=.8, annot=True)
# plt.show()

# distribution and probability plots
# sns.distplot(df_numerics['fantasy_points'], fit=norm)

# fig = plt.figure()
# res = stats.probplot(df_numerics['fantasy_points'], plot=plt)
# plt.show()

# ----------------------------------------------------------------------- #

# We can see that there is a positive skew, so we will apply a yeo-johnson
# transformation on the data to make the distribution more symmetric. We
# use this transformation rather than the log transformation as there are
# many variables with values of 0 or negative

# find variables with skewness over 0.75 to apply transformation onto
# skewed_variables = df_numerics.apply(lambda x: skew(x.dropna()))
# skewed_variables = skewed_variables[skewed_variables > .75]

# pt = PowerTransformer(method='yeo-johnson')
# df_numerics[skewed_variables.index] = pd.DataFrame(pt.fit_transform(df_numerics[skewed_variables.index]), 
#                                                    columns = skewed_variables.index, index = df_numerics.index)

# R2: 0.9137 for linear regression when transformed and R2: 0.9618 without transformation
# R2: 0.9337 for random forest when transformed and R2: 0.9555 without transformation
# R2: 0.9404 for gradient boosting when transformed and R2: 0.9612 without transformation

# so, we will not use the transformation

# ----------------------------------------------------------------------- #


# modelling

X = df_numerics.drop(['season', 'fantasy_points'], axis = 1)
y = df_numerics['fantasy_points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

fantasy_points_preds = lr_model.predict(X_test)
fantasy_points_preds2 = rf_model.predict(X_test)
fantasy_points_preds3 = gb_model.predict(X_test)

mse = mean_squared_error(y_test, fantasy_points_preds)
rmse = math.sqrt(mean_squared_error(y_test, fantasy_points_preds))
r2 = r2_score(y_test, fantasy_points_preds)

mse2 = mean_squared_error(y_test, fantasy_points_preds2)
rmse2 = math.sqrt(mean_squared_error(y_test, fantasy_points_preds2))
r2_2 = r2_score(y_test, fantasy_points_preds2)

mse3 = mean_squared_error(y_test, fantasy_points_preds3)
rmse3 = math.sqrt(mean_squared_error(y_test, fantasy_points_preds3))
r2_3 = r2_score(y_test, fantasy_points_preds3)

print(f"MSE: {mse:.2f} RF: {mse2:.2f} GB: {mse3:.2f}")
print(f"RMSE: {rmse:.2f} RF: {rmse2:.2f} GB: {rmse3:.2f}")
print(f"R2 Score: {r2:.4f} RF: {r2_2:.4f} GB: {r2_3:.4f}")