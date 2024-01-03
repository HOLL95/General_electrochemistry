import numpy as np

# 1. Generate some data.
n = 10          # Data set size
p = 2           # Number of regressors
np.random.seed(17)
z = 3*np.random.normal(size=(n, p+1))+8
col_names = [f"x{i}" for i in range(1, p+2)]
y = z[:, p]
x = z[:, :-1]

# 2. Find the OLS coefficients from the covariances only.
a = np.cov(x, rowvar=False)  # Covariance matrix of x
b = np.cov(x, y, rowvar=False)[:-1, -1]  # Covariances between x and y
beta_hat = np.linalg.solve(a, b)  # Coefficients from the covariance matrix

# 2a. Find the intercept from the means and coefficients.
y_bar = np.mean(y)
x_bar = np.mean(x, axis=0)
intercept = y_bar - np.dot(x_bar, beta_hat)
import pandas as pd
import statsmodels.api as sm

# Create a DataFrame to store the results
results = pd.DataFrame(index=['From covariances', 'From data via OLS'], columns=['(Intercept)'] + col_names)

# Fill the DataFrame with coefficients
print(intercept, beta_hat)
print(sm.OLS(y, sm.add_constant(x)).fit().params)
