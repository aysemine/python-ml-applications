from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

xg_reg = XGBRegressor(
    n_estimatprs = 200,
    max_depth = 6,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    early_stopping_rounds = 10,
    eval_metric = "rmse",
    random_states = 42
)

xg_reg.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = True)
y_pred = xg_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"mse : {mse}")
print(f"rmse : {np.sqrt(mse)}")
print(f"r2 score : {r2_score(y_test, y_pred)}")

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0,5], [0,5], color = "red")
plt.xlabel("true")
plt.ylabel("prediction")
plt.grid(True)
plt.show()
