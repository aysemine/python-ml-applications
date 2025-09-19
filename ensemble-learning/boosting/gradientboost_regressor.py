from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

diabets = load_diabetes()

X = diabets.data
y = diabets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gboost = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=4,
    validation_fraction=0.1,
    n_iter_no_change= 5,
    random_state=42)

gboost.fit(X_train, y_train)

y_pred = gboost.predict(X_test)

print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"rmse: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"r2 : {r2_score(y_test, y_pred)}")

residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color = "r")
plt.xlabel("Predicted Values")
plt.title("GB - Error Distribution")
plt.ylabel("Error")
plt.show()

