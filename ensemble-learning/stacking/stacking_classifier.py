from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_models = [
    ("dt", DecisionTreeRegressor(max_depth=5, random_state=42)),
    ("svr", SVR(kernel="rbf", C=100))
]

meta_model = Ridge(alpha=1)

stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator = meta_model,
    cv=5
)

stacking_reg.fit(X_train, y_train)

y_pred = stacking_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f"rmse : {np.sqrt(mse)}")
print(f"r2 score : {r2_score(y_test, y_pred)}")