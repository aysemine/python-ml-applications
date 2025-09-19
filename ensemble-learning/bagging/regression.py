from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "bagging regressor" : BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=100,
        max_features=0.8,
        max_samples=0.8,
        random_state=42
    ),
    "random forest regressor" : RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ),
    "extra tree regressor" : ExtraTreesRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
}

results = {}
predictions = {}

for name, model in tqdm(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"mse" : mse, "r2" : r2}
    predictions[name] = y_pred

result_df = pd.DataFrame(results).T

plt.figure()
for i, (name, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.5, label = name)
    plt.plot([y_test.min(), y_test.max()], [y_test.max(), y_test.min()], "r--", lw=2)
    plt.legend()
plt.tight_layout()
plt.show()