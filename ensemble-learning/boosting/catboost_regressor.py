import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("ensemble-learning/boosting/diamonds.csv")

X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

cat_reg = CatBoostRegressor(
    iterations=500,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=4,
    loss_function="RMSE",
    cat_features=["cut","color","clarity"],
    random_state=42,
    verbose=100,
    early_stopping_rounds=50
)

cat_reg.fit(X_train, y_train)

y_pred = cat_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"rmse : {np.sqrt(mse)}")
print(f"r2 score : {r2_score(y_test, y_pred)}")


plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, 20000], [0,20000], color="red")
plt.show()