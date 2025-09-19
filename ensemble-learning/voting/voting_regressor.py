from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model1 = SVR(kernel="rbf", C=100)
model2 = DecisionTreeRegressor(max_depth=5)
model3 = LinearRegression()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

y_pred_avg = (y_pred1 + y_pred2 + y_pred3) / 3

mse = mean_squared_error(y_test, y_pred_avg)

print(f"rmse: {np.sqrt(mse)}")
print(f"r2 score: {r2_score(y_test, y_pred_avg)}")