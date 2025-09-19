from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing = fetch_california_housing(as_frame=True)

X = housing.data[["MedInc", "AveRooms", "HouseAge"]]
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ada_reg = AdaBoostRegressor(n_estimators=50, learning_rate=1, random_state=42)
ada_reg.fit(X_train, y_train)

y_pred = ada_reg.predict(X_test)
print("r2 score : {}".format(r2_score(y_test, y_pred)))
