from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

X = np.sort(5 * np.random.rand(80, 1), axis = 0)
y = np.sin(X).ravel()

y[::5] += 0.5 * (0.5 - np.random.rand(16))

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=15)

regr_1.fit(X,y)
regr_2.fit(X,y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred1 = regr_1.predict(X_test)
y_pred2 = regr_2.predict(X_test)

plt.figure()
plt.scatter(X, y, c="red", label="data")
plt.plot(X_test, y_pred1, color = "blue", label = "max_depth: 2", linewidth=2)
plt.plot(X_test, y_pred2, color = "green", label = "max_depth: 15", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.show()
