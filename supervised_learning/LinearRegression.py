from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import numpy as np


diabets = load_diabetes()

diabets_X, diabets_y = load_diabetes(return_X_y= True)

diabets_X = diabets_X[:, np.newaxis, 2]

diabets_X_train = diabets_X[:-20]
diabets_X_test = diabets_X[-20:]

diabets_y_train = diabets_y[:-20]
diabets_y_test = diabets_y[-20:]

lin_reg = LinearRegression()

lin_reg.fit(diabets_X_train, diabets_y_train)

diabets_y_pred = lin_reg.predict(diabets_X_test)

mse = mean_squared_error(diabets_y_test, diabets_y_pred)

rmse = np.sqrt(mse)

print(rmse)

plt.scatter(diabets_X_test, diabets_y_test, cmap="black")
plt.plot(diabets_X_test, diabets_y_pred, color = "red")
plt.show()