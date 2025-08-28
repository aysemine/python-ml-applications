import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# dont forget sort always sort ascending order
# if you want descending order you can add [::-1]
X = np.sort(5* np.random.rand(40 ,1), axis = 0)
y = np.sin(X).ravel()

plt.scatter(X, y)
plt.show()

# add noised value to each 5th number(total 8)
y[::5] += 1 * (0.5 - np.random.rand(8))

plt.scatter(X,y)
plt.show()

# create 500 evenly spaced values between 0 and 5 to use as test data
T = np.linspace(0, 5, 500)[:, np.newaxis]


for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i+1)
    plt.scatter(X, y, color = "green", label = "data")
    plt.plot(T, y_pred, color = "red", label = "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN regressor weights = {}.".format(weight))

plt.tight_layout()
plt.show()