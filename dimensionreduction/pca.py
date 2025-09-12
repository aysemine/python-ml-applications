from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


iris = load_iris()

X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
for i in range (len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y==i, 1], label = iris.target_names[i])
plt.show()

pca_3d = PCA(n_components=3)
X_pca = pca_3d.fit_transform(X)

fig = plt.figure(1, figsize=(8,6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c= y, s= 40)

plt.show()
