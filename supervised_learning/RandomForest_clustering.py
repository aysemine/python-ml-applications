from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
oli = fetch_olivetti_faces()

plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i], cmap="gray")

plt.show()

X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracy = []
tree_counts = [5, 50, 100, 500]

for tree_count in tree_counts:

    rf_clf = RandomForestClassifier(n_estimators=tree_count, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)

    print(acc)

plt.figure()
plt.plot(tree_counts, accuracy, marker="o")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.title("RandomForest Accuracy on Olivetti Faces")
plt.grid(True)
plt.show()