from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ada_Clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=42)
ada_Clf.fit(X_train, y_train)

y_pred = ada_Clf.predict(X_test)

print("acc : {}".format(accuracy_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt= "d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("prediction")
plt.ylabel("True")
plt.show()