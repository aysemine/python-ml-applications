from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# load dataset 
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(criterion="gini", max_depth= 5, random_state=42) # criterion might also be entropy
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(accuracy)
print(conf_mat)

plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled= True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.show()

feature_importences = tree_clf.feature_importances_

feature_names = iris.feature_names

feature_importences_sorted = sorted(zip(feature_importences, feature_names), reverse=True)

for importance, feature_name in feature_importences_sorted:
    print(f"{feature_name}: {importance}")
