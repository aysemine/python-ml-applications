from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X= iris.data
y= iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

tree = DecisionTreeClassifier()

tree_param_grid = {"max_depth": [3, 5, 7],
                   "max_leaf_nodes": [None, 5, 10, 20, 30, 50]}

tree_grid_search = GridSearchCV(tree, tree_param_grid, cv=3)
tree_grid_search.fit(X_train, y_train)
print(tree_grid_search.best_params_)
print(tree_grid_search.best_score_)