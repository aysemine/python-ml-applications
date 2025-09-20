from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ada_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42)

param_grid = {
    "n_estimators" : [50, 100, 200],
    "learning_rate" : [0.01, 0.1, 1],
    "estimator__max_depth" : [1,2,3]
}

gridsearch = GridSearchCV(
    estimator=ada_clf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)
gridsearch.fit(X_train, y_train)

print(f"best params : {gridsearch.best_params_}")

best_model = gridsearch.best_estimator_
y_pred = best_model.predict(X_test)

print(f"best model acc : {accuracy_score(y_test, y_pred)}")
