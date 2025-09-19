from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_models = [
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("svc",SVC(probability=True, kernel="rbf", C=1, random_state=42))
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method="predict_proba"
)

stacking_clf.fit(X_train,  y_train)

y_pred = stacking_clf.predict(X_test)

print(f"acc: {accuracy_score(y_test, y_pred)}")
print(f"classification report : \n{classification_report(y_test, y_pred)}")
