from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model1 = LogisticRegression(max_iter=200)
model2 = DecisionTreeClassifier(max_depth=2)
model3 = SVC()

voting_clf = VotingClassifier(
    estimators=[("lr", model1), ("dt", model2), ("svc", model3)],
    voting="hard" # hard soft 
)

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)

print(f"acc: {accuracy_score(y_test, y_pred)}")
