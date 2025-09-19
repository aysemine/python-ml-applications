from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

grad_boost = GradientBoostingClassifier(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=4, 
    subsample=0.8, 
    min_samples_split=5, 
    min_samples_leaf=3, 
    max_features="sqrt", 
    validation_fraction=0.1, 
    n_iter_no_change=5,  
    random_state=42)

grad_boost.fit(X_train, y_train)
y_pred = grad_boost.predict(X_test)

print("acc: {}".format(accuracy_score(y_test, y_pred)))
print(f"GB classification report : \n{classification_report(y_test, y_pred)}")
