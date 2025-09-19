import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("ensemble-learning/boosting/diamonds.csv")

data["price_category"] = (data["price"] > 3000).astype(int)

X = data.drop(columns=["price", "price_category"])
y = data.price_category

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

cat_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5,
    loss_function="Logloss",
    eval_metric="Accuracy",
    cat_features=["cut", "color", "clarity"],
    verbose=100,
    early_stopping_rounds=30, 
)

cat_clf.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred = cat_clf.predict(X_test)

print(f"acc: {accuracy_score(y_test, y_pred)}")
print(f"classification report: \n{classification_report(y_test, y_pred)}")

plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
plt.show()