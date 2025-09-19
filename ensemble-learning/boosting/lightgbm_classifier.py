from sklearn.datasets import load_breast_cancer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lgbm_clf = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1, # L1 reduce overfitting
    reg_lambda=0.2, # l2 
    min_child_samples=20,
    min_split_gain=0.01,
    class_weight="balanced",
    boosting_type="gbdt",
    random_state=42
)

lgbm_clf.fit(X_train, y_train)

y_pred = lgbm_clf.predict(X_test)

print(f"acc : {accuracy_score(y_test, y_pred)}")
print(f"classification report : \n{classification_report(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.show()