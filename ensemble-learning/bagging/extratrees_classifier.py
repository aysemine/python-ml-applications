from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np 
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

extra_tree = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

extra_tree.fit(X_train, y_train)

y_pred = extra_tree.predict(X_test)

acc = accuracy_score(y_test, y_pred)
clf_rep = classification_report(y_test, y_pred)
print(acc)
print(clf_rep)

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.show()

feature_importance = extra_tree.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
features = cancer.feature_names

plt.figure()
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align="center")
plt.xticks(range(X.shape[1]), features[sorted_idx], rotation = 90)
plt.tight_layout()
plt.show()

