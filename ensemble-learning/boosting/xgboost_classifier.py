from xgboost import XGBClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
y = digits.target

fig, axes = plt.subplots(1, 2, figsize = (8,4))

for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap = "gray")
    ax.set_title(f"label: {digits.target[i]}")
    ax.axis("off")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xg_clf = XGBClassifier(
    n_estimators = 150,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 0.8, # sample rate for each tree
    colsample_bytree = 0.5, # use this much of the cols
    min_child_weight = 3, # min sample count on a leaf
    gamma = 0,
    early_stopping = 5, # if gamma = (convergence rate) for earlystopping times than stop
    eval_metrics = "mlogloss",
    random_state = 42,
    use_label_encoder = "False"
    )

xg_clf.fit(X_train, y_train, eval_set =[(X_test, y_test)], verbose = True)

y_pred = xg_clf.predict(X_test)

print(f"acc: {accuracy_score(y_test, y_pred)}")


