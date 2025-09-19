# import libraries
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset 
iris = load_iris()
X = iris.data #features
y = iris.target

# data train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define base model 
base_model = DecisionTreeClassifier(random_state=42)

# create bagging model 
bagging_model = BaggingClassifier(
    estimator= base_model,
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)
# model training
bagging_model.fit(X_train, y_train)

# model testing
y_pred = bagging_model.predict(X_test)

# evaluate model accuracy
acc = accuracy_score(y_test, y_pred)

print("acc: {}".format(acc))


