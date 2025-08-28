from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
# load dataset 
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns= cancer.feature_names)
df["target"] = cancer.target

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# standartization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# hyperparameter 
accuracy_values = []
k_values = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker = "s", linestyle = "--")
plt.title("Accuracy based on k values")
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
