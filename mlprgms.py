# Question 1: Perform basic NumPy matrix operations (addition, element-wise multiplication, dot product,
# transpose, inverse, determinant, and mean) and visualize matrices A, B, and A @ B.
import matplotlib.pyplot as plt
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("Add:\n", a + b)
print("Multiply:\n", a * b)
print("Dot:\n", a @ b)
print("Transpose:\n", a.T)
print("Inverse:\n", np.linalg.inv(a))
print("Determinant:", np.linalg.det(a))
print("Mean:", np.mean(a))

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, mat, title in zip(axes, [a, b, a @ b], ["Matrix A", "Matrix B", "A @ B"]):
    ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center")
plt.tight_layout()
plt.show()


# Question 2: Create a Pandas DataFrame of employee data, inspect summary statistics, filter salaries
# above 58000, compute average salary by department, and plot a bar chart.
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Name": ["Amit", "Priya", "Rohan", "Sneha", "Vikram"],
    "Department": ["IT", "HR", "IT", "Finance", "HR"],
    "Salary": [60000, 52000, 75000, 68000, 55000],
    "Experience": [3, 2, 5, 4, 2],
}
df = pd.DataFrame(data)

print(df.head())
print(df.describe())
print(df[df["Salary"] > 58000])

dept_mean = df.groupby("Department")["Salary"].mean()
print(dept_mean)

dept_mean.plot(kind="bar", title="Average Salary by Department")
plt.ylabel("Salary")
plt.tight_layout()
plt.show()


# Question 3: Plot a line plot and a bar chart side by side using Matplotlib subplots.
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 40, 50]
categories = ["A", "B", "C", "D"]
values = [23, 45, 56, 78]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y, marker="o")
plt.title("Line Plot")

plt.subplot(1, 2, 2)
plt.bar(categories, values)
plt.title("Bar Chart")

plt.tight_layout()
plt.show()


# Question 4: Train a Decision Tree classifier on the Iris dataset, evaluate test accuracy,
# and plot a scatter plot of predicted classes using the first two features.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
preds = clf.predict(X_test)

print("Iris Accuracy:", accuracy_score(y_test, preds))

plt.scatter(X_test[:, 0], X_test[:, 1], c=preds, cmap="viridis", edgecolor="k")
plt.title("Iris Classification")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()


# Question 5: Fit a Simple Linear Regression model, print slope, intercept, MSE, and R2 score,
# and plot the actual points alongside the fitted regression line.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([3, 5, 7, 9, 11, 13])

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

print("Slope:", model.coef_[0], "Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y, y_pred), "R2:", r2_score(y, y_pred))

plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, color="red", label="Line")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


# Question 6: Fit a Multiple Linear Regression model on multi-feature data, print coefficients,
# intercept, and test R2 score, and plot actual vs. predicted values.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X = np.array(
    [
        [1, 2, 10],
        [2, 1, 20],
        [3, 3, 30],
        [4, 2, 40],
        [5, 4, 50],
        [6, 3, 60],
        [7, 5, 70],
        [8, 4, 80],
    ]
)
y = np.array([15, 22, 33, 41, 52, 61, 73, 81])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)

print("Coefficients:", model.coef_, "Intercept:", model.intercept_)
print("R2 Score:", r2_score(y_test, preds))

plt.scatter(y_test, preds)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.title("Multiple Linear Regression")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# Question 7: Fit a degree-2 Polynomial Regression model to non-linear data, print R2 score,
# and plot the quadratic curve fit.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = X.ravel() ** 2 + 2 * X.ravel() + 1

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
y_pred = model.predict(X_poly)

print("Polynomial R2:", r2_score(y, y_pred))

plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, color="red", label="Poly Fit")
plt.title("Polynomial Regression")
plt.legend()
plt.show()


# Question 8: Implement Batch Gradient Descent and Stochastic Gradient Descent from scratch for linear
# regression, compare final weights, and plot their MSE loss curves.
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)
X_b = np.c_[np.ones((len(X), 1)), X]

w_bgd = np.zeros(2)
w_sgd = np.zeros(2)
lr = 0.01
bgd_loss, sgd_loss = [], []
rng = np.random.default_rng(42)

for _ in range(200):
    w_bgd -= lr * (2 / len(X)) * X_b.T @ (X_b @ w_bgd - y)
    bgd_loss.append(np.mean((X_b @ w_bgd - y) ** 2))

    i = rng.integers(0, len(X))
    w_sgd -= lr * 2 * X_b[i : i + 1].T @ (X_b[i : i + 1] @ w_sgd - y[i : i + 1])
    sgd_loss.append(np.mean((X_b @ w_sgd - y) ** 2))

print("Batch GD:", w_bgd, "Stochastic GD:", w_sgd)

plt.plot(bgd_loss, label="Batch GD")
plt.plot(sgd_loss, label="Stochastic GD")
plt.title("Gradient Descent Loss")
plt.legend()
plt.show()


# Question 9: Compare Ridge and Lasso regression coefficients on a dataset and plot a grouped bar chart.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge

X = np.array(
    [
        [1, 2, 3],
        [2, 4, 6],
        [3, 1, 2],
        [4, 5, 7],
        [5, 2, 4],
        [6, 6, 9],
        [7, 3, 5],
        [8, 7, 10],
    ]
)
y = np.array([10, 20, 12, 25, 18, 32, 22, 38])

ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.5).fit(X, y)

print("Ridge Coefs:", ridge.coef_)
print("Lasso Coefs:", lasso.coef_)

idx = np.arange(3)
plt.bar(idx - 0.15, ridge.coef_, width=0.3, label="Ridge")
plt.bar(idx + 0.15, lasso.coef_, width=0.3, label="Lasso")
plt.xticks(idx, ["F1", "F2", "F3"])
plt.title("Ridge vs Lasso")
plt.legend()
plt.show()


# Question 10: Train a Gaussian Naive Bayes classifier on the Iris dataset, print accuracy,
# and display the confusion matrix.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

nb = GaussianNB().fit(X_train, y_train)
preds = nb.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, preds))

ConfusionMatrixDisplay.from_predictions(
    y_test, preds, display_labels=iris.target_names
)
plt.title("Naive Bayes Confusion Matrix")
plt.show()


# Question 11: Train a Logistic Regression classifier on a 2D synthetic classification dataset,
# print test accuracy, and visualize the decision boundary.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = LogisticRegression().fit(X_train, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
plt.title("Logistic Regression Boundary")
plt.show()


# Question 12: Train a Decision Tree classifier with max_depth=3 on the Iris dataset and plot the tree structure.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
)
plt.title("Decision Tree")
plt.show()


# Question 13: Compare Decision Tree accuracy using Gini vs. Entropy splitting criteria on the Iris dataset
# and plot a bar comparison.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

gini_acc = accuracy_score(
    y_test,
    DecisionTreeClassifier(criterion="gini", random_state=42)
    .fit(X_train, y_train)
    .predict(X_test),
)
ent_acc = accuracy_score(
    y_test,
    DecisionTreeClassifier(criterion="entropy", random_state=42)
    .fit(X_train, y_train)
    .predict(X_test),
)

print("Gini:", gini_acc, "Entropy:", ent_acc)

plt.bar(["Gini", "Entropy"], [gini_acc, ent_acc])
plt.ylim(0, 1.1)
plt.title("Gini vs Entropy")
plt.show()


# Question 14: Train a Linear Support Vector Machine (SVC) on 2D data, print accuracy, and plot
# the decision boundary along with the margin lines (-1, 0, +1).
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

svm = SVC(kernel="linear", C=1.0, random_state=42).fit(X_train, y_train)
print("Linear SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
)
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z > 0, alpha=0.25)
plt.contour(
    xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k"
)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
plt.title("Linear SVM")
plt.show()


# Question 15: Perform 5-fold cross-validation and train/test evaluation with Logistic Regression on Iris,
# then plot CV fold scores and the confusion matrix side by side.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

iris = load_iris()
clf = LogisticRegression(max_iter=200)
cv_scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("CV Scores:", cv_scores, "Mean:", cv_scores.mean())

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
clf.fit(X_train, y_train)
cm = confusion_matrix(y_test, clf.predict(X_test))
print("Confusion Matrix:\n", cm)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(range(1, 6), cv_scores)
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Cross Validation")

ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(ax=axes[1])
axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.show()


# Question 16: Evaluate a binary Logistic Regression classifier using Precision, Recall, F1-score,
# and ROC-AUC, and plot a bar chart of metrics alongside the ROC curve.
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=300, n_features=5, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = LogisticRegression().fit(X_train, y_train)
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]

p = precision_score(y_test, preds)
r = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)
print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

fpr, tpr, _ = roc_curve(y_test, probs)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(["Precision", "Recall", "F1"], [p, r, f1])
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Metrics")

axes[1].plot(fpr, tpr, label=f"AUC={auc:.2f}")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.show()


# Question 17: Compare accuracy of a Random Forest classifier vs. a hard VotingClassifier on Iris
# and plot a bar comparison.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(
    X_train, y_train
)
voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=200)),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ],
    voting="hard",
).fit(X_train, y_train)

rf_acc = accuracy_score(y_test, rf.predict(X_test))
vt_acc = accuracy_score(y_test, voting.predict(X_test))
print("RF:", rf_acc, "Voting:", vt_acc)

plt.bar(["Random Forest", "Voting"], [rf_acc, vt_acc])
plt.ylim(0, 1.1)
plt.title("Random Forest vs Voting")
plt.show()


# Question 18: Compare AdaBoost and Gradient Boosting classifiers on the Iris dataset and plot
# their test accuracies.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

ada = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(
    X_train, y_train
)

ada_acc = accuracy_score(y_test, ada.predict(X_test))
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print("AdaBoost:", ada_acc, "Gradient Boosting:", gb_acc)

plt.bar(["AdaBoost", "Gradient Boosting"], [ada_acc, gb_acc])
plt.ylim(0, 1.1)
plt.title("AdaBoost vs Gradient Boosting")
plt.show()


# Question 19: Evaluate K-Nearest Neighbors (KNN) across K values from 1 to 15 on the Iris dataset,
# print accuracy at K=5, and plot accuracy vs. K.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

k_vals = range(1, 16)
accs = [
    accuracy_score(
        y_test,
        KNeighborsClassifier(n_neighbors=k)
        .fit(X_train, y_train)
        .predict(X_test),
    )
    for k in k_vals
]

print("KNN (k=5):", accs[4])

plt.plot(k_vals, accs, marker="o")
plt.title("KNN Accuracy vs K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()


# Question 20: Cluster synthetic blob data using K-Means and Ward Agglomerative Hierarchical Clustering,
# print sample labels, and plot K-Means clusters (with centroids) alongside a truncated dendrogram.
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X)
agg_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)

print("K-Means:", km_labels[:10], "Hierarchical:", agg_labels[:10])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=km_labels)
axes[0].scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=150,
)
axes[0].set_title("K-Means")

dendrogram(
    linkage(X, method="ward"), ax=axes[1], truncate_mode="lastp", p=12
)
axes[1].set_title("Hierarchical Dendrogram")

plt.tight_layout()
plt.show()
