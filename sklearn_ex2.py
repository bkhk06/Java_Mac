import mglearn
import matplotlib.pyplot as plt
# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
print(mglearn.discrete_scatter(X[:, 0], X[:, 1], y))
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()

X, y = mglearn.datasets.make_wave(n_samples=40)
print("X:",X)
print("y:",y)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

import numpy as np
print("Breast cancer:")
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))

from sklearn.datasets import load_boston
boston = load_boston()
print("Date shape: {}".format(boston.data.shape))

X,y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
print("x_train: {}".format(X_train))
print("y_train: {}".format(y_train))

print("x_test: {}".format(X_test))
print("y_test: {}".format(y_test))

from sklearn.neighbors import  KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {}".format(clf.score(X_test,y_test)))