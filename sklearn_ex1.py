# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22  4 09:57:00 2018

@author: Liu-Da
"""
from sklearn.model_selection import  KFold

import mglearn as mg

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

import pandas as pd
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
print(pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mg.cm3))

import matplotlib.pyplot as plt

df = iris_dataframe.cumsum()
plt.figure(); df.plot();
plt.show()
