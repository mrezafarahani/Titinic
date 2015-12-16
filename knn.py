from sklearn import neighbors, datasets
import pandas as pd
import numpy as np
import data_prep as dp
import math as m

train = dp.prep("train.csv")
test = dp.prep("test.csv")


w = test["Fare"]
test["Fare"] = map(m.log,w.values+5)

temp=np.random.rand(len(train))
temp = map(int, temp *5)
index = temp < 10
train["test"] = temp
init_train = train[train["test"]<4]
test_train = train[train["test"]>3]

X = dp.fix_size(init_train)
X_test = dp.fix_size(test_train)
X_t = dp.fix_size(test)
y = init_train.Survived.values
y_test = test_train.Survived.values

n_neighbors = 10
h = .02


clf_u = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
clf_u.fit(X, y)

y1 = clf_u.predict(X_test)
err1 = float(sum(abs(y1 - y_test)))/len(y_test)
print err1

clf_d = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
clf_d.fit(X, y)
y2 = clf_d.predict(X_test)
err2 = float(sum(abs(y2 - y_test)))/len(y_test)
print err2


