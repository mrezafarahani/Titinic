from sklearn import svm
import pandas as pd
import numpy as np
import data_prep as dp
import math as m

train = dp.prep("train.csv")
test = dp.prep("test.csv")

w = train["Fare"]
train["Fare"] = map(m.log, w.values+5)
w = test["Fare"]
test["Fare"] = map(m.log, w.values+5)

temp = np.random.rand(len(train))
temp = map(int, temp *5)
index = temp < 5
train["test"] = temp
init_train = train[train["test"] < 4]
test_train = train[train["test"] == 4]

X = dp.fix_size(init_train)
X_test = dp.fix_size(test_train)
X_t = dp.fix_size(test)
Y = init_train.Survived.values
y_test = test_train.Survived.values


clf = svm.SVC()
clf.fit(X, Y)
y1 = clf.predict(X_test)
err1 = float(sum(abs(y1 - y_test)))/len(y_test)
print err1

poly_svc = svm.SVC(kernel='poly', degree=2).fit(X, Y)
y2 = poly_svc.predict(X_test)
err2 = float(sum(abs(y2 - y_test)))/len(y_test)
print err2

rbf_svc = svm.SVC(kernel='rbf', gamma=0.7).fit(X, Y)
y3 = rbf_svc.predict(X_test)
err3 = float(sum(abs(y3 - y_test)))/len(y_test)
print err3

u = rbf_svc.predict(X_t)
test["Survived"] = u
test.to_csv("result.csv")
