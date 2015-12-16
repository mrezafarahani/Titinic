import pandas as pd
import numpy as np
import math as m

def prep(address):
    train = pd.read_csv(address)
    w = train["Fare"]
    train["Age"] /= 20.0
    train["Fare"] = map(m.log,w.values+5)
    train = train.drop("Name", axis=1)
    train = train.drop("Ticket", axis=1)
    train = train.drop("Cabin", axis=1)
    train = train.fillna(-1)
    index = train.Sex=="male"
    train.Sex = index + 0
    train.Sex *= 2
    j = 0
    embarked = index * 0
    for i in train.Embarked.unique():
        index = train.Embarked == i
        index += 0
        index = index * j
        embarked += index
        j += 1
    train.Embarked = embarked
    return train


def fix_size(test):
    X = [test.Pclass.values, test.Sex.values, test.Age.values, test.SibSp.values, test.Parch.values, test.Fare.values, test.Embarked.values]
    X = np.array(X)
    X = X.T
    return X