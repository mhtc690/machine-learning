#!/usr/local/bin/python3

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# read csv file and extract values
my_data = pd.read_csv("drug200.csv", delimiter=',')
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
Y = my_data[['Drug']].values

# transform word values to int
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Cho = preprocessing.LabelEncoder()
le_Cho.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Cho.transform(X[:,3])

# prepare train set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Create decision tree
DT = DecisionTreeClassifier(criterion='entropy')
# Train decision tree
DT.fit(X_train, Y_train)

# predict and print result
prd = DT.predict(X_test).reshape(len(X_test), 1)
comp = np.equal(prd, Y_test).astype(int)
print("predition accurate:", np.sum(comp)/len(prd))
