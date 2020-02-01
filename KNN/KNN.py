#!/Users/mahe/anaconda3/bin/python

import numpy as np
import pandas as pd

# Prepare data
data = pd.read_csv("teleCust1000t.csv")
X = data[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
Y = data[['custcat']].values

# Normalize
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Prepare train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

# Find best k
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
mean_acc = np.zeros((49))
for k in range(1,50):
    model = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)
    Y_ = model.predict(X_test)
    mean_acc[k-1] = metrics.accuracy_score(Y_test, Y_)

import matplotlib.pyplot as plt
plt.plot(range(1,50), mean_acc, 'g')
plt.show()

# best k is 40, train again
model = KNeighborsClassifier(n_neighbors = 40).fit(X_train, Y_train)
Y_ = model.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, model.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Y_))
