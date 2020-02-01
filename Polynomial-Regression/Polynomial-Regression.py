#!/Users/mahe/anaconda3/bin/python

import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("FuelConsumption.csv")
#print(data.head())
data = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
plt.scatter(data.ENGINESIZE, data.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
#plt.show()

# prepare train data and test data
mask = np.random.rand(len(data)) < 0.8
train = data[mask]
test = data[~mask]
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# transform x to polynomia data
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)

# Train
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(train_x_poly, train_y)
print ('Coefficients: ', model.coef_)
print ('Intercept: ',model.intercept_)

# Predict
test_x_poly = poly.fit_transform(test_x)
test_y_ = model.predict(test_x_poly)

from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
