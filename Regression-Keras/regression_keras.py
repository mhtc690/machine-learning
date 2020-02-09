#!/Users/mahe/anaconda3/bin/python

import numpy as np
import pandas as pd
import keras

# process data, seperate to X and Y
data = pd.read_csv('concrete_data.csv')
data_columns = data.columns
data_X = data[data_columns[data_columns != 'Strength']]
data_Y = data['Strength']

# normalization
X_norm = (data_X - data_X.mean())/data_X.std()
input_num = X_norm.shape[1]

# create keras model
model = keras.Sequential()
model.add(keras.layers.Dense(50, activation='relu', input_shape=(input_num, )))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_norm, data_Y, epochs=100, verbose=2, validation_split=0.2)
