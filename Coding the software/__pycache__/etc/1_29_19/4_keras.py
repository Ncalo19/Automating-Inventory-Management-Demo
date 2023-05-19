#Regression Example With Normal Dataset: Standardized and Large (more hidden layers)
import pandas as pd
import numpy as np
import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import loadtxt

df = pd.read_csv(r'C:\Users\nCalo\Documents\Automifai\Building the SaaS product\Data_sets\2_weeks_SOH_removed.csv', delimiter=",")
df.head()
dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:,0:20]
Y = dataset[:,20]

model = keras.Sequential()
model.add(keras.layers.Dense(20, input_dim=20, kernel_initializer='normal', activation='selu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=50, batch_size=2, verbose=2, shuffle=True)
model.save('normal_data_model.h5')

# evaluate model with standardizestimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)


print("https://youtu.be/eCz_DTtUBfo")
print("https://www.youtube.com/watch?v=zinEPDj7SD8")
print("https://keras.io/activations/")
