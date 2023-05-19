#Regression Example With Normal Dataset: Standardized and Large (more hidden layers)
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import loadtxt

dataframe = read_csv(r'C:\Users\nCalo\Documents\Automifai\Building the SaaS product\Data_sets\2_weeks_SOH_removed.csv', delimiter=",")
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:20]
Y = dataset[:,20]

def larger_model():
    model = Sequential()
    model.add(Dense(20, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=50, batch_size=5, verbose=1)
    model.save('normal_data_model.h5')
    return model

# evaluate model with standardizestimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
estimator = KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=2)
kfold = KFold(n_splits=3)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

print("https://youtu.be/eCz_DTtUBfo")
