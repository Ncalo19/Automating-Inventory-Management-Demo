model = keras.Sequential()
model.add(keras.layers.Dense(27, input_dim=27, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=75, batch_size=25, verbose=2, shuffle=True)
model.save('normal_data_model.h5')

#15
model = keras.Sequential()
model.add(keras.layers.Dense(27, input_dim=27, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=.0005))
model.fit(X, Y, epochs=75, batch_size=25, verbose=2, shuffle=True)
model.save('normal_data_model.h5')
