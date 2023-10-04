import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate
from datetime import timedelta
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pickle

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

################### Setting ###################

start_lat = 25.5
end_lat = 41
start_lon = 120
end_lon = 135.5
lat_bin = 0.5
lon_bin = 0.5

# LSTM input dataset (pickle)
model_input_path = path +'model_input_dataset_for_LSTM/'
input_data_list = os.listdir(model_input_path)
input_data = [j for j in input_data_list if j.endswith('.pickle')]

with open(model_input_path+input_data[0], 'rb') as t1:
    date, values = pickle.load(t1)

#interpolation을 했음에도 불구하고 첫 값이 NAN인 경우 interpolation을 하지 못함 따라서 0으로 변경
nan_value = np.where(np.isnan(values) == True)
values[nan_value] = 0

# configure
past_day = 1
past_history = 24 * past_day
future_target = 24
train_split = 6000
n_neuron = 10
batch_size = 256
epochs = 1000

cal_start_lat = start_lat + lat_bin
cal_end_lat = end_lat - lat_bin
cal_start_lon = start_lon + lon_bin
cal_end_lon = end_lon - lon_bin

cal_lat = np.arange(cal_start_lat, cal_end_lat + lat_bin, lat_bin, dtype=float)
cal_lon = np.arange(cal_start_lon, cal_end_lon + lon_bin, lon_bin, dtype=float)

for t_la in range(len(cal_lat)): #scaled.shape[1]):

    for t_lo in range(len(cal_lon)):

        target_data = values[:, 32 * t_la + (t_lo + 33)]
        resh_target_data = target_data.reshape(len(target_data), 1)
        comb_data = concatenate([date, resh_target_data], 1)  # Date data + neighbor data

        x_train, y_train = multivariate_data(comb_data, target_data, 0, train_split, past_history,
                                             future_target, single_step=False)

        x_test, y_test = multivariate_data(comb_data, target_data, train_split, None, past_history,
                                           future_target, single_step=False)

        scaler_train = MinMaxScaler(feature_range=(0, 1))
        x_train_2d = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        comb_train_data = concatenate([x_train_2d, y_train], axis=1)
        comb_train_data_scaled = scaler_train.fit_transform(comb_train_data)

        x_train_scaled_2d = comb_train_data_scaled[:, :(len(x_train_2d[0, :]))]
        x_train_scaled = x_train_scaled_2d.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
        y_train_scaled = comb_train_data_scaled[:, (len(x_train_2d[0, :])):]

        scaler_test = MinMaxScaler(feature_range=(0, 1))
        x_test_2d = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
        comb_test_data = concatenate([x_test_2d, y_test], axis=1)
        comb_test_data_scaled = scaler_test.fit_transform(comb_test_data)

        x_test_scaled_2d = comb_test_data_scaled[:, :(len(x_test_2d[0, :]))]
        x_test_scaled = x_test_scaled_2d.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
        y_test_scaled = comb_test_data_scaled[:, (len(x_test_2d[0, :])):]

        print('Single window of past history : {}'.format(x_train_scaled[0].shape))

        model = Sequential()
        model.add(LSTM(n_neuron, input_shape=(x_train_scaled.shape[1], x_train_scaled.shape[2])))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(future_target))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto')

        history = model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, shuffle=False, callbacks=[early_stopping])

        # make a prediction
        y_pred_scaled = model.predict(x_test_scaled)
        comb_scaled_test_data_y_pred = concatenate([x_test_scaled_2d, y_pred_scaled], axis=1)
        comb_test_data_orginal_y_pred = scaler_test.inverse_transform(comb_scaled_test_data_y_pred)
        y_pred = comb_test_data_orginal_y_pred[:, (len(x_test_2d[0, :])):]

        # make a prediction
        y_pred_scaled = model.predict(x_test_scaled)
        comb_scaled_test_data_y_pred = concatenate([x_test_scaled_2d, y_pred_scaled], axis=1)
        comb_test_data_orginal_y_pred = scaler_test.inverse_transform(comb_scaled_test_data_y_pred)
        y_pred = comb_test_data_orginal_y_pred[:, (len(x_test_2d[0, :])):]

        test_date = []
        for i in range(0, len(y_pred)):
            spec_date = datetime(int(x_test_2d[i, 0]), int(x_test_2d[i, 1]), int(x_test_2d[i, 2]),
                                 int(x_test_2d[i, 3]), 0, 0)
            test_date = np.append(test_date, spec_date + timedelta(days=past_day))

        pickle_path = '/Users/jeongseheon/Desktop/JSH/[1] Project/Forecast_TEC_LSTM/Data/1by1/' \
                      'Hidden_' + str(n_neuron) + '_pastday_' + str(past_day) + '_batch_' + str(batch_size) + '/'

        createFolder(pickle_path + 'Test_y/')
        createFolder(pickle_path + 'Test_y_pred/')

        import pickle

        with open(pickle_path + 'Test_y/test_y_' + str(32 * t_la + t_lo) + '.pickle', 'wb') as t1:
            pickle.dump([test_date, y_test], t1)

        with open(pickle_path + 'Test_y_pred/y_pred_' + str(32 * t_la + t_lo) + '.pickle', 'wb') as t2:
            pickle.dump(y_pred, t2)
