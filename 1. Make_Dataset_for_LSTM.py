import os
import numpy as np
import pandas as pd
import pickle

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def Make_Dataset_df(path, start_year, end_year, start_lon, end_lon, lon_bin, start_lat, end_lat, lat_bin):
    dif_year = (end_year - start_year) + 1
    total_year = np.linspace(start_year, end_year, dif_year, dtype=int)

    lat = np.arange(start_lat, end_lat + lat_bin, lat_bin, dtype=float)
    lon = np.arange(start_lon, end_lon + lon_bin, lon_bin, dtype=float)

    real_df = pd.DataFrame()

    path = path

    for y in range(len(total_year)):

        data_path = path + str(total_year[y]) + '/'
        file_list = os.listdir(data_path)
        file_list.sort()

        Error_msg = data_path + '.DS_Store'

        if os.path.exists(Error_msg):
            os.remove(Error_msg)

            data_path = path + str(total_year[y]) + '/'
            file_list = os.listdir(data_path)
            file_list.sort()

        else:
            print("Can not delete the file as it doesn't exists")

        for O_f in range(len(file_list)):

            DCGAN_file = data_path + file_list[O_f] + '/completed_pb_lr/1950.txt'

            if os.path.exists(DCGAN_file):

                # DCGAN-PB result
                with open(DCGAN_file, "r") as file:
                    DCGAN_pb_value = np.array([float(i) for line in file for i in line.split('/n') if i.strip()])

                # DCGAN_PB_result = np.reshape(DCGAN_pb_list, (32, 32))
                obs_year = int(file_list[O_f][12:16])
                obs_month = int(file_list[O_f][17:19])
                obs_day = int(file_list[O_f][20:22])
                obs_hour = int(file_list[O_f][23:25])

                # spec_time = datetime(obs_year, obs_month, obs_day, obs_hour, 0, 0)

                nan_value = np.where((DCGAN_pb_value == 9999.0) | (DCGAN_pb_value < 0.5) | (DCGAN_pb_value > 100))

                if len(nan_value[0]) == 0:
                    DCGAN_pb_value = DCGAN_pb_value
                else:
                    DCGAN_pb_value[nan_value] = np.nan

                date_data = {'Year': int(obs_year),
                             'Month': int(obs_month),
                             'Day': int(obs_day),
                             'Hour': int(obs_hour)}

                df_1_date = pd.DataFrame(date_data, index=[0])
                df_1_TEC = (pd.DataFrame(DCGAN_pb_value)).transpose()
                df_comb = pd.concat([df_1_date, df_1_TEC], axis=1)

                real_df = pd.concat([real_df, df_comb], axis=0)

            else:

                print('There is no data')

    return real_df

################### Setting ###################

start_year = 2010
end_year = 2010
start_lat = 25.5
end_lat = 41
start_lon = 120
end_lon = 135.5
lat_bin = 0.5
lon_bin = 0.5

path = '/Users/jeongseheon/Desktop/JSH/[2] Data/DCGAN_result/'
saving_path = '/Users/jeongseheon/Desktop/JSH/[2] Data/DCGAN_result/'

df_data = Make_Dataset_df(path, start_year, end_year, start_lon, end_lon, lon_bin, start_lat, end_lat, lat_bin)
df_data_intp = df_data.interpolate()
date = df_data_intp.values[:, 0:4]
values = df_data_intp.values[:, 4:]

#interpolation을 했음에도 불구하고 첫 값이 NAN인 경우 interpolation을 하지 못함 따라서 0으로 변경
nan_value = np.where(np.isnan(values) == True)
values[nan_value] = 0

createFolder(saving_path + 'model_input_dataset_for_LSTM')
with open(saving_path + 'model_input_dataset_for_LSTM/'
                        'Dataset_'+str(start_year)+'_'+str(end_year)+'.pickle', 'wb') as t1:
    pickle.dump([date, values], t1)

