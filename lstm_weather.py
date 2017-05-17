import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from lstm import generate_data, lstm_model, load_csvdata
from pymongo import MongoClient
from bson.objectid import ObjectId
import dateutil.parser
import datetime
import matplotlib.dates as mdates

LOG_DIR = './ops_logs/lstm_weather'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 10000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

def load_weather_frame(filename):
    #load the weather data and make a date
    data_raw = pd.read_csv(filename, dtype={'Time': str, 'Date': str})
    data_raw['WetBulbCelsius'] = data_raw['WetBulbCelsius'].astype(float)
    times = []
    for index, row in data_raw.iterrows():
        print(row['Time'][:2], row['Time'][:-2])
        _t = datetime.time(int(row['Time'][:2]), int(row['Time'][:-2]), 0) #2153

        _d = datetime.datetime.strptime( row['Date'], "%Y%m%d" ) #20150905
        times.append(datetime.datetime.combine(_d, _t))

    data_raw['_time'] = pd.Series(times, index=data_raw.index)
    df =  pd.DataFrame(data_raw, columns=['_time','WetBulbCelsius'])
    return df.set_index('_time')


# scale values to reasonable values and convert to float
data_weather = load_weather_frame("QCLCD_PDX_20150901.csv")
list(data_weather)
data_weather.shape
data_weather.head(12)

X, y = load_csvdata(data_weather, TIMESTEPS, seperate=False)
X['train'].shape
X['val'].shape
X['test'].shape

y['train'].shape

X['test'][2]
y['test'][1]

plt.plot(y['train'])

keys = list(X.keys())
keys

X['train'].shape
X['train'][0].shape
y['train'].shape
y['train'][0]


regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), model_dir=LOG_DIR)

validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])
predicted = list(predicted)
#not used in this example but used for seeing deviations
#rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)
print ("RMSE: %f" % np.sqrt(score) )


# plot the data
all_dates = data_weather.index.get_values()
type(all_dates[0])
fig, ax = plt.subplots(1)
fig.autofmt_xdate()

predicted_values = predicted #already subset
predicted_dates = all_dates[len(all_dates)-len(predicted_values):len(all_dates)]
predicted_series = pd.Series(predicted_values, index=predicted_dates)
plot_predicted, = ax.plot(predicted_series, label='predicted (c)')

test_values = list(y['test'])
test_dates = all_dates[len(all_dates)-len(test_values):len(all_dates)]
test_series = pd.Series(test_values, index=test_dates)
plot_test, = ax.plot(test_series, label='2015 (c)')

xfmt = mdates.DateFormatter('%b %d %H')
ax.xaxis.set_major_formatter(xfmt)

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('PDX Weather Predictions for 2016 vs 2015')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()



