import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from lstm import  lstm_model, load_csvdata
import matplotlib.dates as mdates


LOG_DIR = './ops_logs/px_lstm'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

df = pd.read_csv('df0.csv', delimiter= ',')
df0 = df.iloc[:,[0,1]]
df0 = df0.set_index('date')
df0.describe()
df0.head(3)
df0.shape

X, y = load_csvdata(df0, TIMESTEPS, seperate=False)

Xtrain = (X["train"] - X["train"].mean()) / X["train"].std()
Xval = (X["val"] - X["val"].mean()) / X["val"].std()
Xtest = (X["test"] - X["test"].mean()) / X["test"].std()
X ={'train': Xtrain, 'val': Xval, 'test': Xtest}

ytrain = (y["train"] - y["train"].mean()) / y["train"].std()
yval = (y["val"] - y["val"].mean()) / y["val"].std()
ytest = (y["test"] - y["test"].mean()) / y["test"].std()
y ={'train': ytrain, 'val': yval, 'test': ytest}


regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), model_dir=LOG_DIR)

validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])
predicted = list(predicted)

score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)
print ("RMSE: %f" % np.sqrt(score) )


# plot the data
all_dates = df0.index.get_values()
all_dates = np.array(all_dates,dtype=np.datetime64)
fig, ax = plt.subplots(1)
fig.autofmt_xdate()


predicted_values = predicted #already subset
predicted_dates = all_dates[len(all_dates)-len(predicted_values):len(all_dates)]
predicted_series = pd.Series(predicted_values, index=predicted_dates)
plot_predicted, = ax.plot(predicted_series, label='predicted')

test_values = list(y['test'])
test_dates = all_dates[len(all_dates)-len(test_values):len(all_dates)]
test_series = pd.Series(test_values, index=test_dates)
plot_test, = ax.plot(test_series, label='test')

xfmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('PX Predictions vs. Test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()

############################################################
########################## Original prediction vs. test
X1, y1 = load_csvdata(df0, TIMESTEPS, seperate=False)

predicted0 = np.array(predicted)
predicted1 = (predicted0 * y1['test'].std()) + y1['test'].mean()
score = mean_squared_error(predicted1, y1['test'])
print ("MSE: %f" % score)
print ("RMSE: %f" % np.sqrt(score) )


# plot the data
all_dates = df0.index.get_values()
all_dates = np.array(all_dates,dtype=np.datetime64)
fig, ax = plt.subplots(1)
fig.autofmt_xdate()

predicted_values = predicted1 #already subset
predicted_dates = all_dates[len(all_dates)-len(predicted_values):len(all_dates)]
predicted_series = pd.Series(predicted_values, index=predicted_dates)
plot_predicted, = ax.plot(predicted_series, label='predicted')

test_values = list(y1['test'])
test_dates = all_dates[len(all_dates)-len(test_values):len(all_dates)]
test_series = pd.Series(test_values, index=test_dates)
plot_test, = ax.plot(test_series, label='test')

xfmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('PX Predictions vs. Test')
plt.legend(handles=[plot_predicted, plot_test],loc='upper left')
plt.show()











