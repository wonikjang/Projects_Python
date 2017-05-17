import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from lstm2 import  lstm_model, load_csvdata
import matplotlib.dates as mdates


LOG_DIR = './ops_logs/px_lstm9'
TIMESTEPS = 7
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [7,7]
TRAINING_STEPS = 20000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

df = pd.read_csv('df0.csv', delimiter= ',')
df0 = df.iloc[:,[0,1]]
df0 = df0.set_index('date')
df0.describe()
df0.head(3)
df0.shape

X, y = load_csvdata(df0, TIMESTEPS)
y
##################################### Normalization1: Z = (X -mena(X))/std(X)

'''
xtrain_original_std = X["train"].std()
xtrain_original_mean = X["train"].mean()


Xtrain = (X["train"] - X["train"].mean()) / X["train"].std()
Xval = (X["val"] - X["val"].mean()) / X["val"].std()
Xtest = (X["test"] - X["test"].mean()) / X["test"].std()
X ={'train': Xtrain, 'val': Xval, 'test': Xtest}

ytrain = (y["train"] - y["train"].mean()) / y["train"].std()
yval = (y["val"] - y["val"].mean()) / y["val"].std()
#ytest = (y["test"] - y["test"].mean()) / y["test"].std()
y ={'train': ytrain, 'val': yval, 'test': y["test"]} ########## ytest should be the original one
'''

##################################### Normalization2: Z = (X -min(X))/ (Max(X) - min(X))
xtrain_original_min = X["train"].min()
xtrain_original_max = X["train"].max()

Xtrain = (X["train"] - X["train"].min()) / ( X["train"].max() - X["train"].min() )
Xval = (X["val"] - X["val"].min()) / ( X["val"].max() - X["val"].min() )
Xtest = (X["test"] - X["test"].min()) / ( X["test"].max() - X["test"].min() )
X ={'train': Xtrain, 'val': Xval, 'test': Xtest}

ytrain = (y["train"] - y["train"].min()) / ( y["train"].max() - y["train"].min() )
yval = (y["val"] - y["val"].min()) / ( y["val"].max() - y["val"].min() )
#ytest = (y["test"] - y["test"].mean()) / y["test"].std()
y ={'train': ytrain, 'val': yval, 'test': y["test"]} ########## ytest should be the original one



##################################### Model

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), model_dir=LOG_DIR)

validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])
predicted = list(predicted)
predicted = np.array(predicted)

##################################### Convert predicted value to original scale using traing set's information
##################################### according to previous normalization method

# 1. Normalizatin 1
'''
predicted = ( predicted * xtrain_original_std ) + xtrain_original_mean
'''
# 2. Normalization 2
predicted = ( predicted * ( xtrain_original_max - xtrain_original_min )  ) + xtrain_original_min


score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)
print ("RMSE: %f" % np.sqrt(score) )

# plot the data
all_dates = df0.index.get_values()
all_dates = np.array(all_dates,dtype=np.datetime64)
fig, ax = plt.subplots(1, figsize =(16, 10))
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
plt.title('PX Predictions vs. Test (TimeStep:7, epoch:20000) : Training 2015/04 ~ 201609')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()








