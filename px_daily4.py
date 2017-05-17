############################ Read data into pandas
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error , mean_absolute_error, median_absolute_error

df = pd.read_csv('px_0511.csv', delimiter= ',')
df['date'] = df[['YEAR', 'MONTH', 'DATE']].apply(lambda s: datetime.date(*s),axis=1)
df.head(3)
df.shape
df = df.iloc[:,[22,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
df = df.rename(columns = {'Y.DAILY':'Y'})
c = df.corr()['Y'].abs(); so = c.sort_values(kind='quicksort')
print(so)

def split_data(data, ntest):
    df_train, df_test = data.iloc[:ntest], data.iloc[ntest:]
    return df_train, df_test

def split_data2(data, nstart,ntest):
    df_train, df_test = data.iloc[nstart:ntest], data.iloc[ntest:]
    return df_train, df_test

# MRMR top 4
spread = df[['Y','M.BALANCE.WORLD.VIEW', 'SPREAD','M.DEMAND.ASIA','M.ASIA.PTA.EXPANSION']]




################################################## training: 2015.1 ~ 2016.6 #########################################

train, test = split_data (spread, 338)
y = np.array(train.iloc[:,0])
X = np.array(train.iloc[:,[1,2]] )

test = np.array(test);
ytest = test[:,0]
Xtest = test[:,[1,2]]

### Linear only
svr_linear = SVR(kernel='linear', C=200, gamma=0.1)
y_linear2 = svr_linear.fit(X,y).predict(Xtest)

############################ Prediction graph Daily

all_dates2 = df['date'][len(train):]
all_dates2 = np.array(all_dates2)
pddf2 = pd.DataFrame({'date':all_dates2, 'testy':ytest, 'y_linear':y_linear2})
pddf2 = pddf2.set_index('date')
pddf2.plot(figsize=(16,10),title = 'SVR_Linear_Kernel (Training: 2015/01~2016/06 Daily data) Top2 from MRMR')

rmse = np.sqrt(mean_squared_error(y_linear2, ytest))
print("linear RMSE %f" % rmse )



############################## Importing weekly data ##########################################
dfweek = pd.read_csv('px_0517_week.csv', delimiter= ',')



####################################### Monthly average
all_dates2 = dfweek['MONTH.Y'][len(train):]
all_dates2 = np.array(all_dates2)
pddf3 = pd.DataFrame({'month':all_dates2, 'testy':ytest, 'y_linear':y_linear2})

pddf3month = pddf3.groupby(['month'])['y_linear','testy'].mean()
pddf3month.plot(figsize=(16,10),title = 'SVR_Linear_Kernel (Training: 2015/01~2016/06 Monthly Average) Top2 from MRMR')

####################################### Weekly average
all_dates3 = dfweek['WEEK.Y'][len(train):]
all_dates3 = np.array(all_dates3)
pddf4 = pd.DataFrame({'week':all_dates3, 'testy':ytest, 'y_linear':y_linear2})

pddf4week = pddf4.groupby(['week'])['y_linear','testy'].mean()
pddf4week.plot(figsize=(16,10),title = 'SVR_Linear_Kernel (Training: 2015/01~2016/06 Weekly Average) Top2 from MRMR')




################################################## training: 2015.4 ~ 2016.9 #########################################

train, test = split_data2(spread, 54, 379)
y = np.array(train.iloc[:,0])
X = np.array(train.iloc[:,[1,2]] )

test = np.array(test);
ytest = test[:,0]
Xtest = test[:,[1,2]]

### Linear only
svr_linear = SVR(kernel='linear', C=110, gamma=0.01)
y_linear2 = svr_linear.fit(X,y).predict(Xtest)

all_dates2 = df['date'][379:]
all_dates2 = np.array(all_dates2)
pddf2 = pd.DataFrame({'date':all_dates2, 'testy':ytest, 'y_linear':y_linear2})
pddf2 = pddf2.set_index('date')
pddf2.plot(figsize=(16,10),title = 'SVR_Linear_Kernel (Training: 2015/04~2016/09 Daily data) Top2 from MRMR')

rmse = np.sqrt(mean_squared_error(y_linear2, ytest))
print("RMSE %f" % rmse )

############################ Prediction graph
all_dates2 = df['date'][379:]
all_dates2 = np.array(all_dates2)
pddf2 = pd.DataFrame({'date':all_dates2, 'testy':ytest,'y_linear':res_y})
pddf2 = pddf2.set_index('date')
pddf2.plot(figsize=(16,10),title='SVR_Linear_Kernel with optimized parameter (Training: 2015/04~2016/09 Daily data) Top2 from MRMR')



####################################### Monthly average
all_dates2 = dfweek['MONTH.Y'][379:]
all_dates2 = np.array(all_dates2)
pddf3 = pd.DataFrame({'month':all_dates2, 'testy':ytest, 'y_linear':y_linear2})

pddf3month = pddf3.groupby(['month'])['y_linear','testy'].mean()
pddf3month.plot(figsize=(16,10),title = 'SVR_Linear_Kernel (Training: 2015/01~2016/06 Monthly Average) Top2 from MRMR')

####################################### Weekly average
all_dates3 = dfweek['WEEK.Y'][379:]
all_dates3 = np.array(all_dates3)
pddf4 = pd.DataFrame({'week':all_dates3, 'testy':ytest, 'y_linear':y_linear2})

pddf4week = pddf4.groupby(['week'])['y_linear','testy'].mean()
pddf4week.plot(figsize=(16,10),title = 'SVR_Linear_Kernel (Training: 2015/01~2016/06 Weekly Average) Top2 from MRMR')



