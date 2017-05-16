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
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import operator

df = pd.read_csv('px_0511.csv', delimiter= ',')
df['date'] = df[['YEAR', 'MONTH', 'DATE']].apply(lambda s: datetime.date(*s),axis=1)
df.head(3)
df.shape
df = df.iloc[:,[22,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
df = df.rename(columns = {'Y.DAILY':'Y'})

############################## Naive Correlation
c = df.corr()['Y'].abs(); so = c.sort_values(kind='quicksort'); print(so)

############################## data split by certain time point
def split_data(data, ntest):
    df_train, df_test = data.iloc[:ntest], data.iloc[ntest:]
    return df_train, df_test

df1 = df.drop('date', 1); df1.head(3)

######################################################## 1. MRMR
matrix = df1.as_matrix(); matrix.shape

corrcoef = np.corrcoef(np.transpose(matrix))
relevancy = np.transpose(corrcoef)[0][1:]

x0 = np.ones(17)
fun = lambda x: sum([corrcoef[i+1, j+1] * x[i] * x[j] for i in range(len(x))
                     for j in range(len(x))]) / (sum(x) ** 2) - (sum(relevancy * x) / sum(x))

bound = ((0,1),) * 17; res = minimize(fun, x0, bounds = bound); res.x

# Rearrage data by Sorted MRMR
df2 = df1.drop('Y',1)
listX = list(df2) # All X
listY = res.x # MRMR X
sortedx = [x for (y,x) in sorted(zip(listY,listX), reverse= True)]

dfx = df2.reindex(columns= sortedx)
dfx.head(3)
dfx.shape

dfy = df1[['Y']]
dfyx = pd.concat([dfy, dfx], axis = 1)
dfyx.shape
########################################################## 2. RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
df2 = df1.drop('Y',1)
X = df2
names = list(X)
dfy = df1[['Y']]

lr = LinearRegression()
rfe = RFE(lr, n_features_to_select = 1) # Rank all features(continue elimination until the last one)
rfe.fit(X,dfy)
rfe.ranking_

print("Features sorted by their rank")
print( sorted(zip(map(lambda x: round(x, 4), rfe.ranking_ ), names)))

# Rearrage data by Sorted RFE ranking
df2 = df1.drop('Y',1)
listX = list(df2) # All X
listY = rfe.ranking_# MRMR X
sortedx = [x for (y,x) in sorted(zip(listY,listX))]

dfx = df2.reindex(columns= sortedx)
dfx.head(3)
dfx.shape

dfy = df1[['Y']]
dfyx = pd.concat([dfy, dfx], axis = 1)
dfyx.head(3)

################################################## training: 2015.1 ~ 2016.6 #########################################

# zig-zag combination of (RFE, MRMR)

#dfyx = df[['Y','PRODUCT.DELTA','M.BALANCE.WORLD.VIEW','W.PTA.OPER.RATE','SPREAD', 'M.DEMAND.ASIA','W.POLY.OPER.RATE', 'M.ASIA.PTA.EXPANSION']]
#dfyx.shape

######
train, test = split_data(dfyx, 338)

y = np.array(train.iloc[:,0])
X = np.array(train.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]] )
X.shape

test = np.array(test)
ytest = test[:,0]
xtest = test[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]


############################## Combinations( MRMR, Parameter)

# SVR with Linear Kernel

#c = [1,10,100,1000]
gam = [1, 0.1, 0.01]

c = list(range(1, 100))

'''
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

gam = list(frange(0.001, 10, 0.1))
'''


def function1(c, gam, kernel1):

    cgam = []
    for i in c:
        for j in gam:
            cgam.append({"c": i, "gam": j})
    cgam = pd.DataFrame(cgam); cgam = np.array(cgam)

    linearlist = []
    for index1, item1 in enumerate(c):
        for index2, item2 in enumerate(gam):
            svr_linear = SVR(kernel=kernel1, C=item1, gamma=item2)
            y_linear2 = svr_linear.fit(X, y).predict(xtest)
            linearlist.append(y_linear2)

    lineararray = np.array(linearlist)

    scorelist = []
    for i in range(len(lineararray)):
        rmse = np.sqrt(mean_squared_error(lineararray.T[:,i] , ytest) )
        scorelist.append(rmse)

    min_index = scorelist.index(min(scorelist))
    min_value = min(scorelist)

    print("min_index %f" % min_index)
    c1, gam1 = cgam[min_index, 0], cgam[min_index, 1]
    res_y = lineararray[min_index]

    return min_value, c1, gam1, res_y

rmse, c1, gam1, res_y = function1(c, gam, kernel1='linear')
print(rmse, c1, gam1)



############################ Prediction graph
all_dates2 = df['date'][len(train):]
all_dates2 = np.array(all_dates2)
pddf2 = pd.DataFrame({'date':all_dates2, 'testy':ytest,'y_linear':res_y})
pddf2 = pddf2.set_index('date')
pddf2.plot(figsize=(16,10))







