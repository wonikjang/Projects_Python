import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from scipy.optimize import minimize

'''
################################################ Import & Transform data set
from sklearn.datasets import load_boston
boston = load_boston()

#ones = np.ones((len(boston.data),1)) # Create 1 for intercept
#Xall = np.hstack((ones, boston.data)) # Add 1 to original data
Xall = boston.data
yall = boston.target
'''

def data_split(data, test_size=0.2):

    ntest = int(round(len(data) * (1 - test_size)))
    df_train, df_test = data[:ntest], data[ntest:]

    return df_train, df_test

# If you want to specify the location of cutting point & starting point

def data_split2(data, nstart, ntest):
    df_train, df_test = data.iloc[nstart:ntest],  data.iloc[ntest:]
    return df_train, df_test

'''
nstart1 = 0; ntest1 = 400;
X, Xtest = data_split2(Xall, nstart1, ntest1);
y, ytest = data_split2(yall, nstart1, ntest1)
'''

################################################ 0. Feature Selection 1. RFE  2. MRMR

from sklearn.feature_selection import RFE

def feature_rearrange(Xall, yall, nstart1, ntest1, mode):

    if not isinstance(Xall, pd.DataFrame):
        Xall = pd.DataFrame(Xall)
    if not isinstance(yall, pd.DataFrame):
        yall = pd.DataFrame(yall)

    X, Xtest = data_split2(Xall, nstart1, ntest1);
    y, ytest = data_split2(yall, nstart1, ntest1)

    # Feature selection w.r.t whole data
    # If you want to change F.S. w.r.t train data => replace Xall,yall with X,y in rfe.fit(Xall,yall)
    # & df1 = pd.concat([yall, Xall], axis=1)

    # names = list(X)
    listX = list(X)  # All X

    # Rearrage data by Sorted RFE ranking
    if mode == 'RFE':
        lr = LinearRegression()
        rfe = RFE(lr, n_features_to_select = 1) # Rank all features(continue elimination until the last one)
        rfe.fit(Xall,yall)
        list_rfe = rfe.ranking_# RFE X

        sortedx = [x for (y,x) in sorted(zip(list_rfe,listX))]
        dfx = X.reindex(columns= sortedx); X = np.array(dfx)
        dfx1 = Xtest.reindex(columns= sortedx); Xtest = np.array(dfx1)

    # Rearrage data by Sorted MRMR
    if mode == 'MRMR':
        df1 = pd.concat([yall, Xall], axis=1); matrix = df1.as_matrix()

        corrcoef = np.corrcoef(np.transpose(matrix))
        relevancy = np.transpose(corrcoef)[0][1:]

        x0 = np.ones(X.shape[1])
        fun = lambda x: sum([corrcoef[i + 1, j + 1] * x[i] * x[j] for i in range(len(x))
                             for j in range(len(x))]) / (sum(x) ** 2) - (sum(relevancy * x) / sum(x))

        bound = ((0, 1),) * X.shape[1]; res = minimize(fun, x0, bounds=bound);

        res.x  # MRMR X
        sortedx = [x for (y, x) in sorted(zip(res.x, listX), reverse=True)]

        dfx = X.reindex(columns=sortedx); X = np.array(dfx)
        dfx1 = Xtest.reindex(columns=sortedx); Xtest = np.array(dfx1)

    y = np.concatenate(np.array(y))
    ytest = np.concatenate(np.array(ytest))

    return X , Xtest, y , ytest , dfx # export dfx for feature name later

'''
X, Xtest, y, ytest, dfx = feature_rearrange(Xall, yall, nstart1, ntest1, mode='MRMR')
'''

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error , mean_absolute_error, median_absolute_error

'''
models = [('linear', LinearRegression()),
          ('ridge', GridSearchCV( Ridge(fit_intercept=True), cv=5, param_grid={"alpha": np.linspace(.001,10,100)})),
          ('lasso', GridSearchCV( Lasso(fit_intercept=True), cv=5, param_grid={"alpha": np.linspace(.001,10,100)})),
          ('elastic-net', GridSearchCV(ElasticNet(fit_intercept=True), cv=5, param_grid={"alpha": np.linspace(.01, 10, 100)})),
          ('svr_linear', GridSearchCV(SVR(kernel='linear', gamma=0.1), cv=5,param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})),
          ('svr_rbf', GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}))
          #('kr', GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}))
          ]
'''

def fs_model_parm(X, Xtest, y, ytest , models, accur):

    dictlist = []
    keys = ['feature_num', 'method', 'model', 'predicted', 'accuracy']

    for i in range(13):
        X0 = X[:,:(i+1)] # sequnetially select variables arranged by MRMR or RFE
        X1 = Xtest[:,:(i+1)]

        print(i, X0.shape)

        # Given data, model

        for method, model in models :

            predicted = model.fit(X0, y).predict(X1)

            if accur == 'rmse':
                accuracy = np.sqrt(mean_squared_error(predicted, ytest))
            if accur == 'mape':
                accuracy = np.mean(np.abs((ytest - predicted) / ytest)) * 100
            if accur == 'min_max':
                accuracy = np.mean( np.minimum(predicted, ytest)/ np.maximum(predicted, ytest) )

            name = [i, method, model, predicted, accuracy]
            dictlist.append(dict(zip(keys, name)))

    # Model minimizing accuracy
    accuracylist = []
    for i in range(len(dictlist)):
        accuracylist.append(dictlist[i]['accuracy'])
    index = accuracylist.index(min(accuracylist))

    res_feature_num, res_method, res_model, res_predicted, res_accuracy =  \
        dictlist[index]['feature_num'], dictlist[index]['method'], dictlist[index]['model'], dictlist[index]['predicted'], dictlist[index]['accuracy']

    print('Feature Number: %d' % res_feature_num)
    print('GLM Method: %s' % res_method)
    print('GLM Model: %s' % res_model)
    print('RMSE_CV:%.2f' % res_accuracy)
    print("\n")

    return res_feature_num, res_method, res_model, res_predicted, res_accuracy


'''
res_feature_num, res_method, res_model, res_predicted, res_accuracy = GLM_SVR(X, Xtest, y, ytest ,'rmse')
print('Feature Name: %s' % list( dfx.iloc[:,:res_feature_num].columns.values ) )
'''



