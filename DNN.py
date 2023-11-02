# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:02:17 2023

@author: MB516
"""

##### Import general packages #####
## tensorflow ##
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.losses import  MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.regularizers import l1, l2, l1_l2
from keras import backend as K
## skopt ##
from skopt import BayesSearchCV
from skopt.space import Real
## py - related ##
import pandas as pd
import numpy as np

## sklearn ##
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
## reg metrics ##
from permetrics.regression import RegressionMetric
## plot ##
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates  
import seaborn as sns

##### Regression #####
def Adj_r2(y_true,y_pred):
    return (1 - (1-r2_score(y_true,y_pred))*(len(y_true)-1)/(len(y_true)-test_X.shape[1]-1))

def calPerformance(y_true,y_pred):
    temp_list=[]
    calPerf = RegressionMetric(y_true, y_pred, decimal=5)
    RMSE_score = calPerf.root_mean_squared_error()
    temp_list.append(RMSE_score)
    MAE_score = calPerf.mean_absolute_error()
    temp_list.append(MAE_score)
    MAPE_score = calPerf.mean_absolute_percentage_error()
    temp_list.append(MAPE_score)
    Adj_r2_score=Adj_r2(y_true,y_pred)
    temp_list.append(Adj_r2_score)
    r2=r2_score(y_true,y_pred)
    temp_list.append(r2)
    return temp_list
def buildTrain(train, pastDay, Y, futureDay):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay][Y]))
        train2 = train.drop([Y], axis=1)
        X_train.append(np.array(train2.iloc[i:i+pastDay]))
    return np.array(X_train), np.array(Y_train)
    

#%% read data & Setting
KPI = 'ALL' # 'ALL'
tuning = True # True, False

data = pd.read_csv('Data/TrDatasetAfterPreces(beta,win).csv',index_col=0)
pub_data = pd.read_csv('Data/PubDatasetAfterPreces(beta,win).csv',index_col=0)
df_x = data.drop(['單價'],axis=1)
df_y = data['單價']
df_x, df_y = shuffle(df_x, df_y, random_state=516)
train_X, test_X, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=516)

if KPI == True:
    featureSelect = ['鄉鎮市區', '屋齡', '是否有捷運站', '縣市', '總樓層數', '扶幼比', '到火車站距離', '綜合所得(中位數)', '到醫療機構距離', '到金融機構距離', '人口密度(人/平方公里)']
    train_X = train_X[featureSelect]
    test_X = test_X[featureSelect]
    pub_data = pub_data[featureSelect]

    
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)  

#%%
## standardize
#MinMax
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
#Std
# X_scaler = StandardScaler()
# y_scaler = StandardScaler()

train_y, test_y = train_y.values.reshape(-1, 1), test_y.values.reshape(-1, 1) 

train_X = X_scaler.fit_transform(train_X)
test_X = X_scaler.transform(test_X)
pub_X = X_scaler.transform(pub_data)

train_y = y_scaler.fit_transform(train_y)
test_y = y_scaler.transform(test_y)
#%%
if tuning == True:
    lr_scheduler = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000,decay_rate=0.98)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.7,patience=10, min_lr=0.000001)
    params = {
        # "n_dropout": [0,0.2, 0.1], #[0.2,0.5]
        'first_hidden_neuron': [16,32,64,128,256,512],
        "n_hidden": [3,5,7,11,15], #
        "n_neurons": [16,32,64,128,256,512,1024],
        # "learning_rate": Real(low=1e-5, high=9e-2, prior='log-uniform'),
        "activation" : ['relu'], #,'tanh'
        "kernel_initializer" : ['random_uniform','random_normal'], #'random_uniform','uniform','normal', 'lecun_uniform', 'normal', 'zero','lecun_normal','glorot_normal', 'glorot_uniform'
        # "momentum":[0.9,0.95],#[0.5, 0.9, 0.95, 0.99],
        "n_epochs": [1000],
        "n_batch_size": [10,12,15],
        "select_optimizer": ['SGD', 'Adam'] #optimizers.Adagrad,optimizers.Adam,optimizers.SGD
    }
    def create_model1(first_hidden_neuron=7,n_hidden=1,n_neurons=4,learning_rate=1e-2,
                     activation='relu',kernel_initializer='uniform',n_epochs=10,n_batch_size=10,
                     select_optimizer=SGD, n_dropout=0.2):
                         
        np.random.seed(516)
        model = Sequential()
        model.add(Dense(first_hidden_neuron, input_dim=train_X.shape[1], activation=activation, kernel_initializer=kernel_initializer))
        
        for layer in range(n_hidden):
            # model.add(BatchNormalization())
            model.add(Dense(n_neurons,activation=activation,kernel_initializer=kernel_initializer))
            # model.add(Dropout(n_dropout))
            
        model.add(Dense(1, activation='linear'))
        if select_optimizer == 'Adam':
            model.compile(loss='mean_squared_error',metrics=['mse'],optimizer = Adam(lr = lr_scheduler)) #mean_absolute_percentage_error / metrics=[MeanAbsolutePercentageError]
        if select_optimizer == 'SGD':
            model.compile(loss='mean_squared_error',metrics=['mse'],optimizer = SGD(lr = lr_scheduler, momentum=0.9))
       
        
        # model.fit(train_X,train_y, epochs=n_epochs, batch_size=n_batch_size, validation_split=0.1)
        model.fit(train_X,train_y, epochs=n_epochs, batch_size=n_batch_size, validation_split=0.15, callbacks=[es, reduce_lr_callback])
        return model
    # Bayesian optimization
    keras_reg = KerasRegressor(build_fn=create_model1)
    param_distribs = params
    # Parameter tuning using Bayes Search
    bayes_search_cv = BayesSearchCV(keras_reg,param_distribs,n_iter=10,cv=3,random_state=516,scoring='neg_mean_absolute_error', n_jobs=-1) #neg_mean_squared_error, neg_mean_absolute_error, neg_mean_absolute_percentage_error
    
    bayes_search_cv.fit(train_X,train_y)
    bayes_search_cv.best_params_
    
    #bayes_search_cv.best_score_
    model = bayes_search_cv.best_estimator_.model
    print(model.summary())
    print(bayes_search_cv.best_params_)
else:
    lrScheduler = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000,decay_rate=0.98)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=7, min_lr=0.00000001)
    # def lr_scheduler(epoch, lr=0.1):
    #     decay_rate = 0.85
    #     decay_step = 1
    #     if epoch % decay_step == 0 and epoch:
    #         return lr * pow(decay_rate, np.floor(epoch / decay_step))
    #     return lr
    # lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)
    np.random.seed(516)
    model = Sequential()
    model.add(Dense(units=128, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal')) #
    model.add(Dense(units=256, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal')) #
    model.add(Dense(units=512, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal')) #
    # model.add(Dropout(0.1))
    model.add(Dense(units=512, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal')) #
    # model.add(Dropout(0.2))
    model.add(Dense(units=256, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal')) #
    model.add(Dense(units=256, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal')) #
    # model.add(Dense(units=180, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='random_normal')) #
    model.add(Dense(units=128, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal')) #
    # model.add(Dense(units=100, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='random_normal')) #'random_normal'
    model.add(Dense(units=64, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal'))
    # model.add(Dense(units=64, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='random_normal'))
    # model.add(Dense(units=32, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='random_normal'))
    model.add(Dense(units=10, input_dim=train_X.shape[1], activation ='relu', kernel_initializer='normal'))
    # # model.add(Dropout(0.1))
   
    model.add(Dense(1,activation='linear',kernel_regularizer = l2(0.001))) #linear
    model.compile(loss = MeanSquaredError(),metrics=['mse'],optimizer = Adam(lr=lrScheduler)) #  MeanAbsoluteError, MeanSquaredError
    # model.compile(loss = MeanSquaredError(),metrics=['mse'],optimizer=SGD(lr=lrScheduler, momentum=0.9)) #  MeanAbsoluteError, MeanSquaredError
    model.summary()
    # model.get_config()

    # K.set_value(model.optimizer.learning_rate, 0.002)
    history = model.fit(train_X,train_y, epochs=1000, batch_size = 15, validation_split=0.15, callbacks=[es, reduce_lr_callback])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])


#%%
###### Performance #######
##### training #####
train_DNN = model.predict(train_X)
train_DNN = y_scaler.inverse_transform(train_DNN) # if normalize y, need to transform back to the origin value
train_y = y_scaler.inverse_transform(train_y) 
DNN_perf_tr = calPerformance(train_y, train_DNN)
pred_tr = pd.DataFrame([train_y.flatten(),train_DNN.flatten()], index = ['real','DNN']).T

##### testing #####
test_DNN=model.predict(test_X)
test_DNN = y_scaler.inverse_transform(test_DNN)
test_y = y_scaler.inverse_transform(test_y)
DNN_perf_ts = calPerformance(test_y,test_DNN)
pred_ts = pd.DataFrame([test_y.flatten(),test_DNN.flatten()], index = ['real','DNN']).T

# pred_ts.to_csv(f'test_DNN_predictedValue({Y}).csv')
# pred_tr.to_csv(f'train_DNN_predictedValue({Y}).csv')

perf = pd.DataFrame()
perf["DNN_Train"] = [round(x, 4) for x in DNN_perf_tr]
perf["DNN_Test"] = [round(x, 4) for x in DNN_perf_ts]
perf.index = ["RMSE", "MAE", "MAPE", "Adj_R2",'R2']
print(perf)

# public_dataset predicted value
pub_DNN = model.predict(pub_X)
pub_DNN = y_scaler.inverse_transform(pub_DNN)
pub_pred = pd.DataFrame([pub_DNN.flatten()], columns=pub_data.index, index=['predicted_price']).T
# pub_pred.to_csv('public_submission.csv')
#%% Plot
max_price = 7
plt.figure(figsize=(20, 5))
plt.title("Actual vs. Predicted house prices",fontsize=20)
plt.scatter(train_y, train_DNN, color="blue",marker="o",facecolors="none")
plt.plot([0,max_price],[0,max_price],"lightcoral",lw=2)
plt.xlabel("\nActual Price",fontsize=16)
plt.ylabel("\nPredict Price",fontsize=16)
# testing
plt.figure(figsize=(20, 5))
plt.title("Actual vs. Predicted house prices",fontsize=20)
plt.scatter(test_y, test_DNN,color="green",marker="X",facecolors="none")
plt.plot([0,max_price],[0,max_price],"lightcoral",lw=2)
plt.xlabel("\nActual Price",fontsize=16)
plt.ylabel("\nPredict Price",fontsize=16)