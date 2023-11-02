# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:03:36 2023

@author: MB516
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates    
from pyearth import Earth
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
np.warnings.filterwarnings('ignore') # 隱藏pandas未來有部分功能會更新的提醒
def calPerformance(y_true,y_pred):
    temp_list=[]  
    MAE_score=mean_absolute_error(y_true,y_pred)  
    temp_list.append(MAE_score)
    RMSE_score=mean_squared_error(y_true,y_pred,squared=False)  
    temp_list.append(RMSE_score)
    MAPE_score=mean_absolute_percentage_error(y_true,y_pred)  
    temp_list.append(MAPE_score) 
    return temp_list
data = pd.read_csv('Data/TrDatasetAfterPreces.csv',index_col=0) #TrDatasetAfterPreces(Beta)
pub_data = pd.read_csv('Data/PubDatasetAfterPreces.csv',index_col=0)
# data = pd.read_csv('Data/TrDatasetAfterPreces(beta,win).csv',index_col=0) #TrDatasetAfterPreces(Beta)
# pub_data = pd.read_csv('Data/PubDatasetAfterPreces(beta,win).csv',index_col=0)
df_x = data.drop(['單價'],axis=1)
df_y =  data['單價']

df_x, df_y = shuffle(df_x, df_y, random_state=516)
tr_x, ts_x, tr_y, ts_y = train_test_split(df_x, df_y, test_size=0.2, random_state=516)

KPI = 'ALL' # 'ALL'
if KPI == True:
    featureSelect = ['教育結構',
     '農林漁牧業就業比例',
     '縣市',
     '遷入人口數',
     '綜合所得(中位數)',
     '鄉鎮市區',
     '建物型態',
     '屋齡',
     '人口密度(人/平方公里)',
     '每醫療院所服務面積',
     '是否有捷運站']
    tr_x = tr_x[featureSelect]
    ts_x = ts_x[featureSelect]
    pub_data = pub_data[featureSelect]


#%% 
# 迴歸模型: 訓練集的績效表
Compare_df_tr = pd.DataFrame()               
Compare_df_tr.index = ['MAE','RMSE','MAPE']
 # 迴歸模型: 測試集的績效表
Compare_df_ts = pd.DataFrame()               
Compare_df_ts.index = ['MAE','RMSE','MAPE']

### RF
#Tunning

# rf_candidates = [{'n_estimators': [150,250,350,400,450], #150,200,300
#                   'max_depth':range(12,35,3),  #range(3,21,3)
#                   'max_features': [0.3,0.6,0.9],
#                   'min_samples_split':range(3,11,3), #range(3,30,3)
#                   'min_samples_leaf':range(3,11,3)}]
# rf = GridSearchCV(estimator = RandomForestRegressor(), param_grid = rf_candidates, cv = 3, scoring='neg_mean_absolute_percentage_error')
# rf.fit(tr_x, tr_y)
# print('\nBest parameters :',rf.best_params_)
# rf_stock = rf.best_estimator_

rf_stock = RandomForestRegressor(n_estimators = 450, max_depth = 18, max_features = 0.9, min_samples_split =3, min_samples_leaf = 2, random_state=516).fit(tr_x, tr_y)

# predict values
rf_trPred = rf_stock.predict(tr_x)
rf_tsPred = rf_stock.predict(ts_x)
rf_pubPred = rf_stock.predict(pub_data)

# preformance comparing table
Compare_df_tr['RF'] = calPerformance(tr_y.to_numpy() , rf_trPred)
Compare_df_ts['RF'] = calPerformance(ts_y.to_numpy() , rf_tsPred)

print(Compare_df_tr)
print(Compare_df_ts)

### XGB ###
# #Tunning
# xgb_candidates = [{'n_estimators': [350,400,450],
#                     'learning_rate': [0.03,0.05, 0.01],
#                     'max_depth': range(3,21,3), 
#                     'subsample': [0.5,0.7,0.9], 
#                     'colsample_bytree': [0.5,0.7,0.9]}] 
# xgb = GridSearchCV(estimator = XGBRegressor(objective = "reg:squarederror"), param_grid = xgb_candidates, cv = 3, scoring='neg_mean_absolute_percentage_error')
# xgb.fit(tr_x,tr_y)
# print('\nBest parameters :', xgb.best_params_)
# xgb_stock = xgb.best_estimator_


# xgb_stock =  XGBRegressor( objective = "reg:squarederror", max_depth = 16, n_estimators = 420, subsample = 0.9, colsample_bytree = 0.7, learning_rate = 0.01, random_state = 516).fit(tr_x,tr_y)
xgb_stock =  XGBRegressor( objective = "reg:squarederror", max_depth = 16, n_estimators = 440, subsample = 0.9, colsample_bytree = 0.7, learning_rate = 0.01, random_state = 516).fit(tr_x,tr_y)
xgb_trPred = xgb_stock.predict(tr_x)
xgb_tsPred = xgb_stock.predict(ts_x)
xgb_pubPred = xgb_stock.predict(pub_data)

# preformance comparing table
Compare_df_tr['XGB'] = calPerformance(tr_y.to_numpy() , xgb_trPred)
Compare_df_ts['XGB'] = calPerformance(ts_y.to_numpy() , xgb_tsPred)

print(Compare_df_tr)
print(Compare_df_ts)

allPredValue_tr = pd.DataFrame([rf_trPred,xgb_trPred]).T
allPredValue_tr.columns = ['RF','XGB']
allPredValue_tr['Average'] = allPredValue_tr.mean(axis=1)
allPredValue_ts = pd.DataFrame([rf_tsPred,xgb_tsPred]).T
allPredValue_ts.columns = ['RF','XGB']
allPredValue_ts['Average'] = allPredValue_ts.mean(axis=1)
Compare_df_tr['Avg. Perf.'] = calPerformance(tr_y.to_numpy() , allPredValue_tr['Average'] )
Compare_df_ts['Avg. Perf.'] = calPerformance(ts_y.to_numpy() , allPredValue_ts['Average'] )
print(Compare_df_tr)
print(Compare_df_ts)

#%%  importance 
#importance
imp = pd.DataFrame()
imp["x"] = tr_x.columns
imp["RF"] = rf_stock.feature_importances_                
imp["XGB"] = xgb_stock.feature_importances_      
imp['Average'] = imp.mean(axis=1)
imp = imp.sort_values(by='Average',ascending=False)
imp['Cum. Avg.'] = imp['Average'].cumsum()
# featureSelect = imp['x'][imp['Average'] >=0.01].to_list()
featureSelect = imp['x'][imp['Cum. Avg.'] <= 0.985].to_list()

imp['Average'] = imp['Average'].apply(lambda x : format(x, '.2%'))
imp['RF'] = imp['RF'].apply(lambda x : format(x, '.2%'))
imp['XGB'] = imp['XGB'].apply(lambda x : format(x, '.2%'))
imp['Cum. Avg.'] = imp['Cum. Avg.'].apply(lambda x : format(x, '.2%'))
print(imp)
#%% plot
max_price = 7
# training
plt.figure(figsize=(30, 5))
plt.title("Actual vs. Predicted house prices",fontsize=20)
# plt.scatter(tr_y, rf_trPred,color="blue",marker="o",facecolors="none")
plt.scatter(tr_y, xgb_trPred,color="green",marker="X",facecolors="none")
# plt.scatter(tr_y, allPredValue_tr['Average'],color="pink",marker="D",facecolors="none")
plt.plot([0,max_price],[0,max_price],"lightcoral",lw=2)
plt.xlabel("\nActual Price",fontsize=16)
plt.ylabel("\nPredict Price",fontsize=16)
# testing
plt.figure(figsize=(30, 5))
plt.title("Actual vs. Predicted house prices",fontsize=20)
# plt.scatter(ts_y, rf_tsPred,color="blue",marker="o",facecolors="none")
plt.scatter(ts_y, xgb_tsPred,color="green",marker="X",facecolors="none")
# plt.scatter(ts_y, allPredValue_ts['Average'],color="pink",marker="D",facecolors="none")
plt.plot([0,max_price],[0,max_price],"lightcoral",lw=2)
plt.xlabel("\nActual Price",fontsize=16)
plt.ylabel("\nPredict Price",fontsize=16)

#%%
#  public_dataset predicted value
pub_pred = pd.DataFrame([rf_pubPred, xgb_pubPred], index=['RF','XGB'], columns=pub_data.index).T
pub_pred['predicted_price'] = pub_pred.mean(axis=1)

# pub_pred['XGB'].to_csv('public_submission.csv')
