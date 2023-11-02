# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:52:59 2023

@author: YuLing
"""
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import math
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

data = pd.read_csv('Data/training_data.csv',index_col=0)
# tr_data = pd.read_csv('Data/training_datatAfterPreces.csv',index_col=0)
orignal_pub_data = pd.read_csv('Data/public_dataset.csv',index_col=0)
# tr_data.info()
tr_data = data.copy()
pub_data = orignal_pub_data.copy()

# 單價分佈圖、boxplot
# fig = plt.figure(figsize=(16, 5))
# fig.add_subplot(2,1,1)
# sns.distplot(data['單價'],kde=True)  #刪除密度曲线，kde=False
# fig.add_subplot(2,1,2)
# sns.boxplot(data['單價'], orient='h') #橫向orient='h'
# plt.tight_layout()

#%% 刪除 '備註','路名','使用分區'
tr_data.drop(['備註','路名','使用分區'], axis=1, inplace=True) 
pub_data.drop(['備註','路名','使用分區'], axis=1, inplace=True)
# tr_data['主建物佔比(%)'] = data['主建物面積']/ data['建物面積']
#%% 加入已處理完成的距離資料
distance_tr = pd.read_csv('Data/distance_tr_NEW.csv',index_col=0)
distance_pub = pd.read_csv('Data/distance_pub_NEW.csv',index_col=0)
# distance_tr = pd.read_csv('Data/distance_tr.csv',index_col=0)
# distance_pub = pd.read_csv('Data/distance_pub.csv',index_col=0)
# distance_tr = distance_tr.iloc[:,:2]
# distance_pub = distance_pub.iloc[:,:2]
tr_data = pd.concat([tr_data,distance_tr], axis=1)
pub_data = pd.concat([pub_data,distance_pub], axis=1)

#%% 國道交流道收費門座標
freeWay = pd.read_csv('Data/external_data/國道計費門架座標及里程牌價表1120426(國一、三、三甲、五).csv')
freeWay1 = freeWay[freeWay['方向'] == 'N'].loc[:,['國道別','迄點交流道','緯度','經度']]
freeWay1.columns = ['國道別','設施名稱','緯度','經度']
freeWay2 = pd.read_excel('Data/external_data/國道2、4、6、8、10(人工經緯度).xlsx')
freeway = pd.DataFrame(columns=['國道別','交流道','緯度','經度'])
pd
#%% 相同鄉鎮市區重新命名
# training_dataset
b=(tr_data['縣市']=='台北市') & (tr_data['鄉鎮市區']=='信義區')
tr_data.loc[b,'鄉鎮市區']='信義區-台北'
b=(tr_data['縣市']=='台北市') & (tr_data['鄉鎮市區']=='中正區')
tr_data.loc[b,'鄉鎮市區']='中正區-台北'
b=(tr_data['縣市']=='台北市') & (tr_data['鄉鎮市區']=='中山區')
tr_data.loc[b,'鄉鎮市區']='中山區-台北'
b=(tr_data['縣市']=='台北市') & (tr_data['鄉鎮市區']=='大安區')
tr_data.loc[b,'鄉鎮市區']='大安區-台北'

b=(tr_data['縣市']=='基隆市') & (tr_data['鄉鎮市區']=='信義區')
tr_data.loc[b,'鄉鎮市區']='信義區-基隆'
b=(tr_data['縣市']=='基隆市') & (tr_data['鄉鎮市區']=='中正區')
tr_data.loc[b,'鄉鎮市區']='中正區-基隆'
b=(tr_data['縣市']=='基隆市') & (tr_data['鄉鎮市區']=='中山區')
tr_data.loc[b,'鄉鎮市區']='中山區-基隆'


b=(tr_data['縣市']=='台中市') & (tr_data['鄉鎮市區']=='北區')
tr_data.loc[b,'鄉鎮市區']='北區-台中'
b=(tr_data['縣市']=='台中市') & (tr_data['鄉鎮市區']=='南區')
tr_data.loc[b,'鄉鎮市區']='南區-台中'
b=(tr_data['縣市']=='台中市') & (tr_data['鄉鎮市區']=='中區')
tr_data.loc[b,'鄉鎮市區']='中區-台中'
b=(tr_data['縣市']=='台中市') & (tr_data['鄉鎮市區']=='東區')
tr_data.loc[b,'鄉鎮市區']='東區-台中'
b=(tr_data['縣市']=='台中市') & (tr_data['鄉鎮市區']=='西區')
tr_data.loc[b,'鄉鎮市區']='西區-台中'
b=(tr_data['縣市']=='台中市') & (tr_data['鄉鎮市區']=='大安區')
tr_data.loc[b,'鄉鎮市區']='大安區-台中'

b=(tr_data['縣市']=='台南市') & (tr_data['鄉鎮市區']=='北區')
tr_data.loc[b,'鄉鎮市區']='北區-台南'
b=(tr_data['縣市']=='台南市') & (tr_data['鄉鎮市區']=='南區')
tr_data.loc[b,'鄉鎮市區']='南區-台南'
b=(tr_data['縣市']=='台南市') & (tr_data['鄉鎮市區']=='東區')
tr_data.loc[b,'鄉鎮市區']='東區-台南'

# a=tr_data[['縣市','鄉鎮市區']].drop_duplicates(keep='first')
# a[a.duplicated('鄉鎮市區')]

# public_dataaet
b=(pub_data['縣市']=='台北市') & (pub_data['鄉鎮市區']=='中正區')
pub_data.loc[b,'鄉鎮市區']='中正區-台北'
b=(pub_data['縣市']=='台北市') & (pub_data['鄉鎮市區']=='信義區')
pub_data.loc[b,'鄉鎮市區']='信義區-台北'
b=(pub_data['縣市']=='台北市') & (pub_data['鄉鎮市區']=='中山區')
pub_data.loc[b,'鄉鎮市區']='中山區-台北'
b=(pub_data['縣市']=='台北市') & (pub_data['鄉鎮市區']=='大安區')
pub_data.loc[b,'鄉鎮市區']='大安區-台北'

b=(pub_data['縣市']=='基隆市') & (pub_data['鄉鎮市區']=='中正區')
pub_data.loc[b,'鄉鎮市區']='中正區-基隆'
b=(pub_data['縣市']=='基隆市') & (pub_data['鄉鎮市區']=='信義區')
pub_data.loc[b,'鄉鎮市區']='信義區-基隆'
b=(pub_data['縣市']=='基隆市') & (pub_data['鄉鎮市區']=='中山區')
pub_data.loc[b,'鄉鎮市區']='中山區-基隆'

b=(pub_data['縣市']=='台中市') & (pub_data['鄉鎮市區']=='北區')
pub_data.loc[b,'鄉鎮市區']='北區-台中'
b=(pub_data['縣市']=='台中市') & (pub_data['鄉鎮市區']=='南區')
pub_data.loc[b,'鄉鎮市區']='南區-台中'
b=(pub_data['縣市']=='台中市') & (pub_data['鄉鎮市區']=='東區')
pub_data.loc[b,'鄉鎮市區']='東區-台中'
b=(pub_data['縣市']=='台中市') & (pub_data['鄉鎮市區']=='中區')
pub_data.loc[b,'鄉鎮市區']='中區-台中'
b=(pub_data['縣市']=='台中市') & (pub_data['鄉鎮市區']=='西區')
pub_data.loc[b,'鄉鎮市區']='西區-台中'
b=(pub_data['縣市']=='台中市') & (pub_data['鄉鎮市區']=='大安區')
pub_data.loc[b,'鄉鎮市區']='大安區-台中'

b=(pub_data['縣市']=='台南市') & (pub_data['鄉鎮市區']=='北區')
pub_data.loc[b,'鄉鎮市區']='北區-台南'
b=(pub_data['縣市']=='台南市') & (pub_data['鄉鎮市區']=='南區')
pub_data.loc[b,'鄉鎮市區']='南區-台南'
b=(pub_data['縣市']=='台南市') & (pub_data['鄉鎮市區']=='東區')
pub_data.loc[b,'鄉鎮市區']='東區-台南'

# a=pub_data[['縣市','鄉鎮市區']].drop_duplicates(keep='first')
# a[a.duplicated('鄉鎮市區')]


#%% 所得中位數、扶幼比、人口密度、平均消費傾向、教育結構

### 所得中位數
income = pd.read_excel('Data/external_data/110年度各縣市綜稅總所得中位數.xlsx', sheet_name = '行政區', index_col = '行政區')
income.drop(['縣市'], axis=1, inplace=True)
income_dict = income.to_dict()['綜合所得中位數(千元)']
tr_data = tr_data.copy() # 避免SettingWithCopyWarning
pub_data = pub_data.copy()
tr_data['綜合所得(中位數)'] = tr_data['鄉鎮市區'].map(income_dict)
pub_data['綜合所得(中位數)'] = pub_data['鄉鎮市區'].map(income_dict)

### 扶幼比
# care = pd.read_excel('Data/external_data/111年度各縣市扶幼比.xlsx', index_col = '縣市')
# care_dict = care.to_dict()['扶幼比']
# tr_data = tr_data.copy()
# pub_data = pub_data.copy()
# tr_data['扶幼比'] = tr_data['縣市'].map(care_dict)
# pub_data['扶幼比'] = pub_data['縣市'].map(care_dict)

### 人口密度
density = pd.read_excel('Data/external_data/111年度各縣市人口密度.xlsx', sheet_name = '行政區', index_col = '行政區')
density.drop(['縣市'], axis=1, inplace=True)
density_dict = density.to_dict()['人口密度(人/平方公里)']
tr_data = tr_data.copy()
pub_data = pub_data.copy()
tr_data['人口密度(人/平方公里)'] = tr_data['鄉鎮市區'].map(density_dict)
pub_data['人口密度(人/平方公里)'] = pub_data['鄉鎮市區'].map(density_dict)

### 平均消費傾向
# comsump_level = pd.read_excel('Data/external_data/2022年平均消費傾向.xlsx', index_col = '縣市')
# comsump_level_dict = comsump_level.to_dict()['平均消費傾向']
# tr_data = tr_data.copy()
# pub_data = pub_data.copy()
# tr_data['平均消費傾向'] = tr_data['縣市'].map(comsump_level_dict)
# pub_data['平均消費傾向'] = pub_data['縣市'].map(comsump_level_dict)

### 教育結構
edu = pd.read_excel('Data/external_data/2022年15歲以上民間人口之教育程度結構-大專及以上.xlsx', index_col = '縣市')
edu_dict = edu.to_dict()['教育結構']
tr_data = tr_data.copy()
pub_data = pub_data.copy()
tr_data['教育結構'] = tr_data['縣市'].map(edu_dict)
pub_data['教育結構'] = pub_data['縣市'].map(edu_dict)

### 農漁牧就業人口比例、自有住宅率、遷入人口、每醫療院所服務面積
other = pd.read_excel('Data/external_data/農漁牧就業人口比例、自有住宅率、遷入人口、每醫療院所服務面積.xlsx', index_col = '縣市')
immigrant_dict = other.to_dict()['遷入人口數(人)']
agri_dict = other.to_dict()['就業者之行業結構-農林漁牧業(％)']
medicPer_dict = other.to_dict()['平均每一醫療院所服務面積(平方公里/所)']
# homeOwnerR_dict = other.to_dict()['自有住宅比率(％)']
### 遷入人口數
tr_data = tr_data.copy()
pub_data = pub_data.copy()
tr_data['遷入人口數'] = tr_data['縣市'].map(immigrant_dict)
pub_data['遷入人口數'] = pub_data['縣市'].map(immigrant_dict)
### 農林漁牧業就業比例
tr_data = tr_data.copy()
pub_data = pub_data.copy()
tr_data['農林漁牧業就業比例'] = tr_data['縣市'].map(agri_dict)
pub_data['農林漁牧業就業比例'] = pub_data['縣市'].map(agri_dict)
### 每醫療院所服務面積
tr_data = tr_data.copy()
pub_data = pub_data.copy()
tr_data['每醫療院所服務面積'] = tr_data['縣市'].map(medicPer_dict)
pub_data['每醫療院所服務面積'] = pub_data['縣市'].map(medicPer_dict)
### 自有住宅比率
# tr_data = tr_data.copy()
# pub_data = pub_data.copy()
# tr_data['自有住宅比率'] = tr_data['縣市'].map(homeOwnerR_dict)
# pub_data['自有住宅比率'] = pub_data['縣市'].map(homeOwnerR_dict)
# tem_data = tr_data.copy()

#%% twd97_to_lonlat
#https://tylerastro.medium.com/twd97-to-longitude-latitude-dde820d83405
def twd97_to_lonlat(x=174458.0,y=2525824.0):
    """
    Parameters
    ----------
    x : float
        TWD97 coord system. The default is 174458.0.
    y : float
        TWD97 coord system. The default is 2525824.0.
    Returns
    -------
    list
        [longitude, latitude]
    """
    
    a = 6378137
    b = 6356752.314245
    long_0 = 121 * math.pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0
    
    e = math.pow((1-math.pow(b, 2)/math.pow(a,2)), 0.5)
    
    x -= dx
    y -= dy
    
    M = y / k0
    
    mu = M / ( a*(1-math.pow(e, 2)/4 - 3*math.pow(e,4)/64 - 5 * math.pow(e, 6)/256))
    e1 = (1.0 - pow((1   - pow(e, 2)), 0.5)) / (1.0 +math.pow((1.0 -math.pow(e,2)), 0.5))
    
    j1 = 3*e1/2-27*math.pow(e1,3)/32
    j2 = 21 * math.pow(e1,2)/16 - 55 * math.pow(e1, 4)/32
    j3 = 151 * math.pow(e1, 3)/96
    j4 = 1097 * math.pow(e1, 4)/512
    
    fp = mu + j1 * math.sin(2*mu) + j2 * math.sin(4* mu) + j3 * math.sin(6*mu) + j4 * math.sin(8* mu)
    
    e2 = math.pow((e*a/b),2)
    c1 = math.pow(e2*math.cos(fp),2)
    t1 = math.pow(math.tan(fp),2)
    r1 = a * (1-math.pow(e,2)) / math.pow( (1-math.pow(e,2)* math.pow(math.sin(fp),2)), (3/2))
    n1 = a / math.pow((1-math.pow(e,2)*math.pow(math.sin(fp),2)),0.5)
    d = x / (n1*k0)
    
    q1 = n1* math.tan(fp) / r1
    q2 = math.pow(d,2)/2
    q3 = ( 5 + 3 * t1 + 10 * c1 - 4 * math.pow(c1,2) - 9 * e2 ) * math.pow(d,4)/24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * math.pow(t1,2) - 3 * math.pow(c1,2) - 252 * e2) * math.pow(d,6)/720
    lat = fp - q1 * (q2 - q3 + q4)
    
    
    q5 = d
    q6 = (1+2*t1+c1) * math.pow(d,3) / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * math.pow(c1,2) + 8 * e2 + 24 * math.pow(t1,2)) * math.pow(d,5) / 120
    lon = long_0 + (q5 - q6 + q7) / math.cos(fp)
    
    lat = (lat*180) / math.pi
    lon = (lon*180) / math.pi
    return [lat,lon] 

LatLng = []
for i in range(len(tr_data)):
    twd97 = tr_data.loc[:,['橫坐標','縱坐標']].values.tolist()[i]
    LatLng.append(twd97_to_lonlat(x=twd97[0],y=twd97[1]))
LatLng_pub = []
for i in range(len(pub_data)):
    twd97 = pub_data.loc[:,['橫坐標','縱坐標']].values.tolist()[i]
    LatLng_pub.append(twd97_to_lonlat(x=twd97[0],y=twd97[1]))
    
#% 計算距離
#到目的地最近距離
def getDistance(position, destination, location):
    dList = []
    for i in range(len(position)):
        d = []
        for j in range(len(destination)):
            d.append(haversine(tuple(position[i]),tuple(destination[j]), unit=Unit.KILOMETERS)) #公里
        df = pd.DataFrame(d)
        dList.append([location[df.idxmin()].values[0],df.min()[0]])
    dList = pd.DataFrame(dList, columns=[location.name,'距離'])
    return dList
distance_tr = pd.DataFrame(index=data.index)
distance_pub = pd.DataFrame(index=orignal_pub_data.index)
### MRT
mrt = pd.read_csv('Data/external_data/捷運站點資料+台中.csv')
mrt_LatLng = mrt.loc[:,['lat','lng']].values.tolist()
# training_dataset
mrt_d = getDistance(LatLng, mrt_LatLng, mrt['站點名稱'])
# distance_tr['到捷運站距離'] = mrt_d['距離'].values
mrt_d['是否有捷運站'] = [1 if x< 0.7 else 0 for x in mrt_d['距離']] #步行距離不超過10分鐘視為有捷運站
distance_tr['是否有捷運站'] = mrt_d['是否有捷運站'].values
# public_dataaet
mrt_d_pub = getDistance(LatLng_pub, mrt_LatLng, mrt['站點名稱'])
# distance_pub['到捷運站距離'] = mrt_d_pub['距離'].values
mrt_d_pub['是否有捷運站'] = [1 if x< 0.7 else 0 for x in mrt_d_pub['距離']] #步行距離不超過10分鐘視為有捷運站
distance_pub['是否有捷運站'] = mrt_d_pub['是否有捷運站'].values

### Train Station
train = pd.read_csv('Data/external_data/火車站點資料.csv')
train_LatLng = train.loc[:,['lat','lng']].values.tolist()
# training_dataset
train_d = getDistance(LatLng, train_LatLng, train['站點名稱'])
# distance_tr['到火車站距離'] = train_d['距離'].values
train_d['是否有火車站'] = [1 if x< 1 else 0 for x in train_d['距離']]
distance_tr['是否有火車站'] = train_d['是否有火車站'].values
train_d.index = tr_data.index
# public_dataset
train_pub = getDistance(LatLng_pub, train_LatLng, train['站點名稱'])
# distance_pub['到火車站距離'] = train_pub['距離'].values
train_pub['是否有火車站'] = [1 if x< 1 else 0 for x in train_pub['距離']]
distance_pub['是否有火車站'] = train_pub['是否有火車站'].values

### 國中國小 (物件到學校的距離)
elemSchool = pd.read_csv('Data/external_data/國小基本資料.csv')
junSchool = pd.read_csv('Data/external_data/國中基本資料.csv')
elemSchool_LatLng = elemSchool.loc[:,['學校名稱','lat','lng']]
junSchool_LatLng = junSchool.loc[:,['學校名稱','lat','lng']]
school = pd.concat([elemSchool_LatLng,junSchool_LatLng], ignore_index = True)
school_LatLng = school.loc[:,['lat','lng']].values.tolist()
# training_dataset
school_d = getDistance(LatLng, school_LatLng, school['學校名稱'])
school_d.index = tr_data.index
# school_d['是否有國中小'] = [1 if x< 1.2 else 0 for x in school_d['距離']] #不準確 學區房定義按里分
# distance_tr['是否有國中小'] = school_d['是否有國中小'].values
distance_tr['到學校距離'] = school_d['距離'].values
# public_dataset
school_d_pub = getDistance(LatLng_pub, school_LatLng, school['學校名稱'])
# school_d_pub['是否有國中小'] = [1 if x< 1.2 else 0 for x in school_d_pub['距離']]
# distance_pub['是否有國中小'] = school_d_pub['是否有國中小'].values
distance_pub['到學校距離'] = school_d_pub['距離'].values

### 金融機構
bank = pd.read_csv('Data/external_data/金融機構基本資料.csv')
postOf = pd.read_csv('Data/external_data/郵局據點資料.csv')
bank_LatLng = bank.loc[:,['局名','lat','lng']]
postOf_LatLng = postOf.loc[:,['局名','lat','lng']]
finaInsitu = pd.concat([bank_LatLng,postOf_LatLng], ignore_index = True)
finaInsitu_LatLng = finaInsitu.loc[:,['lat','lng']].values.tolist()
# training_dataset
finaInsitu_d =  getDistance(LatLng, finaInsitu_LatLng, finaInsitu['局名'])
finaInsitu_d.index = tr_data.index
# finaInsitu_d['是否有金融機構'] = [1 if x< 1 else 0 for x in finaInsitu_d['距離']] #不準確
# distance_tr['是否有金融機構'] = finaInsitu_d['是否有金融機構'].values
distance_tr['到金融機構距離'] = finaInsitu_d['距離'].values
# public_dataset
finaInsitu_d_pub =  getDistance(LatLng_pub, finaInsitu_LatLng, finaInsitu['金融機構名稱'])
# finaInsitu_d_pub['是否有金融機構'] = [1 if x< 1 else 0 for x in finaInsitu_d_pub['距離']] 
# distance_pub['是否有金融機構'] = finaInsitu_d_pub['是否有金融機構'].values
distance_pub['到金融機構距離'] = finaInsitu_d_pub['距離'].values

### 醫療機構
medical = pd.read_csv('Data/external_data/醫療機構基本資料.csv')
medical_LatLng = medical.loc[:,['lat','lng']].values.tolist()
# training_dataset
medical_d = getDistance(LatLng, medical_LatLng, medical['機構名稱'])
# medical_d['是否有醫療機構'] = [1 if x< 1 else 0 for x in medical_d['距離']] 
# distance_tr['是否有醫療機構'] = medical_d['是否有醫療機構'].values
distance_tr['到醫療機構距離'] = medical_d['距離'].values
# public_dataset
medical_d_pub = getDistance(LatLng_pub, medical_LatLng, medical['機構名稱'])
# medical_d_pub['是否有醫療機構'] = [1 if x< 1 else 0 for x in medical_d_pub['距離']] 
# distance_pub['是否有醫療機構'] = medical_d_pub['是否有醫療機構'].values
distance_pub['到醫療機構距離'] = medical_d_pub['距離'].values

distance_tr.to_csv('distance_tr_NEW.csv')
distance_pub.to_csv('distance_pub_NEW.csv')


#%% Label encoding
labelencoder = LabelEncoder()
tr_data['主要用途'] = labelencoder.fit_transform(tr_data['主要用途'])
pub_data['主要用途'] = labelencoder.transform(pub_data['主要用途'])
labelencoder = LabelEncoder()
tr_data['主要建材'] = labelencoder.fit_transform(tr_data['主要建材'])
pub_data['主要建材'] = labelencoder.transform(pub_data['主要建材'])
labelencoder = LabelEncoder()
tr_data['建物型態'] = labelencoder.fit_transform(tr_data['建物型態'])
pub_data['建物型態'] = labelencoder.transform(pub_data['建物型態'])
#%% removeOutlier
def removeOutlierFromCate(col):
    item = col.unique() #每個類別
    new_trdata = pd.DataFrame()
    del_trdata = pd.DataFrame()
    for i in item:
        df = tr_data[col == i ]
        IQR = df['單價'].quantile(0.75) - df['單價'].quantile(0.25)
        Lower_quantile_lower = df['單價'].quantile(0.25) - (IQR * 1.5)
        Upper_quantile_lower = df['單價'].quantile(0.75) + (IQR * 1.5)
        outier = df[(df['單價'] < Lower_quantile_lower)|(df['單價'] > Upper_quantile_lower)]
        no_outier = df[(df['單價'] > Lower_quantile_lower)&(df['單價'] < Upper_quantile_lower)]
        new_trdata = pd.concat([new_trdata,no_outier])
        del_trdata = pd.concat([del_trdata,outier])
    return new_trdata , del_trdata

tr_data, a = removeOutlierFromCate(tr_data['縣市'])
# tr_data, b = removeOutlierFromCate(tr_data['鄉鎮市區'])


def removeOutlierFromCol(df, col):
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    Lower_quantile_lower = df[col].quantile(0.25) - (IQR * 1.5)
    Upper_quantile_lower = df[col].quantile(0.75) + (IQR * 1.5)
    no_outier = df[(df[col] > Lower_quantile_lower)&(df[col] < Upper_quantile_lower)]
    return no_outier
# tr_data = removeOutlierFromCol(tr_data, '土地面積')
# tr_data = removeOutlierFromCol(tr_data, '建物面積')
# tr_data = removeOutlierFromCol(tr_data, '主建物面積')
# tr_data = removeOutlierFromCol(tr_data, '陽台面積')
# tr_data = removeOutlierFromCol(tr_data, '附屬建物面積')
# tr_data = removeOutlierFromCol(tr_data, '單價')
#%% Targetencoding
targetencoder = TargetEncoder()
tr_data['縣市'] = targetencoder.fit_transform(tr_data['縣市'], tr_data['單價'])
pub_data['縣市'] = targetencoder.transform(pub_data['縣市'])
tr_data['鄉鎮市區'] = targetencoder.fit_transform(tr_data['鄉鎮市區'], tr_data['單價'])
pub_data['鄉鎮市區'] = targetencoder.transform(pub_data['鄉鎮市區'])

#%% Leave-One-Out Encoding  

looEncoder = LeaveOneOutEncoder(sigma=0.05)
tr_data['縣市'] = looEncoder.fit_transform(tr_data['縣市'], tr_data['單價'])
pub_data['縣市'] = looEncoder.transform(pub_data['縣市'])
tr_data['鄉鎮市區'] = looEncoder.fit_transform(tr_data['鄉鎮市區'], tr_data['單價'])
pub_data['鄉鎮市區'] = looEncoder.transform(pub_data['鄉鎮市區'])
tr_data['主要用途'] = looEncoder.fit_transform(tr_data['主要用途'], tr_data['單價'])
pub_data['主要用途'] = looEncoder.transform(pub_data['主要用途'])
tr_data['主要建材'] = looEncoder.fit_transform(tr_data['主要建材'], tr_data['單價'])
pub_data['主要建材'] = looEncoder.transform(pub_data['主要建材'])
tr_data['建物型態'] = looEncoder.fit_transform(tr_data['建物型態'], tr_data['單價'])
pub_data['建物型態'] = looEncoder.transform(pub_data['建物型態'])

#%% Beta Target Encoding
class BetaEncoder(object):
       
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha = alpha_prior + n
        beta =  beta_prior + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
encoding_col = ['縣市','鄉鎮市區','主要用途','主要建材','建物型態']
# fit encoder
N_min = 1000

for col in encoding_col:
    beta = BetaEncoder(col)
    beta.fit(tr_data, '單價')
    tr_data[col] = beta.transform(tr_data, 'median', N_min).values #用中位數encoding
    pub_data[col] = beta.transform(pub_data, 'median', N_min).values 

#%%
tr_data.corr()['單價']
plot_data = tr_data
ncols = 6
nrows = 4
# 分佈圖
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in plot_data.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 顯示中文
plt.rcParams['axes.unicode_minus'] = False 
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# boxplot
fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in plot_data.items():
    sns.boxplot(y=k, data=tr_data, ax=axs[index])
    index += 1
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#相關係數
# tr_data.info()
plt.figure(figsize=(16, 6))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False 
heatmap = sns.heatmap(plot_data.corr(), vmin=-1, vmax=1, annot=True)
# 所有feature的散點圖
sns.pairplot(plot_data, height = 2.5)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.show()
#%%
#刪除相關係數<0.5的變數  ['總樓層數', '車位個數', '陽台面積', '橫坐標', '縱坐標']
col_num = list(tr_data.corr()['單價'][tr_data.corr()['單價'].abs() <= 0.05].index) + ['橫坐標','縱坐標']
tr_data.drop(col_num, axis=1, inplace=True) #,'橫坐標','縱坐標'
pub_data.drop(col_num, axis=1, inplace=True)

#%%

# pub_data['建物型態'][pub_data['建物型態'].isnull()]
# pub_data.isnull().sum()
tr_data.to_csv('TrDatasetAfterPreces.csv')
pub_data.to_csv('PubDatasetAfterPreces.csv')


#%%

# data = pd.read_csv('Data/training_data.csv',index_col=0)
tr_data = pd.read_csv('Data/TrDatasetAfterPreces.csv',index_col=0)
pub_data = pd.read_csv('Data/PubDatasetAfterPreces.csv',index_col=0)

# 單價分佈圖、boxplot
fig = plt.figure(figsize=(16, 5))
fig.add_subplot(2,1,1)
sns.distplot(tr_data['單價'],kde=True)  #刪除密度曲线，kde=False
fig.add_subplot(2,1,2)
sns.boxplot(tr_data['單價'],orient='h')
plt.tight_layout()
# # 所有feature的散點圖
# sns.pairplot(tr_data, height = 2.5)
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
# plt.rcParams['axes.unicode_minus'] = False 
# plt.show()
# # 


