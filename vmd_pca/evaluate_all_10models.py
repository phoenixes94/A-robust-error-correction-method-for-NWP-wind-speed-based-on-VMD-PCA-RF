import pandas as pd
import os
import time
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from matplotlib import colors
import matplotlib as mpl
import time  
from sklearn import metrics
start = time.time()

ds=pd.read_csv('/ws_correct_ML/vmd_pca/data_410sitetrain1day_new_all.csv')#去除时间点'2022-07-28 09:00:00' vmd无法处理长序列
print(ds[ds.isnull().T.any()])#找到有空值的行
print(np.isfinite(ds.all()))
print(np.isinf(ds.all()))
ds=ds.drop(columns=['Unnamed: 0'])
ds['time'] = pd.to_datetime(ds['time'])
ds = ds.set_index('time')
ds_test=ds[['海拔','WIN_pre','风速']]
ds_test=ds_test.drop(ds_test['2022-02'].index.unique())
for i in ['lgb','xgb','rf','dbn','mlp']:
  ds_DS=pd.read_csv('/ws_correct_ML/vmd_pca/testand{:}_all.csv'.format(i))#testanddbn #testandrf
  ds_DS['time'] = pd.to_datetime(ds_DS['time'])
  ds_DS = ds_DS.set_index('time')
  ds_DS=ds_DS.drop(ds_DS['2022-02'].index.unique())
  ds_test['vp_{:}'.format(i)]=ds_DS['{}'.format(i)].values
for i in ['lgb','xgb','rf','dbn','mlp']:
  ds_DS=pd.read_csv('/ws_correct_ML/machinelearning/testand{:}_all.csv'.format(i))#testanddbn #testandrf
  ds_DS['time'] = pd.to_datetime(ds_DS['time'])
  ds_DS = ds_DS.set_index('time')
  ds_DS=ds_DS.drop(ds_DS['2022-02'].index.unique())
  ds_DS=ds_DS.drop(ds_DS.loc['2022-07-28 09:00:00',:].index)#去除时间点'2022-07-28 09:00:00' vmd无法处理长序列   ####原始多处理了一个时间点
  ds_test['{:}'.format(i)]=ds_DS['{}'.format(i)].values
ds_test.to_csv('/ws_correct_ML/vmd_pca/all-10models.csv')
ds_test=pd.read_csv('/ws_correct_ML/vmd_pca/all-10models.csv')
ds_test['time'] = pd.to_datetime(ds_test['time'])
ds_test = ds_test.set_index('time')
# #计算各种指标
def RMSE(obs, pre):
    """
    Root mean squared error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: root mean squared error between observed and simulated values
    """
    # obs = obs.flatten()
    # pre = pre.flatten()
    return np.sqrt(np.mean((obs - pre) ** 2,axis=0))
def MAE(obs, pre):
    """
    Mean absolute error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: mean absolute error between observed and simulated values
    """
    obs = obs.flatten()
    pre = pre.flatten()
    return np.mean(np.abs(pre - obs),axis=0)
def FA_axis0(obs, pre):
    """
    风速预报准确率
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: mean absolute error between observed and simulated values
    """
    # obs = obs.flatten()
    # pre = pre.flatten()
    abs_win=np.abs(obs-pre)
    abs_win_r=np.where(abs_win <=1 , 1, 0)
    Nr=np.sum(abs_win_r==1,axis=0)
    return Nr/obs.shape[0]
def rMAE(obs, pre):
    """
    Mean absolute error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: mean absolute error between observed and simulated values
    """
    obs = obs.flatten()
    pre = pre.flatten()
    return np.mean(np.abs(pre - obs),axis=0)/np.mean(obs)
def rRMSE(obs, pre):
    """
    Mean absolute error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: mean absolute error between observed and simulated values
    """
    obs = obs.flatten()
    pre = pre.flatten()
    return np.sqrt(np.mean((obs - pre) ** 2,axis=0))/np.mean(obs)
for i in ['vp_lgb', 'vp_xgb', 'vp_rf', 'vp_dbn', 'vp_mlp','lgb', 'xgb', 'rf',  'dbn', 'mlp','WIN_pre']:
 print(i)
 Df_5=[]
 for month in ['2021-09','2021-10','2021-11','2021-12','2022-01','2022-03','2022-04','2022-05','2022-06','2022-07']:
        Ytrain1,Ytrain_predict=ds_test[month].loc[:,'风速'].values,ds_test[month].loc[:,i].values
        print(month)
        print('MAE=%.2f'%(MAE(Ytrain1,Ytrain_predict)))
        print('RMSE=%.2f'%(RMSE(Ytrain1,Ytrain_predict)))
        print('rMAE=%.2f'%(rMAE(Ytrain1,Ytrain_predict)))
        print('rRMSE=%.2f'%(rRMSE(Ytrain1,Ytrain_predict)))
        print('FA=%.2f'%(FA_axis0(Ytrain1,Ytrain_predict)))
        print('R=%.2f'%(np.corrcoef(Ytrain1,Ytrain_predict)[0][1]))
        df=pd.DataFrame(np.round(np.array([MAE(Ytrain1,Ytrain_predict),
            RMSE(Ytrain1,Ytrain_predict),
            rMAE(Ytrain1,Ytrain_predict)*100,
            rRMSE(Ytrain1,Ytrain_predict)*100,
            FA_axis0(Ytrain1,Ytrain_predict)*100,
            np.corrcoef(Ytrain1,Ytrain_predict)[0][1]]).reshape(1,-1),2),columns=['MAE（m/s）',' RMSE（m/s）','rMAE（%）','rRMSE（%）','FA（%）','R'],index=[month])
        Df_5.append(df)
 df_5=pd.concat(Df_5,axis=0)
 df_5.to_csv('/ws_correct_ML/vmd_pca/evaluate_all_10models/test{}_5evaluate.csv'.format(i))