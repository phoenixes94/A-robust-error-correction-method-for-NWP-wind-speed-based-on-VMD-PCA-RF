import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from bayes_opt import BayesianOptimization
import seaborn as sns
import os
from datetime import datetime,timedelta
import re
import numpy as np
import seaborn as sns
import matplotlib as mpl
from scipy.stats import gaussian_kde
from matplotlib import colors 
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import KFold 
import time
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV   
from sklearn.neural_network import MLPRegressor
from DBN import DBN
import torch
from vmdpy import VMD 
from sklearn.decomposition import PCA
import joblib
import lightgbm as lgb
from xgboost import XGBRegressor
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

ds=pd.read_csv('/ws_correct_ML/machinelearning/data_410sitetrain1day.csv')#全新的春夏秋冬
ds=ds.loc[(ds.loc[:,'time']!='2022-07-28 09:00:00'),:]#去除一个时间点 vmd无法处理长序列
#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.0244            # noise-tolerance (no strict fidelity enforcement)  
K = 9             # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  
for i in range(ds['站点号'].unique().shape[0]):
    print(i)
    f=ds.loc[ds['站点号']==ds.loc[i,'站点号'],'WIN_pre']
    #. Run VMD 
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF0']=u[0,:]
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF1']=u[1,:]
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF2']=u[2,:]
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF3']=u[3,:] 
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF4']=u[4,:]
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF5']=u[5,:]
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF6']=u[6,:]
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF7']=u[7,:]
    ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF8']=u[8,:]
print(ds[ds.isnull().T.any()])#找到有空值的行
print(np.isfinite(ds.all()))
print(np.isinf(ds.all()))
ds=ds.drop(columns=['Unnamed: 0', '站点号','纬度', '经度','降水', '气压','湿度', '温度','time.1'])
ds=ds.dropna(subset=(ds.columns))
ds.to_csv('/ws_correct_ML/vmd_pca/data_410sitetrain1day_new_all.csv')



ds=pd.read_csv('/ws_correct_ML/vmd_pca/data_410sitetrain1day_new_all.csv')
ds=ds.drop(columns=['Unnamed: 0'])
def pca(data,num):
    p=PCA(n_components=num)
    p.fit(data)
    # data_pca=p.transform(data)
    print(p.explained_variance_ratio_)#[0.54420209 0.24125721 0.11476376]
    return p
Nan=ds[ds.isnull().T.any()]#有空值的行输出
ds_des=ds.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T


ds['time'] = pd.to_datetime(ds['time'])
ds = ds.set_index('time')
# ds1=ds['2021-09-01':'2022-03-07']
p1=pca(ds.loc[:,['IMF0', 'IMF1', 'IMF2', 'IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7', 'IMF8']],0.8)
ds_pca=p1.transform(ds.loc[:,['IMF0', 'IMF1', 'IMF2', 'IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7', 'IMF8']])
ds.loc[:,'pca0']=ds_pca[:,0]
ds.loc[:,'pca1']=ds_pca[:,1]
ds.loc[:,'pca2']=ds_pca[:,2]
ds=ds.drop(columns=['D2_pre', 'DIR10_pre',  'U10_pre', 'V10_pre'])
ds.columns=[ 'hgt', 'WIN_obs', 'PRE_pre', 'PRS_pre', 'RHU_pre','TMP_pre', 'WIN_pre', 'lon', 'lat','IMF0', 'IMF1', 'IMF2','IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7', 'IMF8','pca0', 'pca1', 'pca2']
# corr=ds.corr()
# corr.to_csv('/ws_correct_ML/vmd_pca/corr.csv')
# ds_train=ds['2021-09':'2022-02']
ds_train=ds['2022-02']
# print(ds_train)
X = ds_train.drop(columns =['WIN_obs'])
Y = ds_train['WIN_obs']
X.isnull().mean()#探索缺失值
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.1,random_state=2021)#分训练集和测试集
des_xtr=Xtrain.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
des_xte=Xtest.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
cate = Xtrain.columns[Xtrain.dtypes == "object"].tolist()#数值类型，这里都是数值
col = Xtrain.columns.tolist()
ss = StandardScaler()#每一列数据都做了标准化，为了便于训练
ss = ss.fit(Xtrain.loc[:,col])#每一列都拟合
Xtrain.loc[:,col] = ss.transform(Xtrain.loc[:,col])#训练集的标准化
Xtest.loc[:,col] = ss.transform(Xtest.loc[:,col])#测试集的标准化
Xtrain.values.dtype
# print(np.isfinite(Xtrain.all()))
# print(Xtrain[Xtrain.isnull().T.any()])
ss1 = StandardScaler()#把y进行标准化
ss1=ss1.fit(Ytrain.values.reshape(-1,1))
Ytrain=ss1.transform(Ytrain.values.reshape(-1,1))
Ytest=ss1.transform(Ytest.values.reshape(-1,1))
Ytrain1=ss1.inverse_transform(Ytrain)
Ytest1=ss1.inverse_transform(Ytest)






#以下是dbn-rbm交叉验证的函数
input_length = 20
output_length = 1
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam
# network
hidden_units = [400, 200]
# # train & predict
batch_size = 20000
epoch_pretrain = 100
epoch_finetune = 200
# Build model

dbn = DBN(hidden_units, input_length, output_length)

# Train model
dbn.pretrain(Xtrain.values, epoch=epoch_pretrain, batch_size=batch_size)
dbn.finetune(Xtrain.values, Ytrain.reshape(-1,1), epoch_finetune, batch_size, loss_function,
             optimizer(dbn.parameters()))
model=dbn
with open('/ws_correct_ML/vmd_pca/dbnmodel.pickle', 'wb') as f:#
    pickle.dump(model, f)
    print("保存模型成功")
#模型保存后，从此处开始运行
with open('/ws_correct_ML/vmd_pca/dbnmodel.pickle', 'rb') as f:
    model = pickle.load(f)
print("加载DBN模型成功")
########rf
file = open(r'/ws_correct_ML/vmd_pca/rf_parameter_win.txt', mode='r').read()
model = RandomForestRegressor(n_estimators=int(eval(file)['params']['n_estimators']),
 max_features=int(eval(file)['params']['max_features']), 
 max_depth=int(eval(file)['params']['max_depth']),
 min_samples_split=int(eval(file)['params']['min_samples_split']),
 min_samples_leaf=int(eval(file)['params']['min_samples_leaf']),random_state=2)



model.fit(Xtrain, Ytrain.ravel())#dbn需要注释,其他模型不需要

with open('/ws_correct_ML/vmd_pca/rfmodel.pickle', 'wb+') as f:#
    pickle.dump(model, f)
    print("保存模型成功")
#####################lgb
file = open(r'/ws_correct_ML/vmd_pca/lightgbm_parameter_win.txt', mode='r').read()
model = lgb.LGBMRegressor(max_depth=int(eval(file)['params']['max_depth']),
                        n_estimators=int(eval(file)['params']['n_estimators']),
                        num_leaves=int(eval(file)['params']['num_leaves']),
                        min_child_samples=int(eval(file)['params']['min_child_samples']),
                        random_state=2)
model.fit(Xtrain, Ytrain.ravel())#dbn需要注释,其他模型不需要
with open('/ws_correct_ML/vmd_pca/lgbmodel.pickle', 'wb+') as f:#
    pickle.dump(model, f)
    print("保存模型成功")
###########################xgb
file = open(r'/ws_correct_ML/vmd_pca/xgb_parameter_win.txt', mode='r').read()

model = XGBRegressor(max_depth=int(eval(file)['params']['max_depth']),
                        # learning_rate=rf_bo.res[max_index]['params']['learning_rate'],
                        n_estimators=int(eval(file)['params']['n_estimators']),
                        # objective='reg:linear', # 此默认参数与 XGBClassifier 不同
                        # booster='gbtree',
                        gamma=eval(file)['params']['gamma'],
                        min_child_weight=eval(file)['params']['min_child_weight'],
                        # subsample=rf_bo.res[max_index]['params']['subsample'],
                        # colsample_bytree=rf_bo.res[max_index]['params']['colsample_bytree'],
                        # reg_alpha=0,
                        # reg_lambda=1,
                        random_state=2)
model.fit(Xtrain, Ytrain.ravel())#dbn需要注释,其他模型不需要
with open('/ws_correct_ML/vmd_pca/xgbmodel.pickle', 'wb+') as f:#
    pickle.dump(model, f)
    print("保存模型成功")
###############################
###############################mlp
file = open(r'/ws_correct_ML/vmd_pca/mlp_parameter_win.txt', mode='r').read()
model=MLPRegressor(hidden_layer_sizes=(int(eval(file)['params']['hidden_layer_sizes']),int(eval(file)['params']['hidden_layer_sizes'])),
    batch_size=int(eval(file)['params']['batch_size']),
    max_iter=int(eval(file)['params']['max_iter']),
    early_stopping=True,
    random_state=2)
model.fit(Xtrain, Ytrain.ravel())#dbn需要注释,其他模型不需要
with open('/ws_correct_ML/vmd_pca/mlpmodel.pickle', 'wb+') as f:#
    pickle.dump(model, f)
    print("保存模型成功")
######################################
ds_DS1=[]
for month in ['2021-09','2021-10','2021-11','2021-12','2022-01','2022-03','2022-04','2022-05','2022-06','2022-07']:
    # print(month)
    ds_test=ds[month]
    X_DS = ds_test.drop(columns =['WIN_obs'])
    Y_DS = ds_test['WIN_obs']
    X_DS.loc[:,col] = ss.transform(X_DS.loc[:,col])
    Y_DS=ss1.transform(Y_DS.values.reshape(-1,1))
    Y_DS1=ss1.inverse_transform(Y_DS)#反转
    # Y_DS_predict = model.predict(X_DS)#RF、MLP ETC.
    Y_DS_predict = model.predict(X_DS.values,20000).reshape(-1, 1)#dbn
    Y_DS_predict= ss1.inverse_transform(Y_DS_predict)
    ds_DS=pd.DataFrame(np.concatenate((Y_DS1, Y_DS_predict),axis=1),columns=['WIN_obs', 'dbn'])
    ds_DS.index=ds_test.index
    ds_DS1.append(ds_DS)
ds_DS2=pd.concat(ds_DS1,axis=0)
ds_DS2.to_csv('/ws_correct_ML/vmd_pca/testanddbn_all.csv')

Df_5=[]
for month in ['2021-09','2021-10','2021-11','2021-12','2022-01','2022-03','2022-04','2022-05','2022-06','2022-07']:
    # print(month)
    ds_test=ds[month]
    X_DS = ds_test.drop(columns =['WIN_obs'])
    Y_DS = ds_test['WIN_obs']
    X_DS.loc[:,col] = ss.transform(X_DS.loc[:,col])
    Y_DS=ss1.transform(Y_DS.values.reshape(-1,1))
    Y_DS1=ss1.inverse_transform(Y_DS)#反转
    # Y_DS_predict = model.predict(X_DS)
    Y_DS_predict = model.predict(X_DS.values,20000).reshape(-1, 1)#dbn
    Y_DS_predict= ss1.inverse_transform(Y_DS_predict)
    print(month)
    print('MAE=%.2f'%(MAE(Y_DS1, Y_DS_predict)))
    print('RMSE=%.2f'%(RMSE(Y_DS1, Y_DS_predict)))
    print('rMAE=%.2f'%(rMAE(Y_DS1, Y_DS_predict)))
    print('rRMSE=%.2f'%(rRMSE(Y_DS1, Y_DS_predict)))
    print('FA=%.2f'%(FA_axis0(Y_DS1, Y_DS_predict)))
    df=pd.DataFrame(np.round(np.array([MAE(Y_DS1, Y_DS_predict),
        RMSE(Y_DS1, Y_DS_predict)[0],
        rMAE(Y_DS1, Y_DS_predict)*100,
        rRMSE(Y_DS1, Y_DS_predict)*100,
        FA_axis0(Y_DS1, Y_DS_predict)[0]*100]).reshape(1,-1),2),columns=['MAE（m/s）',' RMSE（m/s）','rMAE（%）','rRMSE（%）','FA（%）'],index=[month])
    Df_5.append(df)
df_5=pd.concat(Df_5,axis=0)
df_5.to_csv('/ws_correct_ML/vmd_pca/testdbn_5evaluate_all.csv')



for name in ['rf','mlp','lgb','xgb']:#'rf','mlp','lgb','xgb',
    with open('/ws_correct_ML/vmd_pca/{:}model.pickle'.format(name), 'rb') as f:
       model = pickle.load(f)
    print("加载{:}模型成功".format(model))
    ds_DS1=[]
    for month in ['2021-09','2021-10','2021-11','2021-12','2022-01','2022-03','2022-04','2022-05','2022-06','2022-07']:
        # print(month)
        ds_test=ds[month]
        X_DS = ds_test.drop(columns =['WIN_obs'])
        Y_DS = ds_test['WIN_obs']
        X_DS.loc[:,col] = ss.transform(X_DS.loc[:,col])
        Y_DS=ss1.transform(Y_DS.values.reshape(-1,1))
        Y_DS1=ss1.inverse_transform(Y_DS)#反转
        Y_DS_predict = model.predict(X_DS).reshape(-1,1)#RF、MLP ETC.
        Y_DS_predict= ss1.inverse_transform(Y_DS_predict)
        ds_DS=pd.DataFrame(np.concatenate((Y_DS1, Y_DS_predict),axis=1),columns=['WIN_obs', '{}'.format(name)])
        ds_DS.index=ds_test.index
        ds_DS1.append(ds_DS)
    ds_DS2=pd.concat(ds_DS1,axis=0)
    ds_DS2.to_csv('/ws_correct_ML/vmd_pca/testand{:}_all.csv'.format(name))
    Df_5=[]
    for month in ['2021-09','2021-10','2021-11','2021-12','2022-01','2022-03','2022-04','2022-05','2022-06','2022-07']:
        # print(month)
        ds_test=ds[month]
        X_DS = ds_test.drop(columns =['WIN_obs'])
        Y_DS = ds_test['WIN_obs']
        X_DS.loc[:,col] = ss.transform(X_DS.loc[:,col])
        Y_DS=ss1.transform(Y_DS.values.reshape(-1,1))
        Y_DS1=ss1.inverse_transform(Y_DS)#反转
        Y_DS_predict = model.predict(X_DS).reshape(-1,1)#RF、MLP ETC.
        Y_DS_predict= ss1.inverse_transform(Y_DS_predict)
        print(month)
        print('MAE=%.2f'%(MAE(Y_DS1, Y_DS_predict)))
        print('RMSE=%.2f'%(RMSE(Y_DS1, Y_DS_predict)))
        print('rMAE=%.2f'%(rMAE(Y_DS1, Y_DS_predict)))
        print('rRMSE=%.2f'%(rRMSE(Y_DS1, Y_DS_predict)))
        print('FA=%.2f'%(FA_axis0(Y_DS1, Y_DS_predict)))
        df=pd.DataFrame(np.round(np.array([MAE(Y_DS1, Y_DS_predict),
            RMSE(Y_DS1, Y_DS_predict)[0],
            rMAE(Y_DS1, Y_DS_predict)*100,
            rRMSE(Y_DS1, Y_DS_predict)*100,
            FA_axis0(Y_DS1, Y_DS_predict)[0]*100]).reshape(1,-1),2),columns=['MAE（m/s）',' RMSE（m/s）','rMAE（%）','rRMSE（%）','FA（%）'],index=[month])
        Df_5.append(df)
    df_5=pd.concat(Df_5,axis=0)
    df_5.to_csv('/ws_correct_ML/vmd_pca/test{}_5evaluate_all.csv'.format(name))


###########################plot
def RMSE(obs, pre):
    """
    Root mean squared error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: root mean squared error between observed and simulated values
    """
    obs = np.array(obs).flatten()
    pre = np.array(pre).flatten()
    return np.sqrt(np.mean((obs - pre) ** 2,axis=0))
def FA(obs, pre):
    """
    风速预报准确率
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: mean absolute error between observed and simulated values
    """
    obs = obs.ravel()
    pre = pre.ravel()
    abs_win=np.abs(obs-pre)
    abs_win_r=np.where(abs_win <=1 , 1, 0)
    Nr=np.sum(abs_win_r==1)
    return Nr/len(obs)
def dcmap_24():
  file_path='/ws_correct_ML/circular_2.txt'
  fid=open(file_path)
  data=fid.readlines()
  n=len(data);
  rgb=np.zeros((n,3))
  for i in np.arange(n):
            rgb[i][0]=data[i].split(',')[0]
            rgb[i][1]=data[i].split(',')[1]
            rgb[i][2]=data[i].split(',')[2]
  rgb=rgb/255.0
  rgb = [[round(j,2) for j in rgb[i]] for i in range(len(rgb))]
  icmap=colors.ListedColormap(rgb,name='my_color')
  return icmap
from scipy import optimize as op
plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
plt.rcParams['figure.dpi'] = 300 
plt.rc('font', family='Times New Roman')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.size'] = '20'
def plotreg(ds_test0,ds_DS0,ds_test1,ds_DS1,ds_test2,ds_DS2,ds_test3,ds_DS3,ds_test4,ds_DS4,ds_test5,ds_DS5,ds_test6,ds_DS6,ds_test7,ds_DS7,label,dir_save):
 fig,axes=plt.subplots(8,2,figsize=(22,80))
  # p1, p0 = np.polyfit(Ytrain1, Ytrain_predict, deg=1)  # slope, intercept
 def f_1(x, A):
    return A * x 
 # 得到返回的A，B值
 m=0
 n=0
 x0,y0=ds_test0[ds_test0.index.hour==0]['WIN_obs'].values,ds_test0[ds_test0.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test0[ds_test0.index.hour==j]['WIN_obs'].values,ds_test0[ds_test0.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest0=ds_test0['WIN_obs'].values
 Ytest_predict0=ds_test0['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest0, Ytest_predict0)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest0, Ytest_predict0)[0][1]:.2f}, N={len(Ytest0):d}\n'
        f'RMSE={RMSE(Ytest0, Ytest_predict0):.2f}, FA={FA(Ytest0, Ytest_predict0):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(c)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=0
 n=1
 x0,y0=ds_DS0[ds_DS0.index.hour==0]['Y_DS1'].values,ds_DS0[ds_DS0.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS0[ds_DS0.index.hour==j]['Y_DS1'].values,ds_DS0[ds_DS0.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS0=ds_DS0['Y_DS1'].values
 YDS_predict0=ds_DS0['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS0, YDS_predict0)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS0, YDS_predict0)[0][1]:.2f}, N={len(YDS0):d}\n'
        f'RMSE={RMSE(YDS0, YDS_predict0):.2f}, FA={FA(YDS0, YDS_predict0):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(d)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=1
 n=0
 x0,y0=ds_test1[ds_test1.index.hour==0]['WIN_obs'].values,ds_test1[ds_test1.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test1[ds_test1.index.hour==j]['WIN_obs'].values,ds_test1[ds_test1.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest1=ds_test1['WIN_obs'].values
 Ytest_predict1=ds_test1['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest1, Ytest_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest1, Ytest_predict1)[0][1]:.2f}, N={len(Ytest1):d}\n'
        f'RMSE={RMSE(Ytest1, Ytest_predict1):.2f}, FA={FA(Ytest1, Ytest_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=1
 n=1
 x0,y0=ds_DS1[ds_DS1.index.hour==0]['Y_DS1'].values,ds_DS1[ds_DS1.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS1[ds_DS1.index.hour==j]['Y_DS1'].values,ds_DS1[ds_DS1.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS1=ds_DS1['Y_DS1'].values
 YDS_predict1=ds_DS1['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS1, YDS_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS1, YDS_predict1)[0][1]:.2f}, N={len(YDS1):d}\n'
        f'RMSE={RMSE(YDS1, YDS_predict1):.2f}, FA={FA(YDS1, YDS_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=2
 n=0
 x0,y0=ds_test2[ds_test2.index.hour==0]['WIN_obs'].values,ds_test2[ds_test2.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test2[ds_test2.index.hour==j]['WIN_obs'].values,ds_test2[ds_test2.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest1=ds_test2['WIN_obs'].values
 Ytest_predict1=ds_test2['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest1, Ytest_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest1, Ytest_predict1)[0][1]:.2f}, N={len(Ytest1):d}\n'
        f'RMSE={RMSE(Ytest1, Ytest_predict1):.2f}, FA={FA(Ytest1, Ytest_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=2
 n=1
 x0,y0=ds_DS2[ds_DS2.index.hour==0]['Y_DS1'].values,ds_DS2[ds_DS2.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS2[ds_DS2.index.hour==j]['Y_DS1'].values,ds_DS2[ds_DS2.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS1=ds_DS2['Y_DS1'].values
 YDS_predict1=ds_DS2['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS1, YDS_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS1, YDS_predict1)[0][1]:.2f}, N={len(YDS1):d}\n'
        f'RMSE={RMSE(YDS1, YDS_predict1):.2f}, FA={FA(YDS1, YDS_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=3
 n=0
 x0,y0=ds_test3[ds_test3.index.hour==0]['WIN_obs'].values,ds_test3[ds_test3.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test3[ds_test3.index.hour==j]['WIN_obs'].values,ds_test3[ds_test3.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest1=ds_test3['WIN_obs'].values
 Ytest_predict1=ds_test3['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest1, Ytest_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest1, Ytest_predict1)[0][1]:.2f}, N={len(Ytest1):d}\n'
        f'RMSE={RMSE(Ytest1, Ytest_predict1):.2f}, FA={FA(Ytest1, Ytest_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=3
 n=1
 x0,y0=ds_DS3[ds_DS3.index.hour==0]['Y_DS1'].values,ds_DS3[ds_DS3.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS3[ds_DS3.index.hour==j]['Y_DS1'].values,ds_DS3[ds_DS3.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS1=ds_DS3['Y_DS1'].values
 YDS_predict1=ds_DS3['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS1, YDS_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS1, YDS_predict1)[0][1]:.2f}, N={len(YDS1):d}\n'
        f'RMSE={RMSE(YDS1, YDS_predict1):.2f}, FA={FA(YDS1, YDS_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=4
 n=0
 x0,y0=ds_test4[ds_test4.index.hour==0]['WIN_obs'].values,ds_test4[ds_test4.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test4[ds_test4.index.hour==j]['WIN_obs'].values,ds_test4[ds_test4.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest1=ds_test4['WIN_obs'].values
 Ytest_predict1=ds_test4['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest1, Ytest_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest1, Ytest_predict1)[0][1]:.2f}, N={len(Ytest1):d}\n'
        f'RMSE={RMSE(Ytest1, Ytest_predict1):.2f}, FA={FA(Ytest1, Ytest_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=4
 n=1
 x0,y0=ds_DS4[ds_DS4.index.hour==0]['Y_DS1'].values,ds_DS4[ds_DS4.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS4[ds_DS4.index.hour==j]['Y_DS1'].values,ds_DS4[ds_DS4.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS1=ds_DS4['Y_DS1'].values
 YDS_predict1=ds_DS4['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS1, YDS_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS1, YDS_predict1)[0][1]:.2f}, N={len(YDS1):d}\n'
        f'RMSE={RMSE(YDS1, YDS_predict1):.2f}, FA={FA(YDS1, YDS_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=5
 n=0
 x0,y0=ds_test5[ds_test5.index.hour==0]['WIN_obs'].values,ds_test5[ds_test5.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test5[ds_test5.index.hour==j]['WIN_obs'].values,ds_test5[ds_test5.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest1=ds_test5['WIN_obs'].values
 Ytest_predict1=ds_test5['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest1, Ytest_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest1, Ytest_predict1)[0][1]:.2f}, N={len(Ytest1):d}\n'
        f'RMSE={RMSE(Ytest1, Ytest_predict1):.2f}, FA={FA(Ytest1, Ytest_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=5
 n=1
 x0,y0=ds_DS5[ds_DS5.index.hour==0]['Y_DS1'].values,ds_DS5[ds_DS5.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS5[ds_DS5.index.hour==j]['Y_DS1'].values,ds_DS5[ds_DS5.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS1=ds_DS5['Y_DS1'].values
 YDS_predict1=ds_DS5['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS1, YDS_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS1, YDS_predict1)[0][1]:.2f}, N={len(YDS1):d}\n'
        f'RMSE={RMSE(YDS1, YDS_predict1):.2f}, FA={FA(YDS1, YDS_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=6
 n=0
 x0,y0=ds_test6[ds_test6.index.hour==0]['WIN_obs'].values,ds_test6[ds_test6.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test6[ds_test6.index.hour==j]['WIN_obs'].values,ds_test6[ds_test6.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest1=ds_test6['WIN_obs'].values
 Ytest_predict1=ds_test6['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest1, Ytest_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest1, Ytest_predict1)[0][1]:.2f}, N={len(Ytest1):d}\n'
        f'RMSE={RMSE(Ytest1, Ytest_predict1):.2f}, FA={FA(Ytest1, Ytest_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=6
 n=1
 x0,y0=ds_DS6[ds_DS6.index.hour==0]['Y_DS1'].values,ds_DS6[ds_DS6.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS6[ds_DS6.index.hour==j]['Y_DS1'].values,ds_DS6[ds_DS6.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS1=ds_DS6['Y_DS1'].values
 YDS_predict1=ds_DS6['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS1, YDS_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS1, YDS_predict1)[0][1]:.2f}, N={len(YDS1):d}\n'
        f'RMSE={RMSE(YDS1, YDS_predict1):.2f}, FA={FA(YDS1, YDS_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=7
 n=0
 x0,y0=ds_test7[ds_test7.index.hour==0]['WIN_obs'].values,ds_test7[ds_test7.index.hour==0]['WIN_pre'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_test7[ds_test7.index.hour==j]['WIN_obs'].values,ds_test7[ds_test7.index.hour==j]['WIN_pre'].values".format(j,j))
 Ytest1=ds_test7['WIN_obs'].values
 Ytest_predict1=ds_test7['WIN_pre'].values
 A = np.float(op.curve_fit(f_1, Ytest1, Ytest_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(Ytest1, Ytest_predict1)[0][1]:.2f}, N={len(Ytest1):d}\n'
        f'RMSE={RMSE(Ytest1, Ytest_predict1):.2f}, FA={FA(Ytest1, Ytest_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=7
 n=1
 x0,y0=ds_DS7[ds_DS7.index.hour==0]['Y_DS1'].values,ds_DS7[ds_DS7.index.hour==0]['Y_DS_predict'].values
 for j in range(1,24,1):
    print(j)
    exec("x{:},y{:}=ds_DS7[ds_DS7.index.hour==j]['Y_DS1'].values,ds_DS7[ds_DS7.index.hour==j]['Y_DS_predict'].values".format(j,j))
 YDS1=ds_DS7['Y_DS1'].values
 YDS_predict1=ds_DS7['Y_DS_predict'].values
 A = np.float(op.curve_fit(f_1, YDS1, YDS_predict1)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # p1, p0 = np.polyfit(YDS1, YDS_predict, deg=1)  # slope, intercept
 stats = (f'y = {A:.2f}x\n'
        f'R={np.corrcoef(YDS1, YDS_predict1)[0][1]:.2f}, N={len(YDS1):d}\n'
        f'RMSE={RMSE(YDS1, YDS_predict1):.2f}, FA={FA(YDS1, YDS_predict1):.2f}')
 # axes[m,n].plot(np.arange(0,16,0.01),p1*np.arange(0,16,0.01)+p0,color='r', lw=2)
 # axes[m,n].plot(np.arange(0,16,0.01),np.arange(0,16,0.01),color='k', linestyle='--',lw=2)
 norm = colors.Normalize(vmin=0, vmax=24)
 sc0=axes[m,n].scatter(x0,y0,s=5,marker='o',c=[0]*x0.shape[0],norm=norm,cmap=dcmap_24())
 for j in range(1,24,1):
  exec("sc{}=axes[m,n].scatter(x{},y{},s=5,marker='o',c=[{}]*x{}.shape[0],norm=norm,cmap=dcmap_24())".format(j,j,j,j,j))
 fig.colorbar(sc0,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,24,25),label='Hour')
# add text box for the statistics
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 # axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 return plt.savefig(dir_save)


name=['mlp','dbn','lgb','xgb','rf']
label1=['VMD-PCA-MLP','VMD-PCA-DBN','VMD-PCA-lightGBM','VMD-PCA-XGBoost','VMD-PCA-RF']
name1=['MLP','dbn','LGB','XGB','RF']
for i in range(5):
 ds_DS=pd.read_csv('/ws_correct_ML/vmd_pca/testand{:}_all.csv'.format(name[i]))
 ds_DS['time'] = pd.to_datetime(ds_DS['time'])
 ds_DS = ds_DS.set_index('time')
 ds_DS.columns=['Y_DS1','Y_DS_predict']
 ds=pd.read_csv('/ws_correct_ML/machinelearning/data_410sitetrain1day.csv')
 ds_test=ds[['time','风速','WIN_pre']]
 ds_test['time'] = pd.to_datetime(ds_test['time'])
 ds_test = ds_test.set_index('time')
 ds_test.columns=['WIN_obs','WIN_pre']

 ds_test0=ds_test['2021-09']
 ds_test1=ds_test['2021-10']
 ds_test2=ds_test['2021-11']
 ds_test3=ds_test['2022-03']
 ds_test4=ds_test['2022-04']
 ds_test5=ds_test['2022-05']
 ds_test6=ds_test['2022-06']
 ds_test7=ds_test['2022-07']

 ds_DS0=ds_DS['2021-09']
 ds_DS1=ds_DS['2021-10']
 ds_DS2=ds_DS['2021-11']
 ds_DS3=ds_DS['2022-03']
 ds_DS4=ds_DS['2022-04']
 ds_DS5=ds_DS['2022-05']
 ds_DS6=ds_DS['2022-06']
 ds_DS7=ds_DS['2022-07']
 plotreg(ds_test0,ds_DS0,ds_test1,ds_DS1,ds_test2,ds_DS2,ds_test3,ds_DS3,ds_test4,ds_DS4,ds_test5,ds_DS5,ds_test6,ds_DS6,ds_test7,ds_DS7,label='{:}'.format(label1[i]),dir_save="/ws_correct_ML/vmd_pca/{:}plotreg_all.png".format(name1[i]))