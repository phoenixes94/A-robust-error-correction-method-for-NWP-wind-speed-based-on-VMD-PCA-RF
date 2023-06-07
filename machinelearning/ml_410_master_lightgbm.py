import lightgbm as lgb
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
from vmdpy import VMD 
from sklearn.decomposition import PCA
from multiprocessing import Process
import threading
ds=pd.read_csv('/ws_correct_ML/machinelearning/data_410sitetrain1day.csv',encoding='gbk')
print(ds[ds.isnull().T.any()])#找到有空值的行
print(np.isfinite(ds.all()))
print(np.isinf(ds.all()))
ds=ds.drop(columns=['Unnamed: 0','站点号', '纬度', '经度','降水', '气压','湿度', '温度','time.1'])
# ds=ds.dropna(subset=(ds.columns))
Nan=ds[ds.isnull().T.any()]#有空值的行输出
ds_des=ds.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
ds.columns=['time', 'hgt', 'WIN_obs', 'PRE_pre', 'PRS_pre', 'RHU_pre', 'TMP_pre','WIN_pre', 'D2_pre', 'DIR10_pre',  'U10_pre', 'V10_pre','lon', 'lat']
corr=ds.corr()
corr.to_csv('/ws_correct_ML/machinelearning/corr.csv')
#在泽霞师姐的代码里画了
# plt.figure(figsize=(14,6))
# plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
# plt.rcParams['figure.dpi'] = 300 
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# ax = sns.heatmap(corr, 
#                  # mask=mask,
#                  vmin=-1, vmax=1, 
#                    # square=True,
#                  annot=True,fmt=".2f",
#                   # xticklabels=False,
#                   cbar_kws={"orientation": "horizontal"},
#                      cmap='seismic')
# plt.xticks(rotation = 0)
# plt.savefig("/ws_correct_ML/machinelearning/corr.png")
ds['time'] = pd.to_datetime(ds['time'])
ds = ds.set_index('time')
# ds_train=ds['2021-09':'2022-02']
ds_train=ds['2022-02']
print(ds_train)
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
print(np.isfinite(Xtrain.all()))
print(Xtrain[Xtrain.isnull().T.any()])
ss1 = StandardScaler()#把y进行标准化
ss1=ss1.fit(Ytrain.values.reshape(-1,1))
Ytrain=ss1.transform(Ytrain.values.reshape(-1,1))
Ytest=ss1.transform(Ytest.values.reshape(-1,1))
Ytrain1=ss1.inverse_transform(Ytrain)
Ytest1=ss1.inverse_transform(Ytest)


ds_test=ds['2021-12':'2022-01']
X_DS = ds_test.drop(columns =['WIN_obs'])#0
Y_DS = ds_test['WIN_obs']#0
X_DS.loc[:,col] = ss.transform(X_DS.loc[:,col])#0
Y_DS=ss1.transform(Y_DS.values.reshape(-1,1))#0
Y_DS1=ss1.inverse_transform(Y_DS)#反转#0

# 以下是lightgbm交叉验证的函数
def rf_cv(n_estimators, max_depth,min_child_samples,num_leaves):
    val = cross_val_score(lgb.LGBMRegressor(
        num_leaves=int(num_leaves), 
        n_estimators=int(n_estimators), 
        max_depth=int(max_depth),  
        min_child_samples=int(min_child_samples), 
        random_state=2),Xtrain, Ytrain.ravel(), scoring='neg_mean_squared_error', cv=5,n_jobs=5).mean()
    return val
rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (100, 500),
        'max_depth': (5, 40),
        'num_leaves':(50,300),
        'min_child_samples': (15, 50)})
rf_bo.maximize()
index = []
for i in rf_bo.res:
    index.append(i['target'])
max_index = index.index(max(index))
print(rf_bo.res[max_index])
file = open(r'/ws_correct_ML/machinelearning/lightgbm_parameter_win.txt', mode='w')
file.write('{:}'.format(rf_bo.res[max_index]))      
file.close()

#######1#########
file = open(r'/ws_correct_ML/machinelearning/lightgbm_parameter_win.txt', mode='r').read()

model = lgb.LGBMRegressor(max_depth=int(eval(file)['params']['max_depth']),
                        n_estimators=int(eval(file)['params']['n_estimators']),
                        num_leaves=int(eval(file)['params']['num_leaves']),
                        min_child_samples=int(eval(file)['params']['min_child_samples']),
                        random_state=2)
Matchcount=0;
class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self._stop_event = threading.Event()
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print ("Starting " + self.name)
        print_time(self.name, self.counter, 5)
        print ("Exiting " + self.name)
    def stop(self):
        #self.__flag.set()       # 将线程从暂停状态恢复, 如何已经暂停的话
        print("程序结束"+str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+"...")
        self._stop_event.set()       # 设置为Fal
    def stopped(self):
        return self._stop_event.is_set()
 
def print_time(threadName, delay, counter):   
       while Matchcount == 0:
          time.sleep(100)         
          print("程序正在执行"+str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+"...") 
def GetR2(x,y):
    results = {}
    coeffs = np.polyfit(x,y,1)
    results['polynomial'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['determination'] = ssreg/sstot
    return results['determination']
ProgramLog=[]
def DealData(X,Y,modelName):
   #结果列表
    R_test=[]#测试集 R2
    Rmse_test=[]# 测试集RMSE
    Forest_test=[]#测试集 预测值
    Trues_test=[]#测试集  真实值
    Models=[]#模型
    R_train=[]#训练集 R2
    Rmse_train=[]#训练集Rmse
    Forest_train=[]#训练集 预测值
    Trues_train=[]#训练集 真实值
    Log=[]#训练详细日志
    #五折交叉验证 
    kf = KFold(n_splits=5,shuffle=True,random_state=5)
    # resultpath="./Forest_"+modelName+".csv";
    plog=modelName+"数据训练中."
    ProgramLog.append(plog)
    print(plog)
    log=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    Log.append(log)    
    count=0
    #训练与预测
    model = lgb.LGBMRegressor(max_depth=int(eval(file)['params']['max_depth']),
                        n_estimators=int(eval(file)['params']['n_estimators']),
                        num_leaves=int(eval(file)['params']['num_leaves']),
                        min_child_samples=int(eval(file)['params']['min_child_samples']),
                        random_state=2)
    X.index=range(X.shape[0])
    for train_index, test_index in kf.split(X): 
        count+=1
        x_train, x_test =np.array(X.loc[train_index,:]), np.array(X.loc[test_index,:])
        y_train, y_test = np.array(Y[train_index]), np.array(Y[test_index])
        #训练
        model.fit(x_train, y_train)#dbn需要注释,其他模型不需要
        #训练集结果
        # ytrain_pred= ss1.inverse_transform(model.predict(x_train,200))#dbn需加上batchsize，其他不需要
        ytrain_pred= ss1.inverse_transform(model.predict(x_train).reshape(-1,1)).ravel()
        r2_train=round(GetR2(y_train.ravel(),ytrain_pred) ,3)
        rmse_train=round(np.sqrt(mean_squared_error(y_train, ytrain_pred)),3)
        log="第"+str(count)+"训练,R2:"+ str(r2_train) +"  ,Rmse :"+ str(rmse_train)
        print(log)
        Log.append(log)
        #训练结果保存到列表
        R_train.append(r2_train)
        Rmse_train.append(rmse_train)
        Forest_train.append(ytrain_pred) 
        Trues_train.append(ss1.inverse_transform(y_train.reshape(-1,1)))
        #预测
        # ytest_pred= ss1.inverse_transform(model.predict(x_test,200))
        ytest_pred= ss1.inverse_transform(model.predict(x_test).reshape(-1,1)).ravel()
        #测试集结果
        r2_test=round(GetR2(y_test.ravel(),ytest_pred),3)
        rmse_test=round(np.sqrt(mean_squared_error(y_test, ytest_pred)),3)
        log="第"+str(count)+"验证,R2:"+ str(r2_test) +"  ,Rmse :"+str(rmse_test)
        Log.append(log)
        print(log)
        #预测结果保存到列                   
        R_test.append(r2_test)
        Rmse_test.append(rmse_test)
        Forest_test.append(ytest_pred)
        Trues_test.append(ss1.inverse_transform(y_test.reshape(-1,1)))
        Models.append(model)
    print("结果分析")         
    #统计结果
    # 获取预测结果最大的R2 数据
    testmeanR=round(np.mean(R_test),3)   
    testmeanRmse=round(np.mean(Rmse_test),3) #位置
    trainmeanR=round(np.mean(R_train),3)
    trainmeanRmse=round(np.mean(Rmse_train),3)
    log="验证R2 平均="+str(testmeanR) +"  对应Rmse："+str(testmeanRmse)
    Log.append(log)
    ProgramLog.append(log)
    print(log)
    log="训练R2 平均="+str(trainmeanR) +"  对应Rmse："+str(trainmeanRmse)
    Log.append(log)    
    ProgramLog.append(log)
    print(log)
    print(plog)  
    #训练结果
    plog="保存结果中"
    ProgramLog.append(plog)
    print(plog)  
    #该站所有训练的R RMSE 
    filePath="./训练RRMSE"+modelName+".csv"
    dataframe = pd.DataFrame({'R':R_train,'RMSE':Rmse_train})
    dataframe.to_csv(filePath,index=False, sep=',', encoding="utf_8_sig")
    #该站 所有预测的 R RMSE 
    filePath2="./验证RRMSE"+modelName+".csv"
    dataframe2 = pd.DataFrame({'R':R_test,'RMSE':Rmse_test})
    dataframe2.to_csv(filePath2,index=False, sep=',', encoding="utf_8_sig")
    forest_train=np.concatenate(Forest_train, axis=0)
    trues_train=np.concatenate(Trues_train, axis=0)
    forest_test=np.concatenate(Forest_test, axis=0)
    trues_test=np.concatenate(Trues_test, axis=0)
    # pd.DataFrame({'TrueData':trues_train.ravel(),'ForestData':forest_train}).to_csv('./rf训练ALL.csv')
    # pd.DataFrame({'TrueData':trues_test.ravel(),'ForestData':forest_test}).to_csv('./rf验证ALL.csv')
    pd.DataFrame({'TrueData':trues_train.ravel(),'LgbData':forest_train}).to_csv('./lgb训练ALL.csv')
    pd.DataFrame({'TrueData':trues_test.ravel(),'LgbData':forest_test}).to_csv('./lgb验证ALL.csv')
    # pd.DataFrame({'TrueData':trues_train,'DtrData':forest_train}).to_csv('./dtr训练ALL.csv')
    # pd.DataFrame({'TrueData':trues_test,'DtrData':forest_test}).to_csv('./dtr验证ALL.csv')
    # pd.DataFrame({'TrueData':trues_train.ravel(),'SvrData':forest_train}).to_csv('./svr训练ALL.csv')
    # pd.DataFrame({'TrueData':trues_test.ravel(),'SvrData':forest_test}).to_csv('./svr验证ALL.csv')#
    # pd.DataFrame({'TrueData':trues_train,'MlpData':forest_train}).to_csv('./mlp训练ALL.csv')
    # pd.DataFrame({'TrueData':trues_test,'MlpData':forest_test}).to_csv('./mlp验证ALL.csv')
    # pd.DataFrame({'TrueData':trues_train,'DbnData':forest_train}).to_csv('./dbn训练ALL.csv')
    # pd.DataFrame({'TrueData':trues_test,'DbnData':forest_test}).to_csv('./dbn验证ALL.csv')
    #该站 预测R2最好时候训练结果
    for index in range(0,5):
        train_forest=Forest_train[index]
        train_true=Trues_train[index]
        #预测结果
        test_forest=Forest_test[index]
        test_true=Trues_test[index]
        filePath3="./训练"+str(index)+modelName+".csv"
        # dataframe3 = pd.DataFrame({'TrueData':train_true.ravel(),'ForestData':train_forest})
        dataframe3 = pd.DataFrame({'TrueData':train_true.ravel(),'LgbData':train_forest})
        # dataframe3 = pd.DataFrame({'TrueData':train_true,'DtrData':train_forest})
        # dataframe3 = pd.DataFrame({'TrueData':train_true.ravel(),'SvrData':train_forest})#
        # dataframe3 = pd.DataFrame({'TrueData':train_true,'MlpData':train_forest})
        # dataframe3 = pd.DataFrame({'TrueData':train_true,'DbnData':train_forest})
        dataframe3.to_csv(filePath3,index=False,sep=',')
        #该站预测R2最好时候预测结果
        filePath4="./验证"+str(index)+modelName+".csv"    
        # dataframe4 = pd.DataFrame({'TrueData':test_true.ravel(),'ForestData':test_forest})
        dataframe4 = pd.DataFrame({'TrueData':test_true.ravel(),'LgbData':test_forest})
        # dataframe4 = pd.DataFrame({'TrueData':test_true,'DtrData':test_forest})
        # dataframe4 = pd.DataFrame({'TrueData':test_true.ravel(),'SvrData':test_forest})#
        # dataframe4 = pd.DataFrame({'TrueData':test_true,'MlpData':test_forest})
        # dataframe4 = pd.DataFrame({'TrueData':test_true,'DbnData':test_forest})
        dataframe4.to_csv(filePath4, index=False, sep=',', encoding="utf_8_sig")
# 保存模型 一个文件一个模型 ，注意使用
    with open('/ws_correct_ML/machinelearning/lgbmodel.pickle', 'wb+') as f:#
        pickle.dump(model, f)
    print("保存模型成功")

fileSavePath='/ws_correct_ML/machinelearning'
if __name__=='__main__':
    plog="程序开始："+ str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    print(plog)   
    thread1 = myThread(1, "Thread-1", 1)
    thread1.start()
    DealData(Xtrain, Ytrain, 'lgbtrain')
    # 保存程序日志 
    plog="程序结束："+ str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    ProgramLog.append(plog)
    logpath = fileSavePath+"/LGB程序日志.txt"
    fh = open(logpath, 'w', encoding='utf-8')
    for log in ProgramLog:
        fh.write(log + "\n")
    fh.close()
    Matchcount=1
    thread1.join()


#模型保存后，从此处开始运行
with open('/ws_correct_ML/machinelearning/lgbmodel.pickle', 'rb') as f:
    model = pickle.load(f)
print("加载模型成功")
########1#########





cross_val_predicted = ss1.inverse_transform(cross_val_predict(model, Xtrain, Ytrain, cv=5,n_jobs=5))#dbn需要注释,其他模型不需要
model.fit(Xtrain, Ytrain)#dbn需要注释,其他模型不需要

Ytest_predict = model.predict(Xtest)
Ytest_predict= ss1.inverse_transform(Ytest_predict)
Ytrain_predict = rf.predict(Xtrain)
Ytrain_predict= ss1.inverse_transform(Ytrain_predict)

Y_DS_predict = model.predict(X_DS)#2
Y_DS_predict= ss1.inverse_transform(Y_DS_predict)#2




print('R^2 cross_val: %.3f, R^2 test: %.3f, DS R^2 : %.3f' % (
        (metrics.r2_score(Ytrain1,cross_val_predicted)),#dbn需要注释,其他模型不需要
       # (metrics.r2_score(Ytrain1, Ytrain_predict)), 
      (metrics.r2_score(Ytest1, Ytest_predict)),
      (metrics.r2_score(Y_DS1, Y_DS_predict))))
##绘制残差
# def plot_residual(model, y_train, y_train_pred, y_test, y_test_pred):
    
#     """Print the regression residuals.
    
#     Args:
#         model (str): The model name identifier
#         y_train (series): The training labels
#         y_train_pred (series): Predictions on training data
#         y_test (series): The test labels
#         y_test_pred (series): Predictions on test data
        
#     Returns:
#         Plot of regression residuals
    
#     """
#     plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
#     plt.rcParams['figure.dpi'] = 300 
#     plt.scatter(y_train, y_train_pred - y_train, c='blue', s=2,marker='o', label='训练集')
#     plt.scatter(y_test, y_test_pred - y_test, c='red',s=2, marker='<', label='测试集')
#     plt.xlabel('GPP监测值')
#     plt.ylabel('残差')
#     plt.legend(loc='upper right')
#     plt.hlines(y=0, xmin=-12, xmax=30, color='red', lw=0.5)
#     plt.title(model + '残差')
#     plt.show()

# def get_regression_metrics(model, actual, predicted):
    
#     """Calculate main regression metrics.
    
#     Args:
#         model (str): The model name identifier
#         actual (series): Contains the test label values
#         predicted (series): Contains the predicted values
        
#     Returns:
#         dataframe: The combined metrics in single dataframe
#     """
#     regr_metrics = {
#                         'Root Mean Squared Error' : metrics.mean_squared_error(actual, predicted)**0.5,
#                         'Mean Absolute Error' : metrics.mean_absolute_error(actual, predicted),
#                         'R^2' : metrics.r2_score(actual, predicted),
#                         'R' : np.corrcoef(actual,predicted)[0][1],
#                         'Explained Variance' : metrics.explained_variance_score(actual, predicted)
#                    }
#     #return reg_metrics
#     df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
#     df_regr_metrics.columns = [model]
#     return df_regr_metrics

def plot_features_weights(model, weights, feature_names, weights_type='c'):
    """Plot regression coefficients weights or feature importance.
    Args:
        model (str): The model name identifier
        weights (array): Contains the regression coefficients weights or feature importance
        feature_names (list): Contains the corresponding features names
        weights_type (str): 'c' for 'coefficients weights', otherwise is 'feature importance'
    Returns:
        plot of either regression coefficients weights or feature importance
    """
    plt.figure(figsize=[20,10])
    plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
    plt.rcParams['figure.dpi'] = 300 
    (px, py) = (8, 10) if len(weights) > 30 else (8, 5)
    W = pd.DataFrame({'Weights':weights}, feature_names)
    W.sort_values(by='Weights', ascending=True).plot(kind='barh', color='b', figsize=(px,py))
    label = ' Coefficients' if weights_type =='c' else ' Features Importance'
    plt.xlabel(model + label)
    plt.gca().legend_ = None
    









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
def plotreg(Ytrain1, Ytrain_predict,valid1, valid_predict,ds_test0,ds_DS0,ds_test1,ds_DS1,label,dir_save):
 fig,axes=plt.subplots(3,2,figsize=(22,30))
  # p1, p0 = np.polyfit(Ytrain1, Ytrain_predict, deg=1)  # slope, intercept
 def f_1(x, A):
    return A * x 
 # 得到返回的A，B值
 m=0
 n=0
 A = np.float(op.curve_fit(f_1, Ytrain1, Ytrain_predict)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
# Calculate point density
 xy = np.vstack([Ytrain1, Ytrain_predict])
 z = gaussian_kde(xy)(xy)
 # Sort points by density
 idx = z.argsort()
 x, y, z = Ytrain1[idx], Ytrain_predict[idx], z[idx]
 norm = colors.Normalize(vmin=0, vmax=100)
 sc=axes[m,n].scatter(x,y,c=1000*z,s=30,marker='o',norm=norm,cmap=plt.cm.get_cmap('jet'))
 fig.colorbar(sc,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,100,6),
             label='',
             extend='max')
# add text box for the statistics
 stats = (f'y = {A:.2f}x\n'
    f'R={np.corrcoef(Ytrain1, Ytrain_predict)[0][1]:.2f}, N={len(Ytrain1):d}\n'
    f'RMSE={RMSE(Ytrain1, Ytrain_predict):.2f}, FA={FA(Ytrain1, Ytrain_predict):.2f}'
             )
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 axes[m,n].text(1,14,'(a)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('Estimated',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 # p1, p0 = np.polyfit(Ytest1, Ytest_predict, deg=1)  # slope, intercept
 m=0
 n=1
 A = np.float(op.curve_fit(f_1, valid1, valid_predict)[0])
 axes[m,n].axline(xy1=(0,0), slope=A, color='r', lw=2)
 axes[m,n].axline(xy1=(0,0), slope=1, color='k', linestyle='--',lw=2)
 # Calculate point density
 xy = np.vstack([valid1, valid_predict])
 z = gaussian_kde(xy)(xy)
 # Sort points by density
 idx = z.argsort()
 x, y, z = valid1[idx], valid_predict[idx], z[idx]
 norm = colors.Normalize(vmin=0, vmax=100)
 sc=axes[m,n].scatter(x,y,c=1000*z,s=30,marker='o',norm=norm,cmap=plt.cm.get_cmap('jet'))
 fig.colorbar(sc,ax=axes[m,n],
             # location='right',
             format='%d',
             # size=0.3,
             ticks=np.linspace(0,100,6),label='',extend='max')
# add text box for the statistics
 stats = (f'y = {A:.2f}x\n'
    f'R={np.corrcoef(valid1, valid_predict)[0][1]:.2f}, N={len(valid1):d}\n'
    f'RMSE={RMSE(valid1, valid_predict):.2f}, FA={FA(valid1, valid_predict):.2f}'
             )
 bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
 axes[m,n].text(0.15, 0.82, stats, fontsize=20, bbox=bbox,
            transform=axes[m,n].transAxes, horizontalalignment='left')
 axes[m,n].text(1,14,'(b)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('Estimated',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=1
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
 axes[m,n].text(1,14,'(c)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=1
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
 axes[m,n].text(1,14,'(d)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=2
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
 axes[m,n].text(1,14,'(e)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel('WRF',fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 m=2
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
 axes[m,n].text(1,14,'(f)',fontsize=20)
 axes[m,n].set_xlabel('Observed',fontsize=20)
 axes[m,n].set_ylabel(label,fontsize=20)
 axes[m,n].set_xlim(0,16)
 axes[m,n].set_ylim(0,16)
 return plt.savefig(dir_save)


data_train=pd.read_csv('/ws_correct_ML/machinelearning/lgb训练ALL.csv') 
data_valid=pd.read_csv('/ws_correct_ML/machinelearning/lgb验证ALL.csv')
Ytrain=data_train['TrueData']
Ytrain_predict=data_train['LgbData']
valid=data_valid['TrueData']
valid_predict=data_valid['LgbData']
ds_DS=pd.DataFrame(np.concatenate((Y_DS1, Y_DS_predict[:,np.newaxis]),axis=1),columns=['Y_DS1', 'Y_DS_predict'])#3
ds_DS.index=ds_test.index#3
ds_DS.to_csv('/ws_correct_ML/machinelearning/testandlgb.csv')#3
ds_DS=pd.read_csv('/ws_correct_ML/machinelearning/testandlgb.csv')
ds_DS['time'] = pd.to_datetime(ds_DS['time'])
ds_DS = ds_DS.set_index('time')
ds_test0=ds_test['2021-12']
ds_test1=ds_test['2022-01']
ds_DS0=ds_DS['2021-12']
ds_DS1=ds_DS['2022-01']
plotreg(Ytrain, Ytrain_predict,valid, valid_predict,ds_test0,ds_DS0,ds_test1,ds_DS1,label='lightGBM',dir_save='/ws_correct_ML/machinelearning/LGBplotreg.png')