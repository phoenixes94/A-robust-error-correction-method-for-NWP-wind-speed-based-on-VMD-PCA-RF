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
# from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
#绘制vmd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import threading
# ds=pd.read_csv('/ws_correct_ML/data_411sitetrain1day.csv',encoding='gbk')
# #. some sample parameters for VMD  
# alpha = 2000       # moderate bandwidth constraint  
# tau = 0.0244            # noise-tolerance (no strict fidelity enforcement)  
# K = 9             # 3 modes  
# DC = 0             # no DC part imposed  
# init = 1           # initialize omegas uniformly  
# tol = 1e-7  
# for i in range(ds['站点号'].unique().shape[0]):
#     print(i)
#     f=ds.loc[ds['站点号']==ds.loc[i,'站点号'],'WIN_predict']
#     #. Run VMD 
#     u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF0']=u[0,:]
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF1']=u[1,:]
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF2']=u[2,:]
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF3']=u[3,:] 
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF4']=u[4,:]
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF5']=u[5,:]
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF6']=u[6,:]
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF7']=u[7,:]
#     ds.loc[ds['站点号']==ds.loc[i,'站点号'],'IMF8']=u[8,:]
# print(ds[ds.isnull().T.any()])#找到有空值的行
# print(np.isfinite(ds.all()))
# print(np.isinf(ds.all()))
# ds=ds.drop(columns=['Unnamed: 0', '站点号','纬度', '经度','降水', '气压','湿度', '温度','time.1'])
# ds=ds.dropna(subset=(ds.columns))
# ds.to_csv('/ws_correct_ML/vmd_pca/data_410sitetrain1day_new.csv')





ds=pd.read_csv('/ws_correct_ML/vmd_pca/data_410sitetrain1day_new.csv')
ds=ds.drop(columns=['Unnamed: 0'])
def pca(data,num):
    p=PCA(n_components=num)
    p.fit(data)
    data_pca=p.transform(data)
    print(p.explained_variance_ratio_)#[0.51716302 0.25381102 0.12449852]
    return data_pca

ds_pca=pca(ds.loc[:,['IMF0', 'IMF1', 'IMF2', 'IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7', 'IMF8']],0.8)
ds.loc[:,'pca0']=ds_pca[:,0]
ds.loc[:,'pca1']=ds_pca[:,1]
ds.loc[:,'pca2']=ds_pca[:,2]
Nan=ds[ds.isnull().T.any()]#有空值的行输出
ds_des=ds.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
ds.columns=['time', 'hgt', 'WIN_obs', 'PRE_pre', 'PRS_pre', 'RHU_pre','TMP_pre', 'WIN_pre', 'lon', 'lat','IMF0', 'IMF1', 'IMF2','IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7', 'IMF8','pca0', 'pca1', 'pca2']
corr=ds.corr()

# ds_3d=ds.loc[ds['hgt']==ds.loc[0,'hgt'],['time','IMF0', 'IMF1', 'IMF2','IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7', 'IMF8','pca0', 'pca1', 'pca2']]
# ds_3d.index=range(ds_3d.shape[0])
# ds_3d.to_csv('/ws_correct_ML/vmd_pca/ds_3d.csv')
# x = range(ds_3d.shape[0])

# # Define the three functions to plot.
# for i in range(9):
#    exec('f{:} = ds_3d["IMF{:}"]'.format(i,i)) 

# for i in range(9,12,1):
#    exec('f{:} = ds_3d["pca{:}"]'.format(i,i-9)) 
# # Sets of [x, y, 0] lines (with constant y) for defining the bases of the polygons.
# # set01 = [x, -7*np.ones(len(x)), np.zeros(len(x))]
# # set02 = [x,  0*np.ones(len(x)), np.zeros(len(x))]
# # set03 = [x,  7*np.ones(len(x)), np.zeros(len(x))]

# # Sets of the form [x, y, fi] (with constant y) representing each function 3D line.
# for i in range(12):
# 	exec('set{:}  = [x, np.repeat(720*{:},len(x)), f{:}]'.format(i,-i,i))

# from matplotlib.ticker import MultipleLocator
# # Plotting part. Figure and axes creation.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d', xlim=(0,ds_3d.shape[0]), ylim=(-720*11,720))
# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.9, 1, 0.6, 1]))
# plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
# plt.rcParams['figure.dpi'] = 300 
# plt.rc('font', family='Times New Roman')
# # plt.rcParams["font.weight"] = "bold"
# # plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams['font.size'] = '5'
# # Plot the f1 line and its fill_between_3d.
# for i in range(12):
#  exec('ax.plot(*set{:}, lw=0.5,zorder={:}, c="C{}")'.format(i,-i,i))
# # fill_between_3d(ax, *set01, *set1, mode = 1, c="C0")

# # Plot the f2 line and its fill_between_3d.
# # ax.plot(*set2, lw=2, zorder=15, c="C1")
# # fill_between_3d(ax, *set02, *set2, mode = 1, c="C1")

# # Plot the f3 line and its fill_between_3d.
# # ax.plot(*set3, lw=2, zorder=10, c="C2")
# # fill_between_3d(ax, *set03, *set3, mode = 1, c="C2")

# # Correct the z limit of the plot.

# ax.set_zlim(np.min([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]),np.max([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]))
# ax.set_xlabel('time', color='b')
# ax.set_ylabel('components of VMD and PCA', color='g')
# ax.set_zlabel('WRF wind speeds(m/s)', color='r')
# ymajorLocator = MultipleLocator(24*30)
# # yminorLocator = MultipleLocator(24*30)
# xmajorLocator = MultipleLocator(24*30)
# # xminorLocator = MultipleLocator(48*7)
# ax.xaxis.set_major_locator(xmajorLocator)
# ax.yaxis.set_major_locator(ymajorLocator)

# # ax.xaxis.set_minor_locator(xminorLocator)
# # ax.yaxis.set_minor_locator(yminorLocator)

# ax.xaxis.grid(True, which='major')
# ax.yaxis.grid(True, which='minor')
# ax.view_init(40, -40)
# var=list(['IMF0', 'IMF1', 'IMF2','IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7', 'IMF8','pca0', 'pca1', 'pca2'])
# var.reverse()      
# ax.set_yticks(range(-720*11,720,720))
# ax.set_yticklabels(var, rotation=0)
# # Show the plot.
# plt.savefig("/ws_correct_ML/vmd_pca/3d_windspeed.png")



corr.to_csv('/ws_correct_ML/vmd_pca/corr.csv')


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




# 2、以下是以往随机森林交叉验证的函数
def rf_cv(n_estimators, min_samples_split, max_features, max_depth,min_samples_leaf):
    val = cross_val_score(RandomForestRegressor(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=int(max_features), # float
            max_depth=int(max_depth),
            random_state=2),Xtrain, Ytrain.ravel(), scoring='neg_mean_squared_error', cv=5,n_jobs=5).mean()
    return val
rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (100, 500),
        'min_samples_split': (2,10),
        'max_features': (1, 20),#20个变量
        'max_depth': (5, 40),
        'min_samples_leaf': (2, 50)})
rf_bo.maximize()
index = []
for i in rf_bo.res:
    index.append(i['target'])
max_index = index.index(max(index))
print(rf_bo.res[max_index])
file = open(r'/ws_correct_ML/vmd_pca/rf_parameter_win.txt', mode='w')
file.write('{:}'.format(rf_bo.res[max_index]))      
file.close()


file = open(r'/ws_correct_ML/vmd_pca/rf_parameter_win.txt', mode='r').read()
model = RandomForestRegressor(n_estimators=int(eval(file)['params']['n_estimators']),
 max_features=int(eval(file)['params']['max_features']), 
 max_depth=int(eval(file)['params']['max_depth']),
 min_samples_split=int(eval(file)['params']['min_samples_split']),
 min_samples_leaf=int(eval(file)['params']['min_samples_leaf']),random_state=2)







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
# 交叉验证的每折的预测总和
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
def DealData(X,Y,stationName):
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
    # resultpath="./Forest_"+stationName+".csv";
    plog=stationName+"数据训练中."
    ProgramLog.append(plog)
    print(plog)
    log=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    Log.append(log)    
    count=0
    #训练与预测
    model = RandomForestRegressor(n_estimators=int(eval(file)['params']['n_estimators']),
                   max_features=int(eval(file)['params']['max_features']), 
                   max_depth=int(eval(file)['params']['max_depth']),
                   min_samples_split=int(eval(file)['params']['min_samples_split']),
                   min_samples_leaf=int(eval(file)['params']['min_samples_leaf']),random_state=2)
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
    filePath="./训练RRMSE"+stationName+".csv"
    dataframe = pd.DataFrame({'R':R_train,'RMSE':Rmse_train})
    dataframe.to_csv(filePath,index=False, sep=',', encoding="utf_8_sig")
    #该站 所有预测的 R RMSE 
    filePath2="./验证RRMSE"+stationName+".csv"
    dataframe2 = pd.DataFrame({'R':R_test,'RMSE':Rmse_test})
    dataframe2.to_csv(filePath2,index=False, sep=',', encoding="utf_8_sig")
    forest_train=np.concatenate(Forest_train, axis=0)
    trues_train=np.concatenate(Trues_train, axis=0)
    forest_test=np.concatenate(Forest_test, axis=0)
    trues_test=np.concatenate(Trues_test, axis=0)
    pd.DataFrame({'TrueData':trues_train.ravel(),'ForestData':forest_train}).to_csv('./rf训练ALL.csv')
    pd.DataFrame({'TrueData':trues_test.ravel(),'ForestData':forest_test}).to_csv('./rf验证ALL.csv')
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
        filePath3="./训练"+str(index)+stationName+".csv"
        dataframe3 = pd.DataFrame({'TrueData':train_true.ravel(),'ForestData':train_forest})
        # dataframe3 = pd.DataFrame({'TrueData':train_true,'DtrData':train_forest})
        # dataframe3 = pd.DataFrame({'TrueData':train_true.ravel(),'SvrData':train_forest})#
        # dataframe3 = pd.DataFrame({'TrueData':train_true,'MlpData':train_forest})
        # dataframe3 = pd.DataFrame({'TrueData':train_true,'DbnData':train_forest})
        dataframe3.to_csv(filePath3,index=False,sep=',')
        #该站预测R2最好时候预测结果
        filePath4="./验证"+str(index)+stationName+".csv"    
        dataframe4 = pd.DataFrame({'TrueData':test_true.ravel(),'ForestData':test_forest})
        # dataframe4 = pd.DataFrame({'TrueData':test_true,'DtrData':test_forest})
        # dataframe4 = pd.DataFrame({'TrueData':test_true.ravel(),'SvrData':test_forest})#
        # dataframe4 = pd.DataFrame({'TrueData':test_true,'MlpData':test_forest})
        # dataframe4 = pd.DataFrame({'TrueData':test_true,'DbnData':test_forest})
        dataframe4.to_csv(filePath4, index=False, sep=',', encoding="utf_8_sig")
# 保存模型 一个文件一个模型 ，注意使用
    with open('/ws_correct_ML/vmd_pca/rfmodel.pickle', 'wb+') as f:#
        pickle.dump(model, f)
    print("保存模型成功")
fileSavePath='/ws_correct_ML/vmd_pca'
if __name__=='__main__':
    plog="程序开始："+ str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    print(plog)   
    thread1 = myThread(1, "Thread-1", 1)
    thread1.start()
    DealData(Xtrain, Ytrain, 'rftrain')
    # 保存程序日志 
    plog="程序结束："+ str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    ProgramLog.append(plog)
    logpath = fileSavePath+"/RF程序日志.txt"
    fh = open(logpath, 'w', encoding='utf-8')
    for log in ProgramLog:
        fh.write(log + "\n")
    fh.close()
    Matchcount=1
    thread1.join()

with open('/ws_correct_ML/vmd_pca/rfmodel.pickle', 'rb') as f:
    model = pickle.load(f)
print("加载模型成功")






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
#                         'Root Mean Squared Error' : RMSE(actual, predicted)**0.5,
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
    plt.figure(figsize=[15,15])
    plt.rcParams['savefig.dpi'] = 300 #图片像素(根据需求调,值太小图看不清)
    plt.rcParams['figure.dpi'] = 300 
    # (px, py) = (8, 10) if len(weights) > 30 else (8, 5)
    (px, py) = (15,15) 
    plt.rcParams['font.size'] = '30'
    W = pd.DataFrame({'Weights':weights}, feature_names)
    W.sort_values(by='Weights', ascending=True).plot(kind='barh', color='b', figsize=(px,py))
    label = ' Coefficients' if weights_type =='c' else ' Features Importance'
    plt.xlabel(model + label+'(%)')
    plt.gca().legend_ = None
    


# rf_metrics_test = get_regression_metrics('Random Forest Regression', Ytest1, Ytest_predict)
# rf_metrics_train= get_regression_metrics('Random Forest Regression', Ytrain1, cross_val_predicted)
# rf_metrics_DS= get_regression_metrics('Random Forest Regression', Y_DS1, Y_DS_predict)
# rf_metrics_test
# rf_metrics_train
# rf_metrics_DS

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
# plt.rcParams['axes.unicode_minus']=False
# 随机森林一下4行可执行
# plt.rcParams['font.size'] = '15'
# plot_features_weights('Random Forest Regressor', model.feature_importances_, Xtrain.columns, 't' )#特征重要性排序
# plt.savefig("./plot_features_weights.png")
# pd.DataFrame(np.reshape(model.feature_importances_,(13,1)),index=Xtrain.columns).to_csv("./features_weights.csv")
# feature_importances=pd.read_csv("./features_weights.csv").loc[:,'0'].values
# Xtrain_r_columns=['HGT', 'PRE', 'PRS', 'RH', '$T_{2}$', '$WS_{10}$', 'LON', 'LAT','IMF0', 'IMF1', 'IMF2', 'IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7','IMF8', 'pca0', 'pca1', 'pca2']
# plot_features_weights('Random Forest Regressor', model.feature_importances_*100, Xtrain_r_columns, 't' )#特征重要性排序
# plt.savefig("/ws_correct_ML/vmd_pca/plot_features_weights.png")





# plot_residual('随机森林回归', Ytrain, Ytrain_predict, Ytest, Ytest_predict)#残差比较
# plot_residual('随机森林回归', Ytrain1, Ytrain_predict, Ytest1, Ytest_predict)#残差比较

# data_train=pd.DataFrame(np.concatenate((np.array(Ytrain1)[:,np.newaxis],Ytrain_predict[:,np.newaxis]),axis=1)
#                   ,columns=["Ytrain","Ytrain_predict"])
# data_test=pd.DataFrame(np.concatenate((np.array(Ytest1)[:,np.newaxis],Ytest_predict[:,np.newaxis]),axis=1),
#                         columns=['Ytest','Ytest_predict'])
# data_DS=pd.DataFrame(np.concatenate((np.array(Y_DS1)[:,np.newaxis],Y_DS_predict[:,np.newaxis]),axis=1),
#                         columns=['Y_DS','Y_DS_predict'])






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

data_train=pd.read_csv('/ws_correct_ML/vmd_pca/rf训练ALL.csv') 
data_valid=pd.read_csv('/ws_correct_ML/vmd_pca/rf验证ALL.csv')
Ytrain=data_train['TrueData']
Ytrain_predict=data_train['ForestData']
valid=data_valid['TrueData']
valid_predict=data_valid['ForestData']
ds_DS=pd.DataFrame(np.concatenate((Y_DS1, Y_DS_predict[:,np.newaxis]),axis=1),columns=['Y_DS1', 'Y_DS_predict'])#3
ds_DS.index=ds_test.index#3
ds_DS.to_csv('/ws_correct_ML/vmd_pca/testandrf.csv')#3
ds_DS=pd.read_csv('/ws_correct_ML/vmd_pca/testandrf.csv')
ds_DS['time'] = pd.to_datetime(ds_DS['time'])
ds_DS = ds_DS.set_index('time')
ds_test0=ds_test['2021-12']
ds_test1=ds_test['2022-01']
ds_DS0=ds_DS['2021-12']
ds_DS1=ds_DS['2022-01']
plotreg(Ytrain, Ytrain_predict,valid, valid_predict,ds_test0,ds_DS0,ds_test1,ds_DS1,label='VMD-PCA-RF',dir_save="/ws_correct_ML/vmd_pca/RFplotreg.png")

