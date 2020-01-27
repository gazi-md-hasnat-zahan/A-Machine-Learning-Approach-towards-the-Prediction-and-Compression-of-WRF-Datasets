####wthout location info and missing points 2014
import csv
import numpy as np
import pandas as pd
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from keras.datasets import cifar10
from sklearn.metrics import r2_score
from pathlib import Path
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation,Flatten
from keras.models import Model,Input,Sequential
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import os
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from keras.utils import layer_utils
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
#from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression, PLSSVD
##trial
def read_data(file_name):
    dataset = np.genfromtxt(file_name, dtype = "str", delimiter = ",")
    headers = dataset[0]
    data = dataset[1:,3:29].astype("float").reshape((699, 639, 26))
#     print('reading...' + file_name)
    return data

def level_data(data):
    for j in range(26):
        temp = data[:, j]
        temp = np.interp(temp, (temp.min(), temp.max()), (0, 1))
        data[:, j] = temp
    data = data.astype("int")
    return data
####for svr
####final

# #Take Training files first

# file_path = os.getcwd()
# readstart=0
# for file_name in os.listdir(file_path):
#     if (file_name[-3:]=='csv' and file_name[0:7] not in ['2013-06','2014-07','2015-08','2013-01','2014-12','2015-02']):
#         readstart+=1
#         if(readstart==1):
#             print('reading...' + file_name)
#             result = read_data(file_name).reshape((699* 639, 26))
#             result=level_data(result)
#         else:
#             print('reading...' + file_name)
#             result2 = read_data(file_name).reshape((699* 639, 26))
#             result2=level_data(result2) 
#             result = np.concatenate((result, result2))


# # print(result.shape)
# # print(result.size)

# y=result[:,25]
# x=np.delete(result,25,1)
# print('Actual X shape: ' , x.shape)
# print('Y shape: ' , y.shape) 

# pca = PCA(random_state=42).fit(x)
# #Plotting the Cumulative Summation of the Explained Variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title('Weather Dataset Explained Variance')
# plt.show()
# plt.savefig('PCA_ncomp_var.png')

# ncomp = 2
# print('Number of components taken: ', ncomp)
# X_reduced = PCA(n_components=ncomp,random_state=42).fit_transform(x)
# print('X_reduced shape: ' , X_reduced.shape)
# regr = LinearRegression()
# regr.fit(X_reduced,y)
# ypc=regr.predict(X_reduced)
# print('The R2 score is: ', r2_score(ypc,y))            


result = np.genfromtxt("final_pca_reduced_data_with_label.txt", dtype=None, delimiter=',', skip_header=1)
result=result.astype('float')
print(result.shape)

X_reduced = result[:,0:2]
y=result[:,2]


reg = linear_model.BayesianRidge()
reg.fit(X_reduced, y)

# y_test_pred=pls.predict(X_test)
# print('The R2 score is: ', r2_score(y_test_pred,Y_test))

print('Starting gathering testing samples')

testfilescount = 6
testfilesnames = ['2013-06','2014-07','2015-08','2013-01','2014-12','2015-02']
final_mae=[]
final_mse=[]
final_r2=[]

file_path = os.getcwd()
for i in range(testfilescount):
    #Take Summer Testing files
    for file_name in os.listdir(file_path):
        if (file_name[-3:]=='csv' and file_name[0:7] == testfilesnames[i]):
            print('reading...' + file_name)
            result = read_data(file_name).reshape((699* 639, 26))
            result=level_data(result)     

            y=result[:,25]
            x=np.delete(result,25,1)
        #     print('Actual X shape: ' , x.shape)
        #     print('Y shape: ' , y.shape) 

            ncomp = 2
            print('Number of components taken: ', ncomp)
            X_reduced = PCA(n_components=ncomp,random_state=42).fit_transform(x)
        #     print('X_reduced shape: ' , X_reduced.shape)

            #Get the SoilMois Data , last 3 is the 3rd feature, which is Soil moisture
            Y_test = y


            Y_pred = reg.predict(X_reduced)
            
            print("min and max of y pred: ",min(Y_pred),max(Y_pred))
            print(Y_pred.shape)
            print()
            
            print("min and max of y pred: ",min(Y_pred),max(Y_pred))
            print(file_name, 'result: ')
            r2=r2_score(Y_test,Y_pred)
            mae=mean_absolute_error(Y_test,Y_pred)
            mse=mean_squared_error(Y_test,Y_pred)
            final_r2.append(r2)
            final_mse.append(mse)
            final_mae.append(mae)
            
            print('R2 score: ',r2_score(Y_test,Y_pred))
            print('MAE: ',mean_absolute_error(Y_test,Y_pred))
            print('MSE: ',mean_squared_error(Y_test,Y_pred))
            
            Y_pred=(Y_pred - np.min(Y_pred))/(np.max(Y_pred)-np.min(Y_pred))
            
            originalfigname = file_name + 'original_BR.png'
            preditedfigname = file_name + 'predicted_BR.png'
            
            fig1=plt.figure()
            plt.contourf(Y_test.reshape(699,639), levels = np.linspace(min(Y_test),max(Y_test),10))
            plt.colorbar()
            plt.axis('off')
            plt.savefig(originalfigname,dpi=1000)
            plt.close()

            fig2=plt.figure()
            plt.contourf(Y_pred.reshape(699,639), levels = np.linspace(0,1,10))
            plt.colorbar()
            plt.axis('off')
            plt.savefig(preditedfigname,dpi=1000)
            plt.show()
            plt.close()


print("BR average MSE loss",np.mean(final_mse))
print("BR average MAE",np.mean(final_mae))
print("BR average r2 score",np.mean(final_r2)) 

final_r2.append(np.mean(final_r2))
final_mse.append(np.mean(final_mse))
final_mae.append(np.mean(final_mae))
            
np.savetxt("BR Loss(MSE).txt", final_mse,delimiter=",")
np.savetxt("BR MAE.txt", final_mae, delimiter=",")
np.savetxt("BR r2.txt", final_r2, delimiter=",")          
            
            
# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)

# print("10 Fold Cross Validation Starting...")

# X_train=X_reduced
# Y_train=y

# kf = KFold(n_splits=10, shuffle=True, random_state=seed)
# cvscores = []
# cvscores2 = []
# cvscores1=[]
# for train, test in kf.split(X_train,Y_train):
#     # create model
#     reg=linear_model.BayesianRidge()
#     reg.fit(X_train[train], Y_train[train])
#     Y_pred=reg.predict(X_train[test])
#     scores0=mean_squared_error(Y_train[test], Y_pred)
#     scores1=r2_score(Y_train[test], Y_pred)
#     scores2 = mean_absolute_error(Y_train[test], Y_pred)
    
#     cvscores.append(scores0)
#     cvscores1.append(scores1)
#     cvscores2.append(scores2)

# print("Average CV Loss(MSE): %.2f with St. Dev: (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
# print("Average CV MAE: %.2f with St. Dev: (+/- %.2f)" % (np.mean(cvscores2), np.std(cvscores2)))
# print("Average CV r2_score: %.2f with St. Dev: (+/- %.2f)" % (np.mean(cvscores1), np.std(cvscores1)))
# np.savetxt("BR_Average CV Loss(MSE).txt", cvscores,delimiter=",")
# np.savetxt("BR_Average CV MAE.txt", cvscores2, delimiter=",")
# np.savetxt("BR_Average CV r2_score.txt", cvscores1, delimiter=",")

# fig1=plt.figure()
# plt.plot(cvscores)
# plt.title('Cross Validation Plot')
# plt.ylabel('Loss(MSE)')
# plt.xlabel('Model')
# plt.show()
# plt.savefig('CVPlot_MSE-BRModel.png')

# fig1=plt.figure()
# plt.plot(cvscores1)
# plt.title('Cross Validation Plot')
# plt.ylabel('Loss(MAE)')
# plt.xlabel('Model')
# plt.show()
# plt.savefig('CVPlot_MAE-BRmodel.png')

# fig2=plt.figure()
# plt.plot(cvscores2)
# plt.title('Cross Validation Plot')
# plt.ylabel('r2_score')
# plt.xlabel('Model')
# plt.show()
# plt.savefig('CVPlot_MAE-BRModel.png')
