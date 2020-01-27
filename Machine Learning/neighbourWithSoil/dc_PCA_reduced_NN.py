####wthout location info and missing points 2014
import csv
import numpy as np
import pandas as pd
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.datasets import cifar10
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

def read_data(file_name):#reading data files
    dataset = np.genfromtxt(file_name, dtype = "str", delimiter = ",")
    headers = dataset[0]
    data = dataset[1:,3:29].astype("float").reshape((699, 639, 26))
    return data

def get_neighbors(data,days,features):
    data=data.reshape((days,699,639,features))
    temp = np.zeros((days, 699+ 1 * 2, 639 + 1 * 2, features))
    temp[:, 1:temp.shape[1] - 1, 1: temp.shape[2] - 1] = data
    result = np.zeros((days,699, 639, features, 3 * 3))
    #taking 3 by 3 neighbourhood for each pixel
    for i in range(days):
        result[i, :, :, :, 0] = temp[i, 0:temp.shape[1] - 2, 0:temp.shape[2] - 2, :]
        result[i, :, :, :, 1] = temp[i, 1:temp.shape[1] - 1, 0:temp.shape[2] - 2, :]
        result[i, :, :, :, 2] = temp[i, 2:temp.shape[1] - 0, 0:temp.shape[2] - 2, :]
        result[i, :, :, :, 3] = temp[i, 0:temp.shape[1] - 2, 1:temp.shape[2] - 1, :]
        result[i, :, :, :, 4] = temp[i, 1:temp.shape[1] - 1, 1:temp.shape[2] - 1, :]
        result[i, :, :, :, 5] = temp[i, 2:temp.shape[1] - 0, 1:temp.shape[2] - 1, :]
        result[i, :, :, :, 6] = temp[i, 0:temp.shape[1] - 2, 2:temp.shape[2] - 0, :]
        result[i, :, :, :, 7] = temp[i, 1:temp.shape[1] - 1, 2:temp.shape[2] - 0, :]
        result[i, :, :, :, 8] = temp[i, 2:temp.shape[1] - 0, 2:temp.shape[2] - 0, :]

    result = result.reshape((days*699 * 639, features, 3 * 3))
    return result

def level_data(data):
    #normalizing data
    for j in range(26):
        temp = data[:, j]
        temp = np.interp(temp, (temp.min(), temp.max()), (0, 1))
        data[:, j] = temp
    data = data.astype("int")
    return data

# reading and normalizing data
file_path = os.getcwd()
readstart=0
for file_name in os.listdir(file_path):
    if (file_name[-3:]=='csv' and file_name[0:7] not in ['2013-06','2014-07','2015-08','2013-01','2014-12','2015-02']):
        readstart+=1
        if(readstart==1):
            print('reading...' + file_name)
            result = read_data(file_name).reshape((699* 639, 26))
            result=level_data(result)
        else:
            print('reading...' + file_name)
            result2 = read_data(file_name).reshape((699* 639, 26))
            result2=level_data(result2)
            result = np.concatenate((result, result2))

y=result[:,25]
x=np.delete(result,25,1)
print('Actual X shape: ' , x.shape)
print('Y shape: ' , y.shape)

#doing Principle component Analysis for reducing feature size
pca = PCA(random_state=42).fit(x)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Weather Dataset Explained Variance')
plt.show()
plt.savefig('PCA_ncomp_var.png')

ncomp = 2
print('Number of components taken: ', ncomp)
X_reduced = PCA(n_components=ncomp,random_state=42).fit_transform(x)
print('X_reduced shape: ' , X_reduced.shape)

result=np.column_stack((X_reduced, y))
result=get_neighbors(result,26,3)

print('Splitting training and testing')
nfiles = 26 #number of files for training
#for all lposition, taking 3by3 neighbourhood features as input
X_trainn = result.reshape((699 * 639 * nfiles, 3 * 3 * 3))
#Get the SoilMois Data
Y_train = X_trainn[:,22]
X_train=np.delete(X_trainn,22,1)

print('Building NN...')
epochs=100
model=Sequential()
model.add(Dense(units=26, input_dim=26, activation= "relu",name='layer_in'))
model.add(Dense(52, activation="relu"))
model.add(Dense(104, activation="relu"))
model.add(Dense(510, activation="relu"))
model.add(Dense(104, activation="relu"))
model.add(Dense(52, activation="relu"))
model.add(Dense(1))

model.summary()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

''''
model compile
'''

model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["mae"])

''''
model train
'''
print("Starting training ")

h=model.fit(X_train, Y_train, epochs=100, batch_size=100000, validation_split=0.2)

print("Training finished \n")

for i in range(epochs):
    if i % 1 == 0:
      los = h.history['loss'][i]
      mae = h.history['mean_absolute_error'][i] 
      val_mae=h.history['val_mean_absolute_error'][i]
      val_loss=h.history['val_loss'][i]
      print("epoch: %5d loss = %0.4f mae = %0.2f" \
        % (i, los, mae))
      print("epoch: %5d loss = %0.4f mae = %0.2f (Validation)" \
        % (i, val_loss, val_mae))

#saving model
model.save('model_V1-MissingPoints.h5')


fig1=plt.figure()
plt.plot(h.history['mean_absolute_error'])
plt.plot(h.history['val_mean_absolute_error'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
plt.savefig('NNModel_MAE_Plot.png')

# summarize history for loss
fig2=plt.figure()
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss(MSE)')
plt.ylabel('Loss(MSE)')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
plt.savefig('NnModel_Loss_Plot.png')



