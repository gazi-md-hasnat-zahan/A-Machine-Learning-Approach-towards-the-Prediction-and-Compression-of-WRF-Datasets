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
##trial
def read_data(file_name):
    dataset = np.genfromtxt(file_name, dtype = "str", delimiter = ",")
    headers = dataset[0]
    data = dataset[1:,3:29].astype("float").reshape((699, 639, 26))
#     print('reading...' + file_name)
    return data

def get_neighbors(data,days,features):
    data=data.reshape((days,699,639,features))
    temp = np.zeros((days, 699+ 1 * 2, 639 + 1 * 2, features))
    temp[:, 1:temp.shape[1] - 1, 1: temp.shape[2] - 1] = data

    result = np.zeros((days,699, 639, features, 3 * 3))
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
    for j in range(26):
        temp = data[:, j]
        temp = np.interp(temp, (temp.min(), temp.max()), (0, 1))
        data[:, j] = temp
    data = data.astype("int")
    return data
####for svr
####final

#Take Training files first

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

'''
#Take Summer Testing files

for file_name in os.listdir(file_path):
    if (file_name[-3:]=='csv' and file_name[0:7] in ['2013-06','2014-07','2015-08']):
        print('reading...' + file_name)
        result2 = read_data(file_name).reshape((699* 639, 26))
        result2=level_data(result2) 
        result = np.concatenate((result, result2))     
        
#Take Winter Testing files

for file_name in os.listdir(file_path):
    if (file_name[-3:]=='csv' and file_name[0:7] in ['2013-01','2014-12','2015-02']):
        print('reading...' + file_name)
        result2 = read_data(file_name).reshape((699* 639, 26))
        result2=level_data(result2) 
        result = np.concatenate((result, result2))  
'''

# print(result.shape)
# print(result.size)

y=result[:,25]
x=np.delete(result,25,1)
print('Actual X shape: ' , x.shape)
print('Y shape: ' , y.shape) 

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
regr = LinearRegression()
regr.fit(X_reduced,y)
ypc=regr.predict(X_reduced)
print('The R2 score is: ', r2_score(ypc,y))            

print('calling get_neighbour function...')
result=get_neighbors(X_reduced,26,2)
#np.savetxt("final_pca_reduced_data.csv", result, delimiter=",")

##NN start
print('Starting gathering training samples')
##NN start

#for all locations, take ALBEDO,EMISS and reshape that as location vs neighbors of ALBEDO and EMISS
X_train = result.reshape(699 * 639 * 26, 2 * 3 * 3)
# print(X_train[100:101, :])

#Get the SoilMois Data , last 3 is the 3rd feature, which is Soil moisture
Y_train = y

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print("10 Fold Cross Validation Starting...")
input_dim=18

kf = KFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
cvscores2 = []
for train, test in kf.split(X_train,Y_train):
  # create model
    model=Sequential()
    model.add(Dense(units=input_dim, input_dim=input_dim, activation= "relu",name='layer_in'))
    model.add(Dense(52, activation="relu"))
    model.add(Dense(104, activation="relu"))
    model.add(Dense(52, activation="relu"))
    model.add(Dense(1))
    # Compile model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
    # Fit the model
    model.fit(X_train[train], Y_train[train], epochs=100, batch_size=100000, verbose=0)
    # evaluate the model
    scores = model.evaluate(X_train[test], Y_train[test], verbose=0)
    print("%s: %.2f" % (model.metrics_names[0], scores[0]))
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))
    cvscores.append(scores[0])
    cvscores2.append(scores[1])

print("Average CV Loss(MSE): %.2f with St. Dev: (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
print("Average CV MAE: %.2f with St. Dev: (+/- %.2f)" % (np.mean(cvscores2), np.std(cvscores2)))

fig1=plt.figure()
plt.plot(cvscores)
plt.title('Cross Validation Plot')
plt.ylabel('Loss(MSE)')
plt.xlabel('Model')
plt.show()
plt.savefig('CVPlot_Loss-NNModel_v2_NOsoil.png')

fig2=plt.figure()
plt.plot(cvscores2)
plt.title('Cross Validation Plot')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Model')
plt.show()
plt.savefig('CVPlot_MAE-NNModel_v2_NOsoil.png')


# Final Model
print('Starting NN...')
epochs=1
model=Sequential()
model.add(Dense(units=input_dim, input_dim=input_dim, activation= "relu",name='layer_in'))
model.add(Dense(52, activation="relu"))
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

print(h.history.keys())
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
        
#eval = model.evaluate(X_test, Y_test, verbose=0)
#print("\nEvaluation on test data: \n loss(mse) = %0.4f \n mae = %0.2f" % (eval[0], eval[1]) )

#saving model
model.save("model_latest_V2-NoSoil.h5")

#print(h.history.keys())
# this returns array of predicted results from test_data
fig1=plt.figure()
plt.plot(h.history['mean_absolute_error'])
plt.plot(h.history['val_mean_absolute_error'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
plt.savefig('NNModel_V2_no soil_MAE_Plot.png')

# summarize history for loss
fig2=plt.figure()
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss(MSE)')
plt.ylabel('Loss(MSE)')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
plt.savefig('NnModel_V2_no soil_Loss_Plot.png')



