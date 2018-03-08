"""
legacy
"""

import pandas as pd
import numpy as np
import pprint
import os
from datetime import datetime
import glob
import pandas as pd
import pickle
import time
import importlib

import pandas as pd
import sys, getopt
import numpy as np
import pprint
import os
from datetime import datetime
import glob
import pandas as pd

path=['emg_data','ipa_emg','severalObjects']
folds = 4

def selectFileNames(dataList,name):
    return [os.path.basename(f)
             for f in dataList
             if name in os.path.basename(f.split('-',1)[0]) and 'aux' not in os.path.basename(f.split('-',1)[0]) and 'imu' not in os.path.basename(f.split('-',1)[0])]

def expandVectors(vec):
    return np.expand_dims(np.expand_dims(vec, axis=1),axis=-1)

def augmentData(x,y,nb_roll,steps):
    assert len(x.shape) > 1
    x_cp = x
    y_cp = y

    for i in range(-nb_roll,nb_roll,steps):
        x_cp = np.vstack((x_cp,np.roll(x,i,axis=1)))
        y_cp = np.vstack((y_cp,y))

    return x_cp,y_cp

def augmentData2(x,y,shiftList=[0.1,0.5,1],outlen=8):
    assert len(x.shape) > 1
    x_cp = x
    y_cp = y

    datapoint_number = np.arange(0,8)
    period = 8
    x_cp = list()
    y_cp = list()

    if 0 not in shiftList:
        shiftList.append(0)

    shiftList.sort(key=abs)


    for shift in shiftList:
        shiftVec = np.arange(-1*(outlen-8)/2,8 + (outlen-8)/2) + shift
        for i,element in enumerate(x):
            x_cp.append(np.interp(shiftVec,datapoint_number,element,period=period))
            y_cp.append(y[i])


    return np.vstack(x_cp),np.vstack(y_cp)

def padStuff(matrix,expsize=(2,2),axis=0,mode='wrap'):
    dim = len(matrix.shape)
    tup1 = [(0,0) for _ in range(0,dim)]
    tup1[axis] = expsize
    return np.pad(matrix,tup1,mode)

def kfold(x,y,folds=2):
    assert (len(x) == len(y))
    randArray = np.random.randint(0,folds,len(y))
    out = list()
    for i in range(0,folds):
        out.append((x[randArray == i] , y[randArray == i]))
    return out

dataList = glob.glob(os.path.join(os.path.expanduser('~'),*path,'**'),recursive=True)

fileNameDict = dict()
categories =  ['empty','screwdriver','hammer','weightP']
persons = ['daniel','tobias']

for p in persons:
    fileNameDict[p] = dict()
    for c in categories:
        fileNameDict[p][c] = selectFileNames(selectFileNames(dataList,p),c)

fileDict = dict()
for p in fileNameDict:
    fileDict[p] = dict()
    for f in fileNameDict[p]:
        tmpList = list()
        for k in fileNameDict[p][f]:
            tmpList.append(pd.read_csv(os.path.join(os.path.expanduser('~'),*path,k), skiprows=1, header=None, delim_whitespace=True))
        fileDict[p][f] = pd.concat(tmpList)


tmpList = list()
tmpList2 = list()
for p in fileDict:
    for j,f in enumerate(fileDict[p]):
        for i, r in fileDict[p][f].iterrows():
            tmpList.append(np.array(r[1:]))
            tmpList2.append(j)

tmp1 = np.array(tmpList)
tmp2 = np.array(tmpList2)

print("\n--------------\nLoading done, now preping data\n--------------\n")

folder = kfold(tmp2,tmp1,folds)


from keras.utils.np_utils import to_categorical
augData = list()
for r in folder:
    tt = augmentData2(r[1],r[0],list(np.linspace(-2,2,3)),10)
    augData.append( (expandVectors(tt[0]) ,to_categorical(tt[1],num_classes=4) ) )

folder2 = list()
for i,r in enumerate(folder):
    tt = augmentData2(r[1],r[0],[0],10)
    folder2.append([tt[0],tt[1]])

# Load everything keras
from keras.layers.convolutional import Conv1D
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected2D
from keras.layers import Conv2D, MaxPooling2D, LSTM, Conv1D, MaxPooling1D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import EarlyStopping


class default_config:
    loadstuff = False
    filename = 'stuff2'
    explorationStruct = { 'hidden_layer':      [1,3,1],
                          'neurons_per_layer': [100,201,25],
                          'filter_layer':      [50,151,50],
                          'kernel_size':       [4,6,1]
                          }
    model_params = { 'model':'CNN' # or CNN

                    }
    convFilterDict = { "conv":  {
                                "filters":       30,
                                "kernel_size":   (1,6),
                                "input_shape":   (1,10,1)
                                },
                       "maxpool":{"pool_size":(1,5)}
                     }
    output_classes = 4
    activation = 'relu'

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.accuracy = []
    def on_epoch_end(self,epoch,logs={}):
        self.accuracy.append(logs.get('acc'))


config = default_config
saveList = list()
tb4 = datetime.now()
count = 0


model_architecture = []
for hidden_layer in range(*config.explorationStruct['hidden_layer']):
    for neurons_per_layer in range(*config.explorationStruct['neurons_per_layer']):
        for filter_layer in range(*config.explorationStruct['filter_layer']):
            for kernel_size in range(*config.explorationStruct['kernel_size']):
                model_architecture.append([hidden_layer,neurons_per_layer,kernel_size,filter_layer, 100] )

startlen = len(model_architecture)

# calculate input_shape
config.convFilterDict["conv"]["input_shape"] = augData[0][0].shape[1:]

if config.loadstuff:
    l = glob.glob("/home/myo/network_tests/*")
    try:
        l = [i for i in l if config.filename and '_im' in i]
        l.sort(key=lambda k: k[-18:-3])
        model_architecture = pickle.load(open(l[-1],'rb'))
        print('loading model of length ' + str(len(model_architecture)) + " / " + str(startlen))
    except:
        print('Couldnt load model architecture list, using default')

while len(model_architecture) > 10:
    scaler = 1- len(model_architecture)/(startlen)
    pat = int(45*scaler)+5
    delta = -0.01*(scaler -1)
    epos = int(390*scaler +10)
    print('epos: ' + str(epos))
    print('pat: ' + str(pat))
    print('delta: ' + str(delta))

    for j,architecture in enumerate(model_architecture):
        print("At Model " + str(j) + " / " + str(len(model_architecture)))

        tst = datetime.now()

        hidden_layers = max(0,int(architecture[0]))
        neurons_per_layer = max(1,int(architecture[1]))
        kernel_size = max(1,int(architecture[2]))
        filter_layer = max(1,int(architecture[3]))

        print("Architecture: ")
        print("Hidden Layers: ", hidden_layers)
        print("Neurons:       ", neurons_per_layer)
        print("ConvFilters:   ", kernel_size)
        print("FilterLayer:   ", filter_layer)

        config.convFilterDict["conv"]["filters"] = filter_layer
        config.convFilterDict["conv"]["kernel_size"] = (1,kernel_size)
        config.convFilterDict["maxpool"]["pool_size"] = (1, config.convFilterDict["conv"]["input_shape"][1] - config.convFilterDict["conv"]["kernel_size"][1] + 1)

        # Network architecture
        output_classes    = config.output_classes
        activation        = config.activation

        model = Sequential()
        if config.model_params['model'] == 'CNN':
            model.add(Conv2D(**config.convFilterDict["conv"]))
            model.add(MaxPooling2D(**config.convFilterDict["maxpool"]))
        elif config.model_params['model'] == 'MLP':
            model.add(Dense(neurons_per_layer,input_shape=config.convFilterDict["conv"]["input_shape"]))
        model.add(Activation(config.activation))
        model.add(Dropout(0.2))

        model.add(Flatten())

        for i in range(0,hidden_layers):
            model.add(Dense(neurons_per_layer))
            model.add(Activation(config.activation))
            model.add(Dropout(0.2))

        model.add(Dense(len(fileDict[persons[0]]),activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], class_mode="sparse" )
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss',patience=pat,min_delta=delta)
        history = AccuracyHistory()

        # learn model and plot show progress
        score = 0
        for jj,_ in enumerate(augData):
            x_train_exp = np.vstack([n[0] for i,n in enumerate(augData) if i != jj])
            y_train_exp = np.vstack([n[1] for i,n in enumerate(augData) if i != jj])
            model.fit(x_train_exp,y_train_exp,epochs = epos, batch_size = 1500, validation_split=0.2, shuffle=True, callbacks = [early_stopping,history], verbose=0)
            score += model.evaluate(expandVectors(folder2[jj][0]), to_categorical(folder2[jj][1],num_classes=4), batch_size=1000, verbose=0)[1]
        architecture[4] = score/folds
        tbn = datetime.now()

        # Print pretty stuff
        print("\n------------------------")
        print("timestamp: ", tbn , " | Time diff: ", tbn - tb4)
        tb4 = tbn
        print("Architecture: ")
        print("Hidden Layers: ", hidden_layers)
        print("Neurons:       ", neurons_per_layer)
        print("ConvFilters:   ", kernel_size)
        print("\nScore: ", architecture[4])
        print("\n")

        timestr = time.strftime("%Y%m%d-%H%M%S")
        pickle.dump(model_architecture,  open('/home/myo/network_tests/' + config.filename + timestr, 'wb'))



    model_architecture.sort(key = lambda L: L[4],reverse=True)
    nb_rmArch = int(max(len(model_architecture)*0.5,1))
    print("worst:")
    print(model_architecture[0][4])
    print("best:")
    print(model_architecture[-1][4])
    del model_architecture[:nb_rmArch]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    pickle.dump(model_architecture,  open('/home/myo/network_tests/' + config.filename + timestr + '_im', 'wb'))


# save last file, if finished
timestr = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(model_architecture,  open('/home/myo/network_tests/' + config.filename + timestr + "_final", 'wb'))
