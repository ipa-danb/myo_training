"""Helper functions

This section is for helper functions that are used to prepare data

Todo:
    * Move to own module if finished

"""

# various imports
import pandas as pd
import numpy as np
import pprint
import os
from datetime import datetime
import glob



def selectFileNames(dataList,name):
    """Function so select data from list by name

    This functions searches a list of datafile given with full path based on naming standards

    Parameters
    ----------
    dataList : list of strings
        list of strings with full path for each file.
    name : str
        part of the filename which should be selected.

    Returns
    -------
    list of strings with paths of selected files

    """

    return [f
             for f in dataList
             if name in os.path.basename(f.split('-',1)[0]) and 'aux' not in os.path.basename(f.split('-',1)[0]) and 'imu' not in os.path.basename(f.split('-',1)[0])]

def expandVectors(vec):
    """Expands the given vector so it is accepted by keras

    Parameters
    ----------
    vec : numpy array
        array of numpy data

    Returns
    -------
    expanded numpy array (to 3 dimensions)

    """
    return np.expand_dims(np.expand_dims(vec, axis=1),axis=-1)

def augmentData(x,y,nb_roll,steps):
    """legacy
    """
    assert len(x.shape) > 1
    x_cp = x
    y_cp = y

    for i in range(-nb_roll,nb_roll,steps):
        x_cp = np.vstack((x_cp,np.roll(x,i,axis=1)))
        y_cp = np.vstack((y_cp,y))

    return x_cp,y_cp

def augmentData2(x,y,shiftList=[0.1,0.5,1],outlen=8):
    """Function to shift data vectors to a certain output length

    This function shifts a vector x by the amounts specified in shiftList. It also interpolates between datapoints.
    It expands the vector x to length outlen. Unshifted and shifted data are concatinated and returned

    Parameters
    ----------
    x : numpy matrix <samples x recording vector>
        vector of input vectors for shifting
    y : numpy matrix <samples x 1>
        part of the filename which should be selected.

    Returns
    -------
    x_cp : x vector plus shifted data
    y_cp : y vector expanded to fit x_cp
    """
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

def kfold(x,y,folds=2):
    """legacy
    """
    #assert len(x) == len(y)
    randArray = np.random.randint(0,folds,len(y))
    out = list()
    for i in range(0,folds):
        out.append((x[randArray == i] , y[randArray == i]))
    return out


def padStuff(matrix,expsize=(2,2),axis=0,mode='wrap'):
    """legacy
    """
    dim = len(matrix.shape)
    tup1 = [(0,0) for _ in range(0,dim)]
    tup1[axis] = expsize
    return np.pad(matrix,tup1,mode)

def normalization_std(in_dict,*args):
    name = args[0]
    for person in in_dict:
        norm_data = in_dict[person][name]
        mm = np.mean(norm_data,axis=0)
        stdd = np.std(norm_data,axis=0)
        for element in in_dict[person]:
            in_dict[person][element] = (in_dict[person][element] - mm)/stdd
    return in_dict

def normalization_none(in_data,*args):
    return in_data
