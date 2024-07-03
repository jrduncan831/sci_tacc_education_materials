import os 
import numpy as np
from dask.distributed import Client
from dask import delayed
import dask.array as da
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imread

def read_resize(path, size=(224,224)):
    return resize(imread(path),size)

def read_resize_gray(path, size):
    return rgb2gray(resize(imread(path), size))

def get_data(path, size=(224,224), gray=False):
    '''
    takes path to directory of files and gets list of delayed read_resize
    
    returns list of delayed objects
    '''
    if gray:
        files = os.listdir(path)
        if '.DS_Store' in files :
            files.remove('.DS_Store')
        if '.ipynb_checkpoints' in files:
            files.remove('.ipynb_checkpoints')
        data = [delayed(read_resize_gray)(path+file,size) for file in files]
        data = [da.from_delayed(x, shape=size,dtype=np.float32) for x in data]
    else:
        files = os.listdir(path)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        if '.ipynb_checkpoints' in files:
            files.remove('.ipynb_checkpoints')
        data = [ delayed(read_resize)(path+file,size) for file in files]
        size = list(size)
        size.append(3)
        size = tuple(size)
        data = [da.from_delayed(x, shape=size,dtype=np.float32) for x in data]
    return data 

def get_all_data(path_list_train, path_list_test, classes, size=(224,224),gray=False):
    '''
    takes list of paths to directories
    
    returns numpy array of all image data
    '''
    X_train,X_test,y_train,y_test =  [],[],[],[]
    for path,class_ in zip(path_list_train,classes):
        print(path)
        data = get_data(path,size=size,gray=gray)
        X_train += data
        y_train += len(data)*[class_]
    for path,class_ in zip(path_list_test,classes):
        print(path)
        data = get_data(path,size=size,gray=gray)
        X_test += data
        y_test += len(data)*[class_]
    return da.stack(X_train), da.stack(X_test), y_train, y_test



