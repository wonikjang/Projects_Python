import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
import tensorflow as tf
import pickle
from scipy.misc import toimage

### 1-1 Save 200 gabor masks of panda data frame ( Panda is the fastest one ) & convert it as tf objects
# -> Output: array of tf objects

def gabor(gaborfiles):

    w = np.array([np.array(pd.read_csv(file, index_col=None, header=-1)) for file in gaborfiles])

    wlist = [] # 200 gabor wavelet
    for i in range(len(w)):
        chwi = np.reshape(np.array(w[i], dtype='f'), (w[i].shape[0], w[i].shape[1], 1, 1))
        wlist.append( tf.Variable(np.array(chwi)) )

    wlist2 = np.array(wlist)

    return wlist2


### 1-2 Import & resize image  (convert into array and change data type as float32)
def image_resize(allimage, shape):

    badlist = []; refined = []
    for i in range(len(allimage)):
        badlist.append(Image.open(allimage[i]))
        refined.append(badlist[i].resize(shape, Image.ANTIALIAS))

    refinedarr = np.array([np.array((fname), dtype='float32') for fname in refined])

    imgreshape = refinedarr.reshape(-1,refinedarr.shape[1],refinedarr.shape[2],1)

    return imgreshape


### 1-3 Convolution b/w allimages and 200 gabor masks

def conv_model(imgresize, wlist, stride):
    imgresize3 = []
    for i in range(len(wlist)):
        imgresize3.append(tf.nn.conv2d(imgresize, wlist[i],
                                       strides=[1, stride, stride, 1], padding='SAME'))

    imgresize5 = tf.transpose(imgresize3, perm=[0, 1, 2, 3, 4])

    return imgresize5


### 2-0 Result save by pickle and the name of image

def save_img_pickle_name(dir_path, convresult, filename, batch_size):

    convarray = np.array(convresult)
    convarray2 = convarray.reshape(convarray.shape[0], convarray.shape[1], convarray.shape[2],
                                   convarray.shape[3] * convarray.shape[4])

    dirnum = len(convresult)
    folders = [];
    training_batch = zip(range(0, len(filename), batch_size), range(batch_size, len(filename) + 1, batch_size))

    for start, end in training_batch:
        folders.append("image" + str(start) + '~' + str(end))

    for folder in folders:
        os.mkdir(os.path.join(dir_path, folder))
        for i in range(convarray2.shape[0]):  # i index image
            if folders.index(folder) == i:
                test1 = []
                for j in range(convarray2.shape[1]):  # create (200,1024) array per image
                    test1.append(np.vstack(convarray2[i][j, :, :]))
                    list1 = filename[20 * i + j]
                    with open(os.path.join(dir_path, folder, '%s.pickle' % list1 ), 'wb') as outfile:
                        pickle.dump(test1[j], outfile, pickle.HIGHEST_PROTOCOL)

### Reshape from (140, 200, 1024) to (140, 200, 32, 32) array, and convert it as image
def reshape_imported(imgres):

    imgre = []
    for i in range(len(imgres)):
            imgre.append(np.reshape(imgres[i], (200, 32, 32)))
    imgre = np.array(imgre)

    return imgre

def save_gabor_image(save_path, imgname, imgre, gaborname):

    for folder in imgname:
        os.mkdir(os.path.join(save_path, folder))
        for i in range(len(imgre)):  # i index image
            if imgname.index(folder) == i:
                test1 = []
                for j in range(imgre.shape[1]):  # 200 * (32,32)
                    test1.append(toimage( Image.fromarray( imgre[i][j][ :, :]) ) )
                    list1 = gaborname[j]
                    test1[j].save(os.path.join(save_path, folder, '%s.png' % list1))





#### ------------------------------------------------------------- ####



### 2-1 Result save by using pickle (1 file per folder): collect image and save

def save_img_pickle(dir_path, convresult):

    convarray = np.array(convresult)
    convarray2 = convarray.reshape(convarray.shape[0], convarray.shape[1], convarray.shape[2],convarray.shape[3] * convarray.shape[4])
    # (7, 20, 200, 32, 32, 1)

    dirnum = len(convresult)
    folders = [];
    dir = list(range(dirnum))
    for i in dir: folders.append("gabor" + str(i))

    for folder in folders:
        os.mkdir(os.path.join(dir_path, folder))
        for i in range(convarray2.shape[0]):  # i index image
            if folders.index(folder) == i:
                test1 = []
                for j in range(convarray2.shape[1]):  # create (200,1024) array per image
                    test1.append(np.vstack(convarray2[i][j, :, : ]))
                    with open(os.path.join(dir_path, folder, 'gabor_img%i.pickle' % j), 'wb') as outfile:
                        pickle.dump(test1[j], outfile, pickle.HIGHEST_PROTOCOL)

### 2-2 Result save by using pickle (20 files per folder) : per image save

def save_img_pickle2(dir_path, convresult):

    convarray = np.array(convresult)
    convarray2 = convarray.reshape(convarray.shape[0], convarray.shape[1], convarray.shape[2],
                                                       convarray.shape[3] * convarray.shape[4])

    dirnum = len(convresult)
    folders = [];
    dir = list(range(dirnum))
    for i in dir: folders.append("gabor" + str(i))

    for folder in folders:
        os.mkdir(os.path.join(dir_path, folder))
        for i in range(convarray2.shape[0]):  # i index image
            if folders.index(folder) == i:
                test1 = []
                for j in range(convarray2.shape[1]):  # create (200,1024) array per image
                    test1.append(np.vstack(convarray2[i][j,: ,: ]))
                    with open(os.path.join(dir_path, folder, 'pickle%i.pickle' % j),'wb') as outfile:
                        pickle.dump(test1[j], outfile, pickle.HIGHEST_PROTOCOL)


### 2-3 Result save by using numpy

def save_img(dir_path, convresult):

    convarray = np.array(convresult)
    convarray2 = convarray.reshape(convarray.shape[0], convarray.shape[1], convarray.shape[2],convarray.shape[3] * convarray.shape[4])

    dirnum = len(convresult)
    folders = [];
    dir = list(range(dirnum))
    for i in dir: folders.append("gabor" + str(i))

    for folder in folders:
        os.mkdir(os.path.join(dir_path, folder))
        for i in range(convarray2.shape[0]):  # i index image
            if folders.index(folder) == i:
                test1 = []
                for j in range(convarray2.shape[1]):  # create (200,1024) array per image
                    test1.append(np.vstack(convarray2[i][j, :, :]))
                    np.savetxt(os.path.join(dir_path, folder, 'gabor_img%i.txt' % j), test1[j])


### 3-1 import multiple files from multiple folders in the path (mathch with pickle2 function)
def img_import(path):

    folder = []
    for foldername in os.listdir(path): folder.append(foldername)

    finpath = []
    for i in range(len(folder)): finpath.append(os.path.join(path + folder[i] + '/*.pickle'))

    img1 = []
    for i in range(len(finpath)):
        what = finpath[i]
        files = glob.glob(what)
        for filename in files:
            with open(filename, 'rb') as f:
                img1.append(pickle.load(f))

    return img1
