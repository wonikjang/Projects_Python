from scipy.misc import imsave, imresize, imread
from PIL import Image
import tensorflow as tf
import numpy as np
import time
from os import listdir
from os.path import isfile, join


def imgResize(file):
    img = Image.open(file)
    img = np.reshape(np.array(img.getdata()), (1040, 1040))
    img = img[40:1040, 40:1040]
    img = imresize(img, (500, 500), interp="nearest")
    return img


sess = tf.InteractiveSession()
row_saver = tf.train.import_meta_graph('./colModel\colSliceModel.meta')
row_saver.restore(sess, "./colModel/colSliceModel")

tf.get_default_graph().as_graph_def()

x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
cr = tf.argmax(tf.nn.softmax(y_conv), 1)
####### 1 file test
import os
os.getcwd()
os.chdir('D:/공유폴더/X1')
import glob
files = glob.glob('./*.bmp')

stride = 4


filelist = []
for file in files:

    img = imgResize(file)

    input_img = np.zeros(( int(496/stride), 2500))

    for i in range(0, (496) , stride):
        tmp = img[:, i:i + 5]
        tmp = np.concatenate(tmp)
        tmp = np.reshape(tmp, (50, 50), order='F')
        tmp = np.reshape(tmp, (1, 2500)) / 255
        input_img[int(i/stride), :] = tmp

    start = time.time()

    col_result = sess.run(cr, feed_dict={x: input_img})
    filelist.append(col_result)
    zoo = []
    ott = []

    for i in range(len(col_result) - 3):
        if all(np.equal(col_result[i: i + 3], np.array([0, 1, 1]))):
            zoo.append(i)

        if all(np.equal(col_result[i: i + 3], np.array([1, 2, 2]))) or \
            all(np.equal(col_result[i: i + 3], np.array([0, 2, 2]))):
            ott.append(i)

    #for i in zoo:
    #   img[:, i] = 0

    #for i in ott:
    #    img[:, i] = 0

    end = time.time()
    print(end - start)
    img[:, stride*zoo[0] ] = 0
    img[:, stride*ott[-1] ] = 0
    imsave("D:/test_50x50_11_stride42/" + file.split("/")[-1] + ".jpg", img)

    '''
    if len(zoo) == 1 and len(ott) == 1:

        img[ :, zoo[0] - 5] = 0

        img[ :, ott[0] + 5,] = 0

        imsave("D:/test_50x50_11/" + file.split("/")[-1] + ".jpg", img)

    else:
        img[ :, zoo[0] - 5] = 0
        img[ :, ott[-1] + 5] = 0

        imsave("D:/test_50x50_11/" + file.split("/")[-1] + ".jpg", img)
    '''
filelist