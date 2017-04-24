from scipy.misc import imsave, imresize
from PIL import Image
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime
import time


def imgResize(file):
    img = Image.open(file)
    img = np.reshape(np.array(img.getdata()), (1040, 1040))
    img = img[40:1040, 40:1040]
    img = imresize(img, (500, 500), interp="nearest")
    return img


# row_sess = tf.InteractiveSession()
# row_saver = tf.train.import_meta_graph('C:/Users/Administrator/Documents/MDB team/SKI Battery/Python Code/SKIBatteryImageSlicing/rowModel/rowSliceModel.meta')
# row_saver.restore(row_sess, "C:/Users/Administrator/Documents/MDB team/SKI Battery/Python Code/SKIBatteryImageSlicing/rowModel/rowSliceModel")
#
# tf.get_default_graph().as_graph_def()
#
# row_x = row_sess.graph.get_tensor_by_name("input:0")
# row_y_conv = row_sess.graph.get_tensor_by_name("output:0")
# row_keep_prob = row_sess.graph.get_tensor_by_name("keep_prob:0")
# row_cr = tf.argmax(tf.nn.softmax(row_y_conv),1)
#
#
# col_sess = tf.InteractiveSession()
# col_saver = tf.train.import_meta_graph('C:/Users/Administrator/Documents/MDB team/SKI Battery/Python Code/SKIBatteryImageSlicing/colModel/colSliceModel.meta')
# col_saver.restore(col_sess, "C:/Users/Administrator/Documents/MDB team/SKI Battery/Python Code/SKIBatteryImageSlicing/colModel/colSliceModel")
#
# tf.get_default_graph().as_graph_def()
#
# col_x = col_sess.graph.get_tensor_by_name("input:0")
# col_y_conv = col_sess.graph.get_tensor_by_name("output:0")
# col_keep_prob = col_sess.graph.get_tensor_by_name("keep_prob:0")
# col_cr = tf.argmax(tf.nn.softmax(col_y_conv),1)


g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    row_sess = tf.Session()
    row_saver = tf.train.import_meta_graph(
        'C:/Users/Administrator/PycharmProjects/ski_gobber/rowModel/rowSliceModel.meta')
    row_saver.restore(row_sess, "C:/Users/Administrator/PycharmProjects/ski_gobber/rowModel/rowSliceModel")

    tf.get_default_graph().as_graph_def()

    row_x = row_sess.graph.get_tensor_by_name("input:0")
    row_y_conv = row_sess.graph.get_tensor_by_name("output:0")
    row_keep_prob = row_sess.graph.get_tensor_by_name("keep_prob:0")
    row_cr = tf.argmax(tf.nn.softmax(row_y_conv), 1)

with g2.as_default():
    col_sess = tf.Session()
    col_saver = tf.train.import_meta_graph(
        'C:/Users/Administrator/PycharmProjects/ski_gobber/colModel/colSliceModel.meta')
    col_saver.restore(col_sess, "C:/Users/Administrator/PycharmProjects/ski_gobber/colModel/colSliceModel")

    tf.get_default_graph().as_graph_def()

    col_x = col_sess.graph.get_tensor_by_name("input:0")
    col_y_conv = col_sess.graph.get_tensor_by_name("output:0")
    col_keep_prob = col_sess.graph.get_tensor_by_name("keep_prob:0")
    col_cr = tf.argmax(tf.nn.softmax(col_y_conv), 1)

import os
os.chdir("D:/공유폴더/")
p = ["./N/",
     "./O/",
     "./X1/",
     "./X2/"]

# filenum = 100
fileList = []
for path in p:
    onlyfiles = [path + f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if ".bmp" in f]
    fileList += onlyfiles

len(fileList)
csvlist = []
for file in fileList:
    csvlist.append(file)

    img = imgResize(file)

        ##### col #####
    input_img = np.zeros((int(496 / 4), 2500))

    for i in range(0, (496), 4):
        tmp = img[:, i:i + 5]
        tmp = np.concatenate(tmp)
        tmp = np.reshape(tmp, (50, 50), order='F')
        tmp = np.reshape(tmp, (1, 2500)) / 255
        input_img[int(i / 4), :] = tmp

    col_result = col_sess.run(col_cr, feed_dict={col_x: input_img})

    zoo = []
    ott = []

    for i in range(int(len(col_result) / 2), 0, -1):
        if all(np.equal(col_result[i: i + 5], np.array([0, 1, 1, 1, 1]))):
            zoo.append(i)
            break

    for i in range(int(len(col_result) / 2), len(col_result)):
        if all(np.equal(col_result[i: i + 3], np.array([1, 2, 2]))) or \
                all(np.equal(col_result[i: i + 3], np.array([0, 2, 2]))):
            ott.append(i)
            break

    img[:, 4 * zoo[0] + 5] = 0
    img[:, 4 * ott[-1] + 3] = 0

    csvlist.append( 2 * ( 4 * zoo[0] + 5) + 40 )   # left
    csvlist.append( 2 * ( 4 * ott[-1] + 3) + 40 )    # right

    ##### row #####
    input_img = np.zeros((124, 2500))

    for i in range(0, 496, 4):
        tmp = img[i: i + 5, :]
        tmp = np.reshape(tmp, (50, 50), order="F")
        tmp = np.reshape(tmp, (1, 2500)) / 255
        input_img[int(i / 4), :] = tmp

    row_result = row_sess.run(row_cr, feed_dict={row_x: input_img, row_keep_prob: 1.0})

    zoo = []
    ott = []

    for i in range(len(row_result) - 3):
        if all(np.equal(row_result[i: i + 3], np.array([0, 1, 1]))):
            zoo.append(i)

        if all(np.equal(row_result[i: i + 3], np.array([1, 2, 2]))):
            ott.append(i)

    img[4 * zoo[0] - 10, :] = 0
    img[4 * ott[-1] + 30, :] = 0

    csvlist.append( 2* (4 * zoo[0] - 10) + 40)  # up
    csvlist.append( 2 * (4 * ott[-1] + 30) + 40)  # down

        # print("total time : " + str( end - start ) )
        #imsave("D:/test_50x50_module_result/" + file.split("/")[-1], img)

len(csvlist)
csvarray = np.reshape(np.array(csvlist),(len(fileList),5))

import pandas as pd
df = pd.DataFrame(csvarray)
df.columns = ['filename','left','right','up','bottom']

df.to_csv("D:/roi_test_0422.csv", index = False)