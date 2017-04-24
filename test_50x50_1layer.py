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
col_saver = tf.train.import_meta_graph('./colModel\colSliceModel.meta')
col_saver.restore(sess, "./colModel/colSliceModel")

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

'''
os.chdir('D:/공유폴더/X1')
files = ['./X1_S24006BSA0100089_OK_2_135941_25528081 S24006BSA0100089.bmp',
         './X1_S24006BSA0100538_OK_1_192030_44776717 S24006BSA0100538.bmp',
         './X1_S24006BSA0100960_OK_2_212823_52449111 S24006BSA0100960.bmp',
         './X1_S24006BSA0102693_OK_2_052817_81243448 S24006BSA0102693.bmp',
         './X1_S24006BTA0100374_OK_1_081718_91384293 S24006BTA0100374.bmp',
         './X1_S24006BTA0100829_OK_2_095248_97114272 S24006BTA0100829.bmp',
         './X1_S24006BTA0100836_OK_1_095118_97024853 S24006BTA0100836.bmp',
         './X1_S24006BTA0101121_OK_2_112319_102545946 S24006BTA0101121.bmp',
         './X1_S24006BTA0101139_OK_2_112818_102844548 S24006BTA0101139.bmp'
         ]
'''

#os.chdir('D:/test_50x50_1/error')
#files = ['X1_S24006BSA0101366_OK_2_000328_61754290 S24006BSA0101366.bmp']

filelist = []
for file in files:

    img = imgResize(file)

    input_img = np.zeros((496, 2500))

    for i in range(0, 496):
        tmp = img[:, i:i + 5]
        tmp = np.concatenate(tmp)
        tmp = np.reshape(tmp, (50, 50), order='F')
        tmp = np.reshape(tmp, (1, 2500)) / 255
        input_img[i, :] = tmp

    start = time.time()

    col_result = sess.run(cr, feed_dict={x: input_img})
    filelist.append(col_result)
    zoo = []
    ott = []

    for i in range(len(col_result)/2, 0 , -1):
        if all(np.equal(col_result[i: i + 5], np.array([0, 1, 1, 1, 1]))):
            zoo.append(i)
            break

    for i in range(len(col_result) / 2, len(col_result)):
        if all(np.equal(col_result[i: i + 3], np.array([1, 2, 2]))) or \
            all(np.equal(col_result[i: i + 3], np.array([0, 2, 2]))):
            ott.append(i)
            break

    end = time.time()
    print(end - start)


    img[:, zoo[0] - 2] = 0
    img[:, ott[-1] + 2] = 0
    imsave("D:/test_50x50_11_stride43/" + file.split("/")[-1] + ".jpg", img)

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