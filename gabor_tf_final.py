import glob
import tensorflow as tf
import gabor_tf_function_final as gb


# gabor mask
gabor1 = glob.glob('gabor_wavelet_filter_bank/*.csv')
mask = gb.gabor(gabor1)

gaborname = [s.strip('gabor_wavelet_filter_bank\\') for s in gabor1 ]
gaborname = [s.strip('.csv') for s in gaborname ]


# Image
imglist = glob.glob('*.jpg') # len(imglist)
imgresize = gb.image_resize(imglist, (128, 128)) # imgresize.shape # (N,256,256)

imglistfin = [s.strip('.jpg') for s in imglist]


# convolution
X = tf.placeholder("float", [None, 128, 128, 1])
train = gb.conv_model(X, mask, 4)

batch_size = 20

# Tensorflow Sessioin
from datetime import datetime
start_time = datetime.now()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    convresult = []
    training_batch = zip(range(0, len(imgresize), batch_size), range(batch_size, len(imgresize) + 1, batch_size))
    for start, end in training_batch:
        convresult.append( sess.run(train, feed_dict={X:imgresize[start:end] }) )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) # 20 sec


# Save
dir_path = './gabor_output'
gb.save_img_pickle_name(dir_path, convresult, imgname, batch_size)


# Import multiple files from multiple folders at once
path = './gabor_output/'
imgres = gb.img_import(path)
imgre = gb.reshape_imported(imgres)

# Creat folder name as image name and save each gabor as png file
save_path = './gabor_image/'
gb.save_gabor_image(save_path, imgname, imgre, gaborname)
