__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
#        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_glob = image_dir + "/images/" + directory + '/*.' + 'jpg'        
        
        ### ====== Modifation : \ or \\ --> / 
        file_globbed = glob.glob(file_glob)
        file_globbed_slash = [ file_glo.replace("\\","/") for file_glo in file_globbed ]
        file_list.extend(file_globbed_slash)

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                
               
                ### ====== Modifation : \ or \\ --> / 

#                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                
                annotation_file = image_dir + "/annotations/" + directory + '/'+ filename + '.png'
                
                
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                    
                    
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list


#directory = 'training'
#image_dir =   'Data_zoo/MIT_SceneParsing/ADEChallengeData2016'
#import os 
#
#
#file_glob = image_dir + "/images/" + directory + '/*.' + 'jpg'
#file_glob
#globbed = glob.glob(file_glob)
#globbed
#
#filename = os.path.splitext(file_glob.split("/")[-1])[0]
#
#
#
##f         :   Data_zoo/MIT_SceneParsing/ADEChallengeData2016\images\training\ADE_train_00009281.jpg
#f         :   Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/ADE_train_00009281.jpg
#               
#filename  :   ADEChallengeData2016\images\training\ADE_train_00009281 # f 에서 .jpg 없는것 
#
#
#
#annotation_file :   Data_zoo/MIT_SceneParsing/ADEChallengeData2016\
#                    annotations\
#                    training\
#                    
#                    ADEChallengeData2016\images\training\ADE_train_00009281.png
#
#
#Annotation file not found for ADEChallengeData2016\images\training\ADE_train_00009281 - Skipping

#
#
#image_dir :   Data_zoo/MIT_SceneParsing/ADEChallengeData2016
#f          :   Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training\ADE_train_00007983.jpg
# filename  :   training\ADE_train_00007983
# annotation_file :   Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/training/training\ADE_train_00007983.png
#
#
#Annotation file not found for training\ADE_train_00007983 - Skipping










