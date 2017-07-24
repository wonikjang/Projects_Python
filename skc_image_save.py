import os
import numpy as np
import random
from PIL import Image
import PIL
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from datetime import datetime

def readfile(mypath):

    onlyfiles = [f for f in listdir(mypath) if f.endswith('.jpg') ]
    file = [mypath + s for s in onlyfiles]

    print('%d files found : ' % len(file))
    return file

def getnonlabel( tfile, p, savepath):

    #### Cionvert images and name -> arrays of X & Y
    new_width = 512;
    new_height = 64

    imagelist = []; namelist = []
    for file in tfile:
        img0 = Image.open(file)
        img0 = img0.resize((new_width, new_height), resample = Image.NEAREST)
        img0 = np.array(img0)
        img0 = img0 / 255.
        img0 = img0.astype(np.float32)

        fname = file.split(p, 1)[1]
        fname = fname.replace(".jpg", ".csv")

        savingpath = os.path.join ( savepath , fname )

        np.savetxt( savingpath , img0, delimiter=",")

        imagelist.append(img0)
        namelist.append(fname)

    return namelist, imagelist

start = datetime.now()

# Call files
path = '/Users/wonikJang/Desktop/sk_projects/SKC_SmartFacoryThickData/imgunk_sample/'
files = readfile(path)

# Change files into array with name of them
# Save array as csv file into certain path
phead = '/Users/wonikJang/Desktop/sk_projects/SKC_SmartFacoryThickData/imgunk_sample/'

savepath = '/Users/wonikJang/Desktop/sk_projects/SKC_SmartFacoryThickData/imgunk_csv/'

name, image = getnonlabel(files ,phead, savepath )

end = datetime.now()

print("running time : " +str(end-start))
