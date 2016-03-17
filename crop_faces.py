from hashlib import sha256
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


def setup():
    if not os.path.exists('cropped'):
        os.mkdir('cropped') 
        
def crop_all(colour=False):
    setup()
    for filename in os.listdir('uncropped'):
        hash_from_file = sha256(file.read(open('uncropped/' + filename))).hexdigest()
        if filename.split('.')[2] == hash_from_file:   
            x1 = int(filename.split('.')[1].split(',')[0])
            y1 = int(filename.split('.')[1].split(',')[1])
            x2 = int(filename.split('.')[1].split(',')[2])
            y2 = int(filename.split('.')[1].split(',')[3])
            # print filename
            # print x1, y1, x2, y2
            
            new_name = filename.split('.')[0] + '.' + filename.split('.')[3]
        
            try:
                im = imread('uncropped/'+filename, 1)[y1:y2, x1:x2]
                if not colour:
                    gray()
                resized_im = imresize(im, (32, 32))
                imsave('cropped/' + new_name, resized_im)
            except IOError:
                print 'IOError: Skipped ' + new_name
                pass
            except ValueError:
                print 'ValueError: Skipped ' + new_name
                pass