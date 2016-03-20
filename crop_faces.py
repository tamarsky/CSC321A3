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


def setup(dirname):
    '''
    if part 1: dirname = 'cropped'
    if part 2: dirname = 'cropped_rgb'
    '''
    if not os.path.exists(dirname): #'cropped' or 'cropped_rgb'
        os.mkdir(dirname) 
        
def crop_all(part):
    if part ==1 :
        target_dirname = '1cropped/'
        gray = 1
        dim = (32, 32)
    elif part ==2:
        target_dirname = '2cropped_rgb/'
        gray = 0
        dim = (227, 227, 3)
    setup(target_dirname)
    
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
                im = imread('uncropped/'+filename, gray)[y1:y2, x1:x2]
                resized_im = imresize(im, dim) # dim = (32, 32) or (227, 227, 3) for p1, p2 respectfully
                imsave(target_dirname + new_name, resized_im) # target_dirname = 'cropped' / 'cropped_rgb'
            except IOError:
                print 'IOError: Skipped ' + new_name
                pass
            except ValueError:
                print 'ValueError: Skipped ' + new_name
                pass

if __name__ == "__main__":
    '''
    get_data needs to be called before this
    i.e. uncropped folder must be populated
    '''
    crop_all(1)
    crop_all(2)