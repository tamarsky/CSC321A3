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


act = ['Bracco', 'Butler', 'Gilpin', 'Harmon', 'Radcliffe', 'Vartan']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def setup():
    '''Create a file of all actors' and actresses' names, cropping boundaries,
    hashes, etc. Create a directory 'uncropped' for the uncropped images.
    '''
    
    filenames = ['subset_actors.txt', 'subset_actresses.txt']
    if not os.path.isfile('all_actors.txt'):
        with open('all_actors.txt', 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
    if not os.path.exists('uncropped'):
        os.mkdir('uncropped')

    
def get_all():
    '''Download all images from the urls in all_actors.txt, and store them in 
    uncropped directory
    '''
    
    setup()
    testfile = urllib.URLopener()            
    for a in act:
        name = a.lower()
        i = 0
        for line in open("all_actors.txt"):
            if a in line:
                coords = line.split()[5]
                hash = line.split()[6]
                filename = name+str(i)+'.'+coords+'.'+hash+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 10)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                
                print filename
                i += 1
        
    