###############################################################################
## Part 1 - 30%
#
# Using TensorFlow, make a system for classifying faces from the 6 actors in 
# Project 1. Use a fully-connected neural network with a single hidden layer. 
# In your report, include the learning curve for the test, training, and 
# validation sets, and the final performance classification on the test set. 
# Include a text description of your system. In particular, describe how you 
# preprocessed the input and initialized the weights, what activation function 
# you used, and what the exact architecture of the network that you selected was. 
# I got about 80-85% using a single-hidden-layer network. You might get slightly 
# worse results.
###############################################################################


from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf

# x is a placeholder that can be initialized to have any length, i.e. any amount of images
x = tf.placeholder(tf.float32, [None, 1024]) 

# inititalize W - weight matrix and b - bias vector as tensors full of zeroes
# dim(W) = 1024x6
# dim(b) = 6x1
W = tf.Variable(tf.zeros([1024, 6]))
b = tf.Variable(tf.zeros([6]))

def make_data_sets():
    ''' Using directories validation_set, training_set, and test_set, create 
    three data subsets corresponding to the three directories
    '''
    pass
    
def make_net():
    '''Make a fully-connected neural network with a single hidden layer
    '''
    pass
    
def cost_fn():
    pass
    
def classify_face(im):
    '''Return classification label for image im'''
    pass
    
def get_weights():
    # inititalize W - weight matrix and b - bias vector
    # dim(W) = 1024x6
    W = tf.Variable(tf.zeros([1024, 6]))
    b = tf.Variable(tf.zeros([6]))
    pass
    

