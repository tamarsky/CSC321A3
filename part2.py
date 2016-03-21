###############################################################################
## Part 2 (30%)
# 
# Exctract the values of the activations of AlexNet on the face images. Use 
# those as features in order to perform face classification: learn a fully-
# connected neural network that takes in the activations of the units in the 
# AlexNet layer as inputs, and outputs the name of the person. In your report, 
# include a description of the system you built and its performance, similarly 
# to part 1. It is possible to improve on the results of Part 2 by reducing the 
# error rate by at least 30%. I recommend starting out with only using the conv4 
# activations (for Part 2, only using conv4 is sufficient.)
# 
# You should modify the AlexNet code so that the image is put into a placeholder 
# variable.
##############################################################################

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
import partition_data_set
# import part2_alexnet
import scipy.io as sc

import cPickle

import os
from scipy.io import loadmat

IMG_DIM = 13
NUM_LABELS = 6

t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)

import tensorflow as tf

actors = ['gilpin', 'butler', 'vartan', 'radcliffe', 'harmon', 'bracco']

# xs = part2_alexnet.conv4_outputs # not the same as feed_dict xs in part2_alexnet
# ys = part2_alexnet.ys 

''' One Hot Encoding: alphabetical order
bracco      [1,0,0,0,0,0]
butler      [0,1,0,0,0,0]
gilpin      [0,0,1,0,0,0]
harmon      [0,0,0,1,0,0]
radcliffe   [0,0,0,0,1,0]
vartan      [0,0,0,0,0,1]
'''

id_matrix = identity(NUM_LABELS) # 6 actors
actor_to_ohe = {'bracco':id_matrix[0], 
                'butler':id_matrix[1], 
                'gilpin':id_matrix[2], 
                'harmon':id_matrix[3], 
                'radcliffe':id_matrix[4], 
                'vartan':id_matrix[5]}

                
def make_conv4_dicts(xs,ys):
    ''' Return a dictionary where the keys are actors' names and the values are
    the conv4 outputs from AlexNet of those actors
    '''
    M = {'bracco':[],'butler':[],'gilpin':[],'harmon':[],'radcliffe':[],'vartan':[]}
    M_test = M.copy()
    M_val = M.copy()
    ids = array(random.permutation(len(xs)))
    ohe_ind_to_actor = {0:'bracco', 1:'butler', 2:'gilpin', 3:'harmon', 4:'radcliffe',5:'vartan'}
    for i in ids:
        if len(M[ohe_ind_to_actor[list(ys[i]).index(1)]]) < 55:
            M[ohe_ind_to_actor[list(ys[i]).index(1)]].append(xs[i])
        else:
            M_test[ohe_ind_to_actor[list(ys[i]).index(1)]].append(xs[i])
            
    return M, M_test
        
# make mat files for M & M_test - run on first run
# M, M_test  = make_conv4_dicts(xs, ys)
# sc.savemat('conv4dict.mat', M)
# sc.savemat('conv4dictTest.mat', M_test)
# print('done!')

M_loaded = loadmat("conv4dict.mat")
M_test_loaded = loadmat("conv4dictTest.mat")

M = {}
M_test = {}
for actor in actors:
    M[actor] = M_loaded[actor]
    M_test[actor] = M_test_loaded[actor]

def get_train_batch(M, N):
    n = N/6
    batch_xs = zeros((0, IMG_DIM*IMG_DIM*384))
    batch_y_s = zeros( (0, NUM_LABELS))

    
    for actor in M: # for each actor
        train_size = len(M[actor]) # number of images for M[actor]
        idx = array(random.permutation(train_size)[:n]) # array of n random indexes between 0 and train_si
        images = ((array(M[actor])[idx])/255.)
        images = images.reshape((10, 13*13*384))
        batch_xs = vstack((batch_xs, images))
        one_hot = actor_to_ohe[actor]
        
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   )) # add n OHE to batch_y_s with label at k
        
    return batch_xs, batch_y_s
    

def get_test(M_test):
    batch_xs = zeros((0, IMG_DIM*IMG_DIM*384))
    batch_y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M_test:
        images = ((array(M_test[actor]))/255.)  
        images = images.reshape((70, 13*13*384))
        batch_xs = vstack((batch_xs, images))
        one_hot = actor_to_ohe[actor]
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M_test[actor]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = zeros((0, IMG_DIM*IMG_DIM*384))
    batch_y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M:
        images = ((array(M[actor]))/255.)  
        images = images.reshape((70, 13*13*384))
        batch_xs = vstack((batch_xs, images))
        one_hot = actor_to_ohe[actor]

        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[actor]), 1))   ))

    return batch_xs, batch_y_s

def get_validation(M_val):
    xs = zeros((0, IMG_DIM*IMG_DIM*DIM3))
    y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M_val:
        image = ((array(M_val[actor])[:])/255.) 
        image = image.reshape((image.shape[0], IMG_DIM*IMG_DIM*DIM3))
        
        xs = vstack((xs, image  ))
        one_hot = actor_to_ohe[actor]
        y_s = vstack((y_s,   tile(one_hot, (len(M_val[actor]), 1))   ))
    return xs, y_s
        


x = tf.placeholder(tf.float32, [None, IMG_DIM*IMG_DIM*384])

W = tf.Variable(tf.random_normal([IMG_DIM*IMG_DIM*384, 6], stddev=0.01))
b = tf.Variable(tf.random_normal([6], stddev=0.01))
layer = tf.matmul(x, W)+b

y = tf.nn.softmax(layer)
y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])



lam = 0.00085
decay_penalty =lam*tf.reduce_sum(tf.square(W))
NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

alpha = 0.005
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# figure out where predicted correct label
# tf.argmax - gives index of highest entry in tensor along axis
#       tf.argmax(y,1) - gives estimate 
#       tf.argmax(y_,1) - true label
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# cast tf.equal output to 1s and 0s
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_xs, test_ys = get_test(M_test)
# val_x, test_y = get_val(M_val)

acc_train = []
# acc_val = []
acc_test = []

for i in range(500):
    #print i  
    batch_xs, batch_ys = get_train_batch(M, 60)
    batch_xs /= 255.
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if i % 10 == 0:
        print ("i=",i)
        print "Test:", sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})
        
        # acc_val.append(accuracy.eval(feed_dict={x: val_x, y_: val_y}, session=sess))
        acc_test.append(accuracy.eval(feed_dict={x: test_xs, y_: test_ys}, session=sess))
        
        #print(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        xs, ys = get_train(M)
        xs /= 255.
        #print(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        acc_train.append(accuracy.eval(feed_dict={x: xs, y_: ys}, session=sess))
        print(sess.run(NLL, feed_dict={x: test_xs, y_:test_ys}))
        print(sess.run(NLL, feed_dict={x: xs, y_:ys})) 
        
        
        #print "Penalty:", sess.run(decay_penalty)
        
    
    

plt.figure(1)
plt.plot(acc_train)
plt.title('Iterations vs. Training Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
savefig('p2Iteration_vs_Training')

# 
# plt.figure(2)
# plt.plot(acc_val)
# plt.title('Iterations vs. Validation Accuracy')
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# savefig('p2Iteration_vs_Validation')


plt.figure(3)
plt.plot(acc_test)
plt.title('Iterations vs. Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
savefig('p2Iteration_vs_Test')