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

import cPickle

import os
from scipy.io import loadmat

IMG_DIM = 32
NUM_LABELS = 6
DIM3 = 3


t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)

import tensorflow as tf

# M = {'butler': [..., [numpy.array], ...], 'radcliffe': [...]}
M = partition_data_set.get_train_dict(part=1) # need to specify which part
M_test = partition_data_set.get_test_dict(part=1)
M_val = partition_data_set.get_val_dict(part=1)

dim3 = ((array(M['harmon'])[:])/255.).shape[2]/(IMG_DIM*IMG_DIM)
print dim3

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
                


def get_train_batch(M, N):
    n = N/6
    batch_xs = zeros((0, IMG_DIM*IMG_DIM*DIM3))
    batch_y_s = zeros( (0, NUM_LABELS))

    
    for actor in M: # for each actor
        train_size = len(M[actor]) # number of images for M[actor]
        idx = array(random.permutation(train_size)[:n]) # array of n random indexes between 0 and train_si
       
        images = ((array(M[actor])[idx])/255.) 
        images = images.reshape((images.shape[0], IMG_DIM*IMG_DIM*DIM3))
        
        batch_xs = vstack((batch_xs, images  ))
        #batch_xs = vstack((batch_xs, ((array(M[actor])[idx])/255.)  )) # add image M['traink'][inx]
        one_hot = actor_to_ohe[actor]
        
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   )) # add n OHE to batch_y_s with label at k
    #print "Got train batch!", batch_xs.shape, batch_y_s.shape
    return batch_xs, batch_y_s
    

def get_test(M):

    xs = zeros((0, IMG_DIM*IMG_DIM*DIM3))
    y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M_test:
        image = ((array(M_test[actor])[:])/255.) 
        image = image.reshape((image.shape[0], IMG_DIM*IMG_DIM*DIM3))
        
        xs = vstack((xs, image  ))
        one_hot = actor_to_ohe[actor]
        y_s = vstack((y_s,   tile(one_hot, (len(M_test[actor]), 1))   ))
    #print "Got test!", xs.shape, y_s.shape
    return xs, y_s


def get_train(M):
    xs = zeros((0, IMG_DIM*IMG_DIM*DIM3))
    y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M:
        image = ((array(M[actor])[:])/255.) 
        image = image.reshape((image.shape[0], IMG_DIM*IMG_DIM*DIM3))
        
        xs = vstack((xs, image  ))
        one_hot = actor_to_ohe[actor]
        y_s = vstack((y_s,   tile(one_hot, (len(M[actor]), 1))   ))
    #print "Got train!", xs.shape, y_s.shape
    return xs, y_s


def get_validation(M):
    xs = zeros((0, IMG_DIM*IMG_DIM*DIM3))
    y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M_val:
        image = ((array(M_val[actor])[:])/255.) 
        image = image.reshape((image.shape[0], IMG_DIM*IMG_DIM*DIM3))
        
        xs = vstack((xs, image  ))
        one_hot = actor_to_ohe[actor]
        y_s = vstack((y_s,   tile(one_hot, (len(M_val[actor]), 1))   ))
    #print "Got validation!", xs.shape, y_s.shape
    return xs, y_s
        


x = tf.placeholder(tf.float32, [None, IMG_DIM*IMG_DIM*DIM3])/255.


nhid = 300
W0 = tf.Variable(tf.random_normal([IMG_DIM*IMG_DIM*DIM3, nhid], stddev=0.001))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.001))

W1 = tf.Variable(tf.random_normal([nhid, NUM_LABELS], stddev=0.001))
b1 = tf.Variable(tf.random_normal([NUM_LABELS], stddev=0.001))


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])



lam = 0.00085
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
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

test_x, test_y = get_test(M_test)
val_x, val_y = get_validation(M_val)


acc_train = []
acc_val = []
acc_test = []

for i in range(5000):
    #print i  
    batch_xs, batch_ys = get_train_batch(M, 60)
    result = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
        print ("i=",i)
        print "Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        
        acc_val.append(accuracy.eval(feed_dict={x: val_x, y_: val_y}, session=sess))
        acc_test.append(accuracy.eval(feed_dict={x: test_x, y_: test_y}, session=sess))
        
        #print(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        batch_xs, batch_ys = get_train(M)
        print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        acc_train.append(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        
        
        # print "Penalty:", sess.run(decay_penalty)
    
    
    if i == 3000:
        snapshot = {}
        snapshot["W0"] = sess.run(W0)
        snapshot["W1"] = sess.run(W1)
        snapshot["b0"] = sess.run(b0)
        snapshot["b1"] = sess.run(b1)
        
        for i in range(0, 300, 10):
            print(i)
            img = snapshot["W0"][:,i].reshape((IMG_DIM, IMG_DIM, 3))
            r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            imshow (gray/255)
            show()
    
        #cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))

plt.figure(1)
plt.plot(acc_train)
plt.title('Iterations vs. Training Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
savefig('Iteration_vs_Training')


plt.figure(2)
plt.plot(acc_val)
plt.title('Iterations vs. Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
savefig('Iteration_vs_Validation')


plt.figure(3)
plt.plot(acc_test)
plt.title('Iterations vs. Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
savefig('Iteration_vs_Test')