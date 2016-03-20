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

t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)

import tensorflow as tf

# M = {'butler': [..., [numpy.array], ...], 'radcliffe': [...]}
M = partition_data_set.get_train_dict()
M_test = partition_data_set.get_test_dict()


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
    batch_xs = zeros((0, IMG_DIM*IMG_DIM))
    batch_y_s = zeros( (0, NUM_LABELS))

    
    for actor in M: # for each actor
        train_size = len(M[actor]) # number of images for M[actor]
        idx = array(random.permutation(train_size)[:n]) # array of n random indexes between 0 and train_si
        batch_xs = vstack((batch_xs, ((array(M[actor])[idx])/255.)  )) # add image M['traink'][inx]
        one_hot = actor_to_ohe[actor]
        
        # for x in batch_xs:
        #     imshow(x.reshape((32,32)))
        #     show()
        
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   )) # add n OHE to batch_y_s with label at k
        
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, IMG_DIM*IMG_DIM))
    batch_y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M_test:
        batch_xs = vstack((batch_xs, ((array(M_test[actor])[:])/255.)  ))
        one_hot = actor_to_ohe[actor]
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M_test[actor]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = zeros((0, IMG_DIM*IMG_DIM))
    batch_y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M:
        batch_xs = vstack((batch_xs, ((array(M[actor])[:])/255.)  ))
        one_hot = actor_to_ohe[actor]
        #print(one_hot)
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[actor]), 1))   ))
    #print(batch_xs.shape)
    return batch_xs, batch_y_s
        


x = tf.placeholder(tf.float32, [None, IMG_DIM*IMG_DIM])


nhid = 300
W0 = tf.Variable(tf.random_normal([IMG_DIM*IMG_DIM, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, NUM_LABELS], stddev=0.01))
b1 = tf.Variable(tf.random_normal([NUM_LABELS], stddev=0.01))


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])



lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(NLL)

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
test_x, test_y = get_test(M)


acc_test = []
acc_val
for i in range(5000):
    #print i  
    batch_xs, batch_ys = get_train_batch(M, 60)
    
  
        
  
    #print(sum(batch_ys[:,1]))
    #print(batch_ys)
    # for i in range(10):
    #     print(batch_ys[i])
    #     imshow(batch_xs[i].reshape((IMG_DIM,IMG_DIM)))
    #     show()
    
    result = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    

  
    if i % 100 == 0:
        print ("i=",i)
        print "Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        acc.append(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        batch_xs, batch_ys = get_train(M)
        #print(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        
        
        # print "Penalty:", sess.run(decay_penalty)
    
    
        # snapshot = {}
        # snapshot["W0"] = sess.run(W0)
        # snapshot["W1"] = sess.run(W1)
        # snapshot["b0"] = sess.run(b0)
        # snapshot["b1"] = sess.run(b1)
        # 
        # imshow(snapshot["W0"][:,150].reshape((IMG_DIM,IMG_DIM)))
        # show()
    
        #cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))

plt.figure(1)
plt.plot(acc)
show()