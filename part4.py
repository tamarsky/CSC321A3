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
class_names = ['bracco', 'butler', 'gilpin', 'harmon', 'radcliffe', 'vartan']
net_data = load("bvlc_alexnet.npy").item() # weight

# import part2_alexnet
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
        if len(M[ohe_ind_to_actor[list(ys[i]).index(1)]]) < 50:
            M[ohe_ind_to_actor[list(ys[i]).index(1)]].append(xs[i])
        elif len(M[ohe_ind_to_actor[list(ys[i]).index(1)]]) < 60:
            M_val[ohe_ind_to_actor[list(ys[i]).index(1)]].append(xs[i])
        else:
            M_test[ohe_ind_to_actor[list(ys[i]).index(1)]].append(xs[i])
            
    return M, M_val, M_test
        
# make mat files for M & M_test - run on first run
# M, M_val, M_test  = make_conv4_dicts(xs, ys)
# sc.savemat('conv4dict.mat', M)
# sc.savemat('conv4dictVal.mat', M_val)
# sc.savemat('conv4dictTest.mat', M_test)
# print('done!')

M_loaded = loadmat("conv4dict.mat")
M_test_loaded = loadmat("conv4dictTest.mat")
M_val_loaded = loadmat("conv4dictVal.mat")

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

M = {}
M_test = {}
M_val = {}
for actor in actors:
    M[actor] = M_loaded[actor]
    M_test[actor] = M_test_loaded[actor]
    M_val[actor] = M_val_loaded[actor]

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
    xs = zeros((0, IMG_DIM*IMG_DIM*384))
    y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M:
        images = ((array(M[actor]))/255.)  
        images = images.reshape((70, 13*13*384))
        xs = vstack((xs, images))
        one_hot = actor_to_ohe[actor]

        y_s = vstack((y_s,   tile(one_hot, (len(M[actor]), 1))   ))

    return xs, y_s

def get_validation(M_val):
    xs = zeros((0, IMG_DIM*IMG_DIM*384))
    y_s = zeros( (0, NUM_LABELS))
    
    
    for actor in M_val:
        images = ((array(M_val[actor]))/255.)  
        images = images.reshape((images.shape[0], IMG_DIM*IMG_DIM*384))
        
        xs = vstack((xs, images  ))
        one_hot = actor_to_ohe[actor]
        y_s = vstack((y_s,   tile(one_hot, (len(M_val[actor]), 1))   ))
    return xs, y_s
        

def part2():
    x = tf.placeholder(tf.float32, [None, IMG_DIM*IMG_DIM*384])
    
    W = tf.Variable(tf.random_normal([IMG_DIM*IMG_DIM*384, 6], stddev=0.01))*10.
    b = tf.Variable(tf.random_normal([6], stddev=0.01))
    layer = tf.matmul(x, W)+b
    
    y = tf.nn.softmax(layer)
    y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])
    
    
    
    lam = 0.0085
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
    val_xs, val_ys = get_validation(M_val)
    
    acc_train = []
    acc_val = []
    acc_test = []
    
    # W_summary = tf.image_summary(str(W), W)
    # summary_writer = tf.train.SummaryWriter('/tmp/logs', sess.graph_def)
    
    for i in range(500):
        #print i  
        batch_xs, batch_ys = get_train_batch(M, 60)
        result = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        if i % 10 == 0:
            print ("i=",i)
            print "Test:", sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})
            
            acc_val.append(accuracy.eval(feed_dict={x: val_xs, y_: val_ys}, session=sess))
            acc_test.append(accuracy.eval(feed_dict={x: test_xs, y_: test_ys}, session=sess))
    
            xs, ys = get_train(M)
    
            print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            acc_train.append(accuracy.eval(feed_dict={x: xs, y_: ys}, session=sess))
            print(sess.run(NLL, feed_dict={x: test_xs, y_:test_ys}))
            print(sess.run(NLL, feed_dict={x: xs, y_:ys})) 
            

            #print "Penalty:", sess.run(decay_penalty)
            # 
            # summary_writer.add_summary(result[0], i)
            
        if i == 499:
            snapshot = {}
            snapshot["W"] = sess.run(W)
            snapshot["b"] = sess.run(b)

            
            # for i in range(0, 300, 10):
            #     print(i)
            #     img = snapshot["W0"][:,i].reshape((IMG_DIM, IMG_DIM, 3))
            #     r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            #     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            #     imshow (gray/255)
            #     show()
        
            cPickle.dump(snapshot,  open("ConvOuput"+str(i)+".pkl", "w"))
        
    
    plt.figure(1)
    plt.plot(acc_train)
    plt.title('Iterations vs. Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    savefig('p2Iteration_vs_Training')
    
    
    plt.figure(2)
    plt.plot(acc_val)
    plt.title('Iterations vs. Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    savefig('p2Iteration_vs_Validation')
    
    
    plt.figure(3)
    plt.plot(acc_test)
    plt.title('Iterations vs. Test Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    savefig('p2Iteration_vs_Test')
    
    return W, b
    
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
def part4():
    # image we want to test
    x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
    i = x_dummy.copy()
    i[0,:,:,:] = (imread("2test_set/bracco108.jpg")[:,:,:3]).astype(float32)
    i = i-mean(i)
    x = tf.Variable(i/255.)
    # x = tf.placeholder(tf.float32, [None, IMG_DIM,IMG_DIM,3])
    # y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])
    
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                    depth_radius=radius,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                    depth_radius=radius,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)/255.
    
    # our layer
    snapshot = cPickle.load(open("W499.pkl"))
    
    W = tf.Variable(snapshot["W"])
    b = tf.Variable(snapshot["b"]) 
    
    #W,b = part2()
    conv4 = tf.reshape(conv4, [1, 13*13*384])
    layer = (tf.matmul(conv4 , W)+b)
    
    y = tf.nn.softmax(layer)
    

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    output = sess.run(y)
    
    
    # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # test_xs, test_ys = get_test(M_test)
    # accuracy.eval(feed_dict={x: test_xs, y_: test_ys}, session=sess)
    
    inds = argsort(sess.run(y))[0,:]
    for i in range(6):
        print class_names[inds[-1-i]], output[0, inds[-1-i]]
    
    print i
if __name__ == "__main__":
    part4()