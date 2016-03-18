from shutil import copyfile
import os.path
from scipy.misc import imread


def setup():
    if not os.path.exists('training_set'):
        os.mkdir('training_set') 
    if not os.path.exists('validation_set'):
        os.mkdir('validation_set')
    if not os.path.exists('test_set'):
        os.mkdir('test_set')        
                
def partition_data():
    #partition data into Training set, Validation set, Test set
    setup()
    
    act = ['Gerard Butler', 'Daniel Radcliffe','Michael Vartan', 'Lorraine Bracco',
    'Peri Gilpin', 'Angie Harmon']
    
    act_lower= ['butler', 'radcli', 'vartan', 'bracco', 'gilpin', 'harmon']
    counts = [0, 0, 0, 0, 0, 0]
    
    act_count = dict(zip(act_lower, counts))
    for filename in os.listdir('cropped'):
        if filename[:6] in act_lower:
            act_count[filename[:6]] += 1
            if act_count[filename[:6]] <= 70:
                copyfile('cropped/'+filename, 'training_set/'+filename)
                print filename+' in '+'training_set'
            elif act_count[filename[:6]]%2 == 0:
                copyfile('cropped/'+filename, 'validation_set/'+filename)
                print filename+' in '+'validation_set'                
            else:
                copyfile('cropped/'+filename, 'test_set/'+filename)
                print filename+' in '+'test_set'
                
                
def get_train_dict():
    '''Return dictionary of training set where keys are labels (actors' names as 
    lower case strings e.g. 'bracco') and values are lists of numpy arrays of 
    images of that actor's face
    '''
    train_dict = {}
    for file in os.listdir('training_set'):
        label = file.split('.')[0].rstrip('1234567890')
        im = imread('training_set/' + file, 1).flatten()
        if not train_dict.has_key(label):
            train_dict[label] = [im]
        else:
            train_dict[label].append(im)
    return train_dict
    
def get_test_dict():
    '''Return dictionary of test set where keys are labels (actors' names as 
    lower case strings e.g. 'bracco') and values are lists of numpy arrays of 
    images of that actor's face
    '''
    test_dict = {}
    for file in os.listdir('test_set'):
        label = file.split('.')[0].rstrip('1234567890')
        im = imread('test_set/' + file, 1).flatten()
        if not test_dict.has_key(label):
            test_dict[label] = [im]
        else:
            test_dict[label].append(im)
    return test_dict    
    
def get_dict_size(dic):
    dict_size = 0
    for key in dic:
        dict_size += len(dic[key])
    return dict_size
    
    
    
    
    
    