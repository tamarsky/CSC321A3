from shutil import copyfile
import os.path
from scipy.misc import imread


def setup(part):
    if not os.path.exists(str(part) + 'training_set'):
        os.mkdir(str(part) + 'training_set') 
    if not os.path.exists(str(part) + 'validation_set'):
        os.mkdir(str(part) + 'validation_set')
    if not os.path.exists(str(part) + 'test_set'):
        os.mkdir(str(part) + 'test_set')        
                
def partition_data(part):
    #partition data into Training set, Validation set, Test set
    setup(part)
    
    if part == 1:
        dirname = '1cropped/'
    elif part == 2:
        dirname = '2cropped_rgb/'
    if not os.path.exists(dirname):
        os.mkdir(dirname) 
    
    act = ['Gerard Butler', 'Daniel Radcliffe','Michael Vartan', 'Lorraine Bracco',
    'Peri Gilpin', 'Angie Harmon']
    
    act_lower= ['butler', 'radcli', 'vartan', 'bracco', 'gilpin', 'harmon']
    counts = [0, 0, 0, 0, 0, 0]
    
    act_count = dict(zip(act_lower, counts))
    for filename in os.listdir(dirname):
        if filename[:6] in act_lower:
            act_count[filename[:6]] += 1
            if act_count[filename[:6]] <= 70:
                copyfile(dirname+filename, str(part) + 'training_set/'+filename)
                print filename+' in '+ str(part)+ 'training_set'
            elif act_count[filename[:6]]%2 == 0:
                copyfile(dirname+filename, str(part) + 'validation_set/'+filename)
                print filename+' in '+ str(part) + 'validation_set'                
            else:
                copyfile(dirname+filename, str(part) + 'test_set/'+filename)
                print filename+' in '+ str(part) + 'test_set'
                
                
def get_train_dict(part):
    '''Return dictionary of training set where keys are labels (actors' names as 
    lower case strings e.g. 'bracco') and values are lists of numpy arrays of 
    images of that actor's face
    '''
    if part ==1:
        gray = 0 # grayscale
    elif part == 2:
        gray = 0 # not grayscale
        
    train_dict = {}
    for file in os.listdir(str(part)+'training_set'):
        label = file.split('.')[0].rstrip('1234567890')
        im = imread(str(part)+'training_set/' + file, gray)[:,:,:3]
        if part == 1:
            pass
           # im = im.flatten()
        if not train_dict.has_key(label):
            train_dict[label] = [im]
        else:
            train_dict[label].append(im)
    return train_dict
    
def get_test_dict(part):
    '''Return dictionary of test set where keys are labels (actors' names as 
    lower case strings e.g. 'bracco') and values are lists of numpy arrays of 
    images of that actor's face
    '''
    if part ==1:
        gray = 0 # grayscale
    elif part == 2:
        gray = 0 # not grayscale
        
    test_dict = {}
    for file in os.listdir(str(part)+'test_set'):
        label = file.split('.')[0].rstrip('1234567890')
        im = imread(str(part)+'test_set/' + file, gray)[:,:,:3]
        if part == 1:
            pass
       #     im = im.flatten()
        if not test_dict.has_key(label):
            test_dict[label] = [im]
        else:
            test_dict[label].append(im)
    return test_dict    
    
def get_val_dict(part):
    '''Return dictionary of validation set where keys are labels (actors' names as 
    lower case strings e.g. 'bracco') and values are lists of numpy arrays of 
    images of that actor's face
    '''
    if part ==1:
        gray = 0 # grayscale
    elif part == 2:
        gray = 0 # not grayscale
        
    val_dict = {}
    for file in os.listdir(str(part)+'validation_set'):
        label = file.split('.')[0].rstrip('1234567890')
        im = imread(str(part)+'validation_set/' + file, gray)[:,:,:3]
        if part == 1:
            pass
     #       im = im.flatten()
        if not val_dict.has_key(label):
            val_dict[label] = [im]
        else:
            val_dict[label].append(im)
    return val_dict    

def get_dict_size(dic):
    dict_size = 0
    for key in dic:
        dict_size += len(dic[key])
    return dict_size
    
if __name__ == "__main__":
    '''
    crop_all(#) need to called respectfully 
    i.e. cropped folders need to be made already before calling this
    '''
    partition_data(1)
    partition_data(2)

    
    
    
    
    
    