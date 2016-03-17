from shutil import copyfile
import os.path


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
