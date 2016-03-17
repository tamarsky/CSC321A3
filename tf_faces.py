import os.path
import get_data
import crop_faces
import partition_data_set
import tensorflow as tf
import part1


if __name__ == '__main__':
    
    get_data.get_all()      # download images of actors
    crop_faces.crop_all()   # crop all images using bounding boxes and colour = False
    partition_data_set.partition_data() # split up images into validation, training, and test sets
    # part1 - create a fully connected network for classifying 6 actors with a single hidden layer