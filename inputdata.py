# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:39:09 2021

@author: whong
"""

import pathlib
import tensorflow as tf
import numpy as np
import tqdm


np.set_printoptions(threshold=5)


                                         
def AISdata_train():
    print ("1")
    data_root=r"/content/AISdata"
    train_data_root = pathlib.Path(data_root+"/train")
    label_names = sorted(item.name for item in train_data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    train_all_image_paths = [str(path) for path in list(train_data_root.glob('*/*'))]
    train_all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_all_image_paths]

    imgs1=[]
    for file in train_all_image_paths:
        #print (file)
        #print (file)
        feature=tf.io.read_file(file)
        #feature = imageio.imread(file)
        feature = tf.image.decode_jpeg(feature,channels=1)
        #tf.cast(feature, tf.float32)
        imgs1.append(feature)
 
    imgs1 = np.array(imgs1,dtype=float)

    labels = train_all_image_labels
    
    return imgs1, labels

def AISdata_test():
    data_root=r"/content/AISdata"
    test_data_root = pathlib.Path(data_root+"/test")
    label_names = sorted(item.name for item in test_data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    test_all_image_paths = [str(path) for path in list(test_data_root.glob('*/*'))]
    test_all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_all_image_paths]

    imgs2=[]
    con=0
    for file in test_all_image_paths:
        print (con,file)
        con = con+1
        #print (file)
        feature=tf.io.read_file(file)
        #feature = imageio.imread(file)
        feature = tf.image.decode_jpeg(feature,channels=1)
        #tf.cast(feature, tf.float32)
        imgs2.append(feature)
 
    imgs2 = np.array(imgs2,dtype=float)

    labels = test_all_image_labels
    
    print (labels)
    #print (test_all_image_paths)
    
    
    return imgs2, labels