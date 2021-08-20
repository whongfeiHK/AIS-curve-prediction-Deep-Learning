import numpy as np
import tensorflow as tf
from utils.layers import PrimaryCaps, FCCaps, Length, Mask
from keras import regularizers


def efficient_capsnet_graph(input_shape):
    """
    Efficient-CapsNet graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(input_shape)
    
    #x = tf.keras.layers.Conv2D(32,5,2,activation="relu", padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01),kernel_initializer='he_normal')(inputs)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Conv2D(64,4,2, activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Conv2D(64,3,2, activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal')(x)   
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Conv2D(128,2,2,activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),kernel_initializer='he_normal')(x)   
    #x = tf.keras.layers.BatchNormalization()(x)
    
    #x = PrimaryCaps(128, (18,11), 16, 8)(x)
    
    
    x = tf.keras.layers.Conv2D(32,(4,3),2,activation="relu", padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01),kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    print (x.shape)
    x = tf.keras.layers.Conv2D(32,3,2, activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3,2, activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,(3,2),activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    print (x.shape)
    x = tf.keras.layers.Conv2D(128,(2,1),activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),kernel_initializer='he_normal')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = PrimaryCaps(128,(14,10), 16, 8)(x)
    
    digit_caps = FCCaps(2,16)(x)
    
    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs,outputs=[digit_caps, digit_caps_len], name='Efficient_CapsNet')


def generator_graph(input_shape):
    """
    Generator graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16*2)
    
    #x = tf.keras.layers.Dense(198)(inputs)
    #x = tf.keras.layers.Reshape(target_shape=(18,11,1))(x)
    #x = tf.keras.layers.UpSampling2D(size=(3,2), interpolation='bilinear')(x)   #54,22
    #x = tf.keras.layers.Conv2D(16, (2,4), (2,1), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #27,19
    #x = tf.keras.layers.UpSampling2D(size=(3,3), interpolation='bilinear')(x)   #81,57
    
    #x = tf.keras.layers.Conv2D(16, (3,4), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #79,55
    
    #x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)   #158,110
    
    #x = tf.keras.layers.Conv2D(32, (3,4), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #156 105
    
    #x = tf.keras.layers.Conv2D(32, (4,3), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #153 103
    
    #x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)   #306, 206
    
    #x = tf.keras.layers.Conv2D(16, (4,4), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) 
    
    #x = tf.keras.layers.Conv2D(1, (4,4), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.sigmoid)(x) #300,200
    
    x = tf.keras.layers.Dense(140)(inputs)
    x = tf.keras.layers.Reshape(target_shape=(14,10,1))(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)   #28,20
    x = tf.keras.layers.Conv2D(16, 3, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #26,18
    x = tf.keras.layers.UpSampling2D(size=(3,3), interpolation='bilinear')(x)   #78,54
    
    x = tf.keras.layers.Conv2D(16, 3, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #76,52
    
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)   #152,104
    
    x = tf.keras.layers.Conv2D(32, (3,5), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #150 100
    
    #x = tf.keras.layers.Conv2D(32, (4,3), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) #153 103
    
    #x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)   #306, 206
    
    #x = tf.keras.layers.Conv2D(16, (4,4), kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.leaky_relu)(x) 
    
    x = tf.keras.layers.Conv2D(1, 1, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),padding="valid", activation=tf.nn.sigmoid)(x) #150,100
    
    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialize with different modalities.

    Parameters
    ----------   
    input_shape: list
        network input shape
    mode: str
        working mode ('train', 'test' & 'play')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(2,))
    noise = tf.keras.layers.Input(shape=(2, 16))

    efficient_capsnet = efficient_capsnet_graph(input_shape)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    
    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise]) # only if mode is play
    
    masked_by_y = Mask()([digit_caps, y_true])  
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])
    
    generator = generator_graph(input_shape)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    if mode == 'train':   
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='Efficinet_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='Efficinet_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
