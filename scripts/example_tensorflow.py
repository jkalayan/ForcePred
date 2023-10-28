#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras

import numpy as np
#from keras.layers import Input, Dense, concatenate, Layer, initializers, Add  
#from keras.models import load_model                                   
#from keras.models import Model, load_model                                   
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#from keras import backend as K                                              
#import tensorflow as tf
#from ..calculate.MM import MM
#from ..calculate.Converter import Converter
#from ..calculate.Binner import Binner
#from ..calculate.Plotter import Plotter
#from ..write.Writer import Writer
#from ..read.Molecule import Molecule
#from ..nn.Network import Network
#import sys
#import time

tf.enable_eager_execution()

print('start')


a = np.array([[1.0,2.0,3.0]])/1
b = np.array([[4.0,5.0,6.0]])/1

#a = tf.random.normal(shape=(1, 3))
#b = tf.random.normal(shape=(1, 3))

#a = tf.constant([[1,2,3]])
#b = tf.constant([[4,5,6]])


'''
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
a = tf.convert_to_tensor(a)
b = tf.convert_to_tensor(b)
'''

'''
c = a + b
d = tf.square(c)
e = tf.exp(d)


'''

#'''
a = tf.Variable(a)
b = tf.Variable(b)

print(a)
print(b)
with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a) + tf.square(b))
    #c = tf.add(a,b)
    #c = tf.square(a)
    print(c)
    dc_da = tape.gradient(c, a) #differential of c wrt a
    print(dc_da)
#'''

'''
print()
print(a)
print(b)
print(c)
print(d)
print(e)
'''
