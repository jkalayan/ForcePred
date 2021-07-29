#!/usr/bin/env python

'''
This module is for getting forces that conserve energy during MM.
'''

import numpy as np
#from keras.layers import Input, Dense, concatenate, Layer, initializers, Add  
from keras.models import load_model                                   
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
from ..network.Network import Network
#import sys
#import time

class Conservation(object):
    '''
    '''
    #def __init__(self):
        #self.model = None

    def get_conservation(coords, forces, mat_NRF, scale_NRF, model_name):
        #print(coords, forces, mat_NRF)
        _NC2 = mat_NRF.shape[0]
        print(_NC2)
        model = load_model(model_name) 
        mat_NRF_scaled = mat_NRF / scale_NRF
        print(mat_NRF_scaled.shape)
        prediction_scaled = model.predict(mat_NRF_scaled)
        prediction = (prediction_scaled - 0.5) * scale_F
        print(prediction)

        recomp_forces = Network.get_recomposed_forces([coords], 
                [prediction], n_atoms, _NC2)
