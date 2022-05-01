#!/usr/bin/env python

'''
This module is for running a NN with a training set of data.
'''
from __future__ import print_function #for tf printing
import numpy as np
#import keras #older version
from tensorflow import keras #newer version
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, \
        Layer #initializers #, Add, Multiply
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
        ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf

#print("TensorFlow version:", tf.__version__)
#print("List of GPUs:", tf.config.list_physical_devices('GPU'))

#NUMCORES=int(os.getenv("NSLOTS",1))
#print("Using", NUMCORES, "core(s)" )
#tf.config.threading.set_inter_op_parallelism_threads(NUMCORES) 
#tf.config.threading.set_intra_op_parallelism_threads(NUMCORES)
#tf.config.set_soft_device_placement(1)


#import tensorflow_addons as tfa
#import tensorflow_probability as tfp
from functools import partial
from ..calculate.MM import MM
from ..calculate.Converter import Converter
from ..calculate.Binner import Binner
from ..calculate.Plotter import Plotter
from ..write.Writer import Writer
from ..read.Molecule import Molecule
from ..calculate.Conservation import Conservation
import sys
import time

start_time = time.time()

tf.compat.v1.disable_eager_execution()

class CoordsToNRF_test(Layer):
    def __init__(self, atoms_flat, max_NRF, _NC2, **kwargs):
        super(CoordsToNRF_test, self).__init__()
        self.atoms_flat = atoms_flat
        #self.atoms2 = tf.Variable(atoms2)
        self.max_NRF = max_NRF
        self._NC2 = _NC2
        #self.name = name
        self.au2kcalmola = 627.5095 * 0.529177

    #def build(self, input_shape):
        #self.kernal = self.add_weight('kernel')

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        n_atoms = input_shape[1]
        #return (batch_size, n_atoms, n_atoms)
        return (batch_size, self._NC2)

    def call(self, coords):
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r = diff_flat**0.5
        _NRF = (((self.atoms_flat * self.au2kcalmola) / (r ** 2)) / 
                self.max_NRF) #scaled
        return _NRF


class CoordsTo_atomNRF(Layer):
    def __init__(self, atoms_flat, atoms2, max_atomNRF, _NC2, **kwargs):
        super(CoordsTo_atomNRF, self).__init__()
        self.atoms_flat = tf.Variable(atoms_flat)
        self.atoms2 = tf.Variable(atoms2)
        self.max_atomNRF = tf.Variable(max_atomNRF)
        self._NC2 = tf.Variable(_NC2)
        #self.name = name
        self.au2kcalmola = tf.Variable(627.5095 * 0.529177)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        n_atoms = input_shape[1]
        #return (batch_size, n_atoms, n_atoms)
        return (batch_size, n_atoms)

    def call(self, coords):
        '''
        get traingle NRF and then sum each row to get atom-wise NRFs
        '''
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r = diff_flat**0.5
        _NRF = ((self.atoms_flat * self.au2kcalmola) / (r ** 2))
        triangle_NRF = Triangle(self.n_atoms, self.name)(_NRF)
        atomNRF = tf.reduce_sum(triangle_NRF, 1, keepdims=True) / 2
        atomNRF_scaled = atomNRF / self.max_atomNRF
        return atomNRF_scaled

class CoordsTo_angleNRF(Layer):
    def __init__(self, n_atoms, atoms_flat, atoms2, max_NRF, _NC2, **kwargs):
        super(CoordsTo_angleNRF, self).__init__()
        self.n_atoms = tf.Variable(n_atoms)
        self.atoms_flat = tf.Variable(atoms_flat)
        self.atoms2 = tf.Variable(atoms2)
        self.max_NRF = tf.Variable(max_NRF)
        self._NC2 = tf.Variable(_NC2)
        #self.name = name
        self.au2kcalmola = tf.Variable(627.5095 * 0.529177)
        #self.test = None #tf.Variable(initial_value=tf.zeros((self._NC2)), 
                #trainable=False)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        n_atoms = input_shape[1]
        #return (batch_size, n_atoms, n_atoms)
        return (batch_size, self._NC2)
        #return (batch_size, 3)
        #return (batch_size, 1)

    def call(self, coords):
        #test = tf.reshape(coords, shape=(tf.shape(coords)[0], self.n_atoms*3)) * 2
        test = coords * 2

        '''
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r = diff_flat**0.5

        _NRF = (((self.atoms_flat * self.au2kcalmola) / (r ** 2)) / 
                self.max_NRF) #scaled
        '''


        '''
        ##get angles for each pair with centre of molecule
        centre = tf.reduce_sum(coords, axis=1) / \
                tf.cast(tf.shape(coords)[1], tf.float32)
        vectors = coords - centre
        #unit_mags = tf.einsum('bij, bij -> b', vectors, vectors) ** 0.5
        unit_mags = tf.reduce_sum(vectors**2, axis=2) ** 0.5
        numerator = tf.einsum('bij, bjk -> bik', vectors, 
                tf.transpose(vectors))
        denominator = unit_mags * tf.transpose(unit_mags)
        cos_theta = numerator / denominator

        tri2 = tf.linalg.band_part(cos_theta, -1, 0) #lower
        nonzero_indices2 = tf.where(tf.not_equal(tri2, tf.zeros_like(tri2)))
        nonzero_values2 = tf.gather_nd(tri2, nonzero_indices2)
        cos_theta_flat = tf.reshape(nonzero_values2, 
                shape=(tf.shape(tri2)[0], -1)) #reshape to _NC2
        '''

        '''
        centre = tf.reduce_sum(coords, axis=1) / self.n_atoms 
        vectors = coords - centre
        unit_mags1 = tf.reduce_sum(vectors**2, axis=-1) ** 0.5 #last axes
        unit_mags = tf.expand_dims(unit_mags1, -1)
        numerator = tf.einsum('bij, bjk -> bik', vectors, 
                tf.transpose(vectors))
        denominator = tf.einsum('bij, bjk -> bik', unit_mags, 
                tf.transpose(unit_mags))
        #denominator = tf.matmul(unit_mags, tf.transpose(unit_mags)) #works too
        cos_theta = numerator / denominator

        tri1 = tf.linalg.band_part(cos_theta, -1, 0) #lower
        diag = tf.linalg.band_part(cos_theta, 0, 0) #diag of ones
        tri2 = tri1 - diag #get lower without diag of ones
        nonzero_indices2 = tf.where(tf.not_equal(tri2, tf.zeros_like(tri2)))
        nonzero_values2 = tf.gather_nd(tri2, nonzero_indices2)
        cos_theta_flat = tf.reshape(nonzero_values2, 
                shape=(-1, self._NC2)) #reshape to _NC2
        
        angleNRF = ((self.atoms_flat * self.au2kcalmola) / (r ** 2)) * \
                (cos_theta_flat + 1)
        scaled_angleNRF = angleNRF / (2 * self.max_NRF) + 0.5 #scaled 
        '''
        #return scaled_angleNRF
        return test #_NRF


class ScaleFE(Layer):
    def __init__(self, _NC2, max_FE, **kwargs):
        super(ScaleFE, self).__init__()
        self._NC2 = _NC2
        self.max_FE = max_FE
        #self.name = name

    #def build(self, input_shape):
        #self.kernal = self.add_weight('kernel')

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, decompFE):
        #rescale decompFE
        decompFE_scaled = (decompFE - 0.5) * (2 * self.max_FE)
        return decompFE_scaled


class Scale_atomE(Layer):
    def __init__(self, n_atoms, max_atomE, **kwargs):
        super(Scale_atomE, self).__init__()
        self.n_atoms = tf.Variable(n_atoms)
        self.max_atomE = tf.Variable(max_atomE)
        #self.name = name

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms)

    def call(self, atomE):
        #rescale atomE
        atomE_scaled = (atomE - 0.5) * (2 * self.max_atomE)
        return atomE_scaled


class atomE_test(Layer):
    def __init__(self, n_atoms, _NC2, max_FE, prescale, **kwargs):
        super(atomE_test, self).__init__()
        self.max_FE = tf.Variable(max_FE)
        self.n_atoms = tf.Variable(n_atoms)
        self._NC2 =tf.Variable(_NC2)
        self.prescale = tf.Variable(prescale)
        #self.name = name

    def compute_output_shape(self, input_shape):
        #batch_size = input_shape[0][0]
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms)
        #return (batch_size, self.n_atoms*3+1)
        #return (batch_size, self._NC2)
        #return (batch_size, self.n_atoms, self.n_atoms, 1)
        #return (batch_size, self.n_atoms, 3)

    def call(self, decompFE):
        '''decompFE is a sq upper and lower triangle matrix'''
        #decompFE = decompFE_coords[:,:,3:]
        E_atoms = tf.reduce_sum(decompFE, 1) / 2
        '''
        E_atoms_postscale = ((E_atoms - self.prescale[2]) / 
                (self.prescale[3] - self.prescale[2]) * 
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])
        '''
        return E_atoms #_postscale


class sum_atomE_test(Layer):
    def __init__(self, n_atoms, _NC2, max_FE, prescale, **kwargs):
        super(sum_atomE_test, self).__init__()
        self.max_FE = tf.Variable(max_FE)
        self.n_atoms = tf.Variable(n_atoms)
        self._NC2 = tf.Variable(_NC2)
        self.prescale = tf.Variable(prescale)
        #self.name = name

    def compute_output_shape(self, input_shape):
        #batch_size = input_shape[0][0]
        batch_size = input_shape[0]
        return (batch_size, 1)
        #return (batch_size, self.n_atoms*3+1)
        #return (batch_size, self._NC2)
        #return (batch_size, self.n_atoms, self.n_atoms, 1)
        #return (batch_size, self.n_atoms, 3)

    def call(self, E_atoms):
        '''decompFE is a sq upper and lower triangle matrix'''
        #decompFE = decompFE_coords[:,:,3:]
        E_atoms_sum = tf.reduce_sum(E_atoms, 1, keepdims=True)
        E_atoms_sum_postscale = ((E_atoms_sum - self.prescale[2]) / 
                (self.prescale[3] - self.prescale[2]) * 
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])

        return E_atoms_sum_postscale


class EijMatrix_test(Layer):
    def __init__(self, n_atoms, _NC2, prescale, **kwargs):
        super(EijMatrix_test, self).__init__()
        #self.max_FE = tf.Variable(max_FE, name='3a')
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        self.prescale = prescale
        #self.name = name

    #def build(self, input_shape):
        #self.kernal = self.add_weight('kernel')

    def compute_output_shape(self, input_shape):
        #batch_size = input_shape[0][0]
        batch_size = input_shape[0]
        return (batch_size, 1)
        #return (batch_size, self.n_atoms*3+1)
        #return (batch_size, self._NC2)
        #return (batch_size, self.n_atoms, self.n_atoms, 1)
        #return (batch_size, self.n_atoms, 3)

    def call(self, coords_decompFE):
    #def call(self, coords):
        '''decompFE is flat, so get a sq upper and lower triangle matrix, 
        from coords we get the eij matrices for Fs and Es.'''
        #coords = coords_decompFE[:,:,:3]
        #decompFE = coords_decompFE[:,:,3:]
        coords, decompFE_flat = coords_decompFE
        decompFE = Triangle(self.n_atoms)(decompFE_flat)


        #'''
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r_flat = diff_flat**0.5


        #get energy 1/r_ij eij matrix
        recip_r_flat = 1 / r_flat
        #ones = tf.ones_like(recip_r_flat)
        Q3 = Triangle(self.n_atoms)(recip_r_flat)
        #Q3 = Triangle(self.n_atoms)(ones) #!!!!!
        eij_E = tf.expand_dims(Q3, 3)
        #dot product of 
        E2 = tf.einsum('bijk, bij -> bk', eij_E, decompFE)
        E = E2/2
        #'''

        '''
        E_atoms = tf.reduce_sum(decompFE, 1) / 2
        E = tf.reduce_sum(E_atoms, 1, keepdims=True)
        '''

        E3 = ((E - self.prescale[2]) / 
                (self.prescale[3] - self.prescale[2]) * 
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])

        #gradients = #tf.gradients(E3, coords, 
                #colocate_gradients_with_ops=True, 
                #unconnected_gradients='zero'
                #) * -1

        #gradients = get_grads(coords, E3)

        '''
        #### FOR FORCES IF REQUIRED
        r = Triangle(self.n_atoms)(r_flat)
        r2 = tf.expand_dims(r, 3)
        eij_F2 = diff / r2
        eij_F = tf.where(tf.math.is_nan(eij_F2), tf.zeros_like(eij_F2), 
                eij_F2) #remove nans
        F = tf.einsum('bijk, bij -> bik', eij_F, decompFE)
        F_reshaped = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        FE = tf.concat([F_reshaped, E], axis=1)
        '''

        return E3


class ForForces(Layer):
    def __init__(self, n_atoms, _NC2, prescale, **kwargs):
        super(ForForces, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        self.prescale = prescale

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, coords_decompFE):
        coords, decompFE_flat = coords_decompFE
        decompFE = Triangle(self.n_atoms)(decompFE_flat)

        #'''
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r_flat = diff_flat**0.5


        #get energy 1/r_ij eij matrix
        recip_r_flat = 1 / r_flat
        #ones = tf.ones_like(recip_r_flat)
        Q3 = Triangle(self.n_atoms)(recip_r_flat)
        #Q3 = Triangle(self.n_atoms)(ones) #!!!!!
        eij_E = tf.expand_dims(Q3, 3)
        #dot product of 
        E2 = tf.einsum('bijk, bij -> bk', eij_E, decompFE)
        E = E2/2
        #'''



        #E_atoms = tf.reduce_sum(decompFE, 1) / 2
        #E = tf.reduce_sum(E_atoms, 1, keepdims=True)
        E3 = ((E - self.prescale[2]) / 
                (self.prescale[3] - self.prescale[2]) * 
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])
        return E3


class EijMatrix_test2(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(EijMatrix_test2, self).__init__()
        #self.max_FE = tf.Variable(max_FE, name='4a')
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        #self.prescale = tf.Variable(prescale, name='4b')
        #self.name = name

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        #return (batch_size, 1)
        #return (batch_size, self.n_atoms*3+1)
        #return (batch_size, self._NC2)
        #return (batch_size, self.n_atoms, self.n_atoms, 1)
        return (batch_size, self.n_atoms, 3)

    def call(self, coords_decompFE):
    #def call(self, coords):
        '''decompFE is a sq upper and lower triangle matrix, from coords we 
        get the eij matrices for Fs and Es.'''
        #coords = coords_decompFE[:,:,:3]
        #decompFE = coords_decompFE[:,:,3:]
        coords, decompFE_flat = coords_decompFE
        decompFE = Triangle(self.n_atoms)(decompFE_flat)

        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r_flat = diff_flat**0.5
        
        '''
        #get energy 1/r_ij eij matrix
        recip_r_flat = 1 / r_flat
        Q3 = Triangle(self.n_atoms)(recip_r_flat)
        eij_E = tf.expand_dims(Q3, 3)
        #dot product of 
        E2 = tf.einsum('bijk, bij -> bk', eij_E, decompFE)
        E = E2/2
        E3 = (E / self.prescale[1]) - self.prescale[0] #orig scale
        #E3 = ((E / self.prescale[3]) * self.prescale[2]) #\
                #* self.prescale[3] + self.prescale[2]
        #E3 = (E * self.prescale[1]) + self.prescale[0]
        '''

        #'''
        #### FOR FORCES IF REQUIRED
        r = Triangle(self.n_atoms)(r_flat)
        r2 = tf.expand_dims(r, 3)
        eij_F2 = diff / r2
        eij_F = tf.where(tf.math.is_nan(eij_F2), tf.zeros_like(eij_F2), 
                eij_F2) #remove nans
        F = tf.einsum('bijk, bij -> bik', eij_F, decompFE)
        #F_reshaped = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        #FE = tf.concat([F_reshaped, E], axis=1)
        #'''

        return F



@tf.function
def get_grads(x, y):
    '''this needs to be wrapped in a tf.function to work'''
    return tf.gradients(y, x, 
            unconnected_gradients='zero', #colocate_gradients_with_ops=True
            )[0] * -1


class EnergyGradient(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(EnergyGradient, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        #self.g = g
        #self.name = name

    #def build(self, input_shape):
        #self.kernal = self.add_weight('kernel')

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms, 3)
        #return (batch_size, self._NC2)

    def call(self, E_coords):
        E, coords = E_coords
        #gradients = get_grads(coords, E)
        gradients = tf.compat.v1.gradients(E, coords, 
                colocate_gradients_with_ops=True, 
                unconnected_gradients='zero')
        return gradients[0] * -1


        #with tf.GradientTape() as g:
            #g.watch(coords)
            #y = model(x)[0]
            #g.watch(E)
        #g = tf.GradientTape(persistent=True)
        #gradients = g.jacobian(E, coords, unconnected_gradients='zero'
                #)#[0][0] * -1
        #gradients = g.gradient(E, coords, unconnected_gradients='zero'
                #) #* -1
        #return gradients




class FlatTriangle(Layer):
    def __init__(self, _NC2, **kwargs):
        super(FlatTriangle, self).__init__()
        self._NC2 = _NC2
        #self.name = name

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, tri):
        '''Take the lower triangle of the sq matrix, remove zeros and reshape
        https://stackoverflow.com/questions/42032517/
                how-to-omit-zeros-in-a-4-d-tensor-in-tensorflow
        '''
        tri = tf.linalg.band_part(tri, -1, 0)
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        reshaped_nonzero_values = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        return reshaped_nonzero_values


class Triangle(Layer):
    def __init__(self, n_atoms, **kwargs):
        super(Triangle, self).__init__()
        self.n_atoms = n_atoms
        #self.name = name

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms, self.n_atoms)

    def call(self, decompFE):
        '''Convert flat NC2 to lower and upper triangle sq matrix, this
        is used in get_FE_eij_matrix to get recomposedFE
        https://stackoverflow.com/questions/40406733/
                tensorflow-equivalent-for-this-matlab-code
        '''
        #decompFE = tf.convert_to_tensor(decompFE, dtype=tf.float32)
        #rescale decompFE
        #decompFE = ((decompFE - 0.5) * (2 * self.max_FE))

        #decompFE.get_shape().with_rank_at_least(1)

        #put batch dimensions last
        decompFE = tf.transpose(decompFE, tf.concat([[tf.rank(decompFE)-1],
                tf.range(tf.rank(decompFE)-1)], axis=0))
        input_shape = tf.shape(decompFE)[0]
        #compute size of matrix that would have this upper triangle
        matrix_size = (1 + tf.cast(tf.sqrt(tf.cast(input_shape*8+1, 
                tf.float32)), tf.int32)) // 2
        matrix_size = tf.identity(matrix_size)
        #compute indices for whole matrix and upper diagonal
        index_matrix = tf.reshape(tf.range(matrix_size**2), 
                [matrix_size, matrix_size])


        tri1 = tf.linalg.band_part(index_matrix, -1, 0) #lower
        diag = tf.linalg.band_part(index_matrix, 0, 0) #diag of ones
        tri2 = tri1 - diag #get lower without diag of ones
        nonzero_indices = tf.where(tf.not_equal(tri2, tf.zeros_like(tri2)))
        nonzero_values = tf.gather_nd(tri2, nonzero_indices)
        reshaped_nonzero_values = tf.reshape(nonzero_values, [-1])

        '''
        diagonal_indices = (matrix_size * tf.range(matrix_size)
                + tf.range(matrix_size))
        upper_triangular_indices, _ = tf.unique(tf.reshape(
                #tf.matrix_band_part(index_matrix, -1, 0) #v1
                tf.linalg.band_part(index_matrix, -1, 0)
                - tf.linalg.diag(diagonal_indices), [-1]))
        '''
        batch_dimensions = tf.shape(decompFE)[1:]
        return_shape_transposed = tf.concat([[matrix_size, matrix_size],
                batch_dimensions], axis=0)
        #fill everything else with zeros; later entries get priority
        #in dynamic_stitch
        result_transposed = tf.reshape(tf.dynamic_stitch([index_matrix,
                #upper_triangular_indices[1:]
                reshaped_nonzero_values
                ],
                [tf.zeros(return_shape_transposed, dtype=decompFE.dtype),
                decompFE]), return_shape_transposed)
        #Transpose the batch dimensions to be first again
        Q = tf.transpose(result_transposed, tf.concat(
                [tf.range(2, tf.rank(decompFE)+1), [0,1]], axis=0))
        Q2 = tf.transpose(result_transposed, tf.concat(
                [tf.range(2, tf.rank(decompFE)+1), [1,0]], axis=0))
        Q3 = Q + Q2

        return Q3


class Network(object):
    '''
    '''
    def __init__(self, molecule):
        self.model = None
        #self.model_name = None
        self.atoms = molecule.atoms
        self.n_atoms = len(self.atoms) 
        self._NC2 = int(self.n_atoms * (self.n_atoms-1)/2) 
        scale_NRF = None 
        scale_NRF_min = None 
        scale_F = None 
        scale_F_min = None


    def get_network(molecule, scale_NRF, scale_NRF_min, 
            scale_F, scale_F_min):
        network = Network(molecule)
        #network.model_name = '../best_ever_model'
        #network.model = load_model(network.model_name)
        network.scale_NRF = scale_NRF 
        network.scale_NRF_min = scale_NRF_min
        network.scale_F = scale_F
        network.scale_F_min = scale_F_min
        return network


    def get_coord_FE_model(self, molecule, prescale1):
        '''Input coordinates and z_types into model to get NRFS which then 
        are used to predict decompFE, which are then recomposed to give
        Cart Fs and molecular E, both of which could be used in the loss
        function, could weight the E or Fs as required.
        '''

        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms*(n_atoms-1)/2)
        atoms = np.array([float(i) for i in molecule.atoms], dtype='float32')
        atoms_ = tf.convert_to_tensor(atoms, dtype=tf.float32)
        #multiply each z with each other to get a sq matrix
        atoms2 = tf.tensordot(tf.expand_dims(atoms_, 0), 
                tf.expand_dims(atoms_, 0), axes=[[0],[0]])

        atoms_flat = []
        for i in range(n_atoms):
            for j in range(i):
                ij = atoms[i] * atoms[j]
                atoms_flat.append(ij)
        atoms_flat = tf.convert_to_tensor(atoms_flat) #_NC2

        training_model = True
        if training_model:
            #for training a model
            get_data = True
            load_first = True
            fit = True
            load_weights = False
            inp_out_pred = True

        if training_model == False:
            #for loading a model to use with openmm
            get_data = False
            load_first = False
            fit = False
            load_weights = True
            inp_out_pred = False

        #get_data = True
        if get_data:
            split = 2 #100 #500 #200 #2
            train = round(len(molecule.coords) / split, 3)
            print('\nget train and test sets, '\
                    'training set is {} points'.format(train))
            Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                    split) #get train and test sets
            '''
            print('!!!use regularly spaced training')
            molecule.train = np.arange(2, len(molecule.coords), split).tolist() 
            molecule.test = [x for x in range(0, len(molecule.coords)) 
                    if x not in molecule.train]
            '''

            input_coords = molecule.coords#.reshape(-1,n_atoms*3)
            input_NRF = molecule.mat_NRF.reshape(-1,_NC2)
            #input_eij = molecule.mat_eij
            output_matFE = molecule.mat_FE.reshape(-1,_NC2)
            output_FE = np.concatenate((molecule.forces.reshape(-1,n_atoms*3), 
                    molecule.energies.reshape(-1,1)), axis=1)
            output_F = molecule.forces.reshape(-1,n_atoms,3)
            output_E = molecule.energies.reshape(-1,1)
            output_E_postscale = ((output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])
            #output_atomFE = molecule.atomFE.reshape(-1,n_atoms) #per atom Es
            #output_atomNRF = molecule.atomNRF.reshape(-1,n_atoms) #per atom NRFs

            train_input_coords = np.take(input_coords, molecule.train, axis=0)
            test_input_coords = np.take(input_coords, molecule.test, axis=0)
            train_input_NRF = np.take(input_NRF, molecule.train, axis=0)
            test_input_NRF = np.take(input_NRF, molecule.test, axis=0)
            #train_input_eij = np.take(input_eij, molecule.train, axis=0)
            train_output_matFE = np.take(output_matFE, molecule.train, axis=0)
            test_output_matFE = np.take(output_matFE, molecule.test, axis=0)
            train_output_FE = np.take(output_FE, molecule.train, axis=0)
            train_output_E = np.take(output_E, molecule.train, axis=0)
            test_output_E = np.take(output_E, molecule.test, axis=0)
            #train_output_atomFE = np.take(output_atomFE, molecule.train, axis=0)
            #test_output_atomFE = np.take(output_atomFE, molecule.test, axis=0)
            #train_output_atomNRF = np.take(output_atomNRF, molecule.train, axis=0)
            #test_output_atomNRF = np.take(output_atomNRF, molecule.test, axis=0)

            train_mat_r = np.take(molecule.mat_r, molecule.train, axis=0)
            train_forces = np.take(molecule.forces, molecule.train, axis=0)
            test_forces = np.take(molecule.forces, molecule.test, axis=0)

            max_NRF1 = np.max(np.abs(train_input_NRF))
            max_FE1 = np.max(np.abs(train_output_FE))
            #max_atomFE1 = np.max(np.abs(train_output_atomFE))
            #max_atomNRF1 = np.max(np.abs(train_output_atomNRF))
            train_output_matFE_scaled = train_output_matFE / (2 * max_FE1) + 0.5
            train_input_NRF_scaled = np.take(input_NRF, molecule.train, axis=0)\
                    / max_NRF1
            print(train_input_coords.shape, train_output_FE.shape)
            print('max_NRF: {}, max_FE: {}'.format(max_NRF1, max_FE1))
            #print('max_atomE: {} max_atomNRF: {}'.format(max_atomFE1, 
                #max_atomNRF1))
            max_NRF = tf.constant(max_NRF1, dtype=tf.float32)
            max_FE = tf.constant(max_FE1, dtype=tf.float32)
            #max_atomFE = tf.constant(max_atomFE1, dtype=tf.float32)
            #max_atomNRF = tf.constant(max_atomNRF1, dtype=tf.float32)
            prescale = tf.constant(prescale1, dtype=tf.float32)
            n_atoms_tf = tf.constant(n_atoms, dtype=tf.int32)


            '''
            ##hard code permutations for now
            print('\n!!!! permuting hard-coded')
            orig_perm = [0,2,3,4,6,7,5,8]
            new_perm = [2,0,4,3,7,6,8,5]
            train_input_coords2 = np.copy(train_input_coords)
            train_input_coords2[:, orig_perm] = train_input_coords2[:, new_perm]
            train_forces2 = np.copy(train_forces)
            train_forces2[:, orig_perm] = train_forces[:, new_perm]
            train_output_E2 = np.copy(train_output_E)
            train_input_coords = np.concatenate((train_input_coords, 
                    train_input_coords2), axis=0)
            train_forces = np.concatenate((train_forces, train_forces2), axis=0)
            train_output_E = np.concatenate((train_output_E, train_output_E2), 
                    axis=0)
            train_output_matFE = \
                    Converter.get_simultaneous_interatomic_energies_forces2(
                    molecule.atoms, train_input_coords, train_forces, 
                    train_output_E, bias_type='1/r')
            Writer.write_xyz(train_input_coords, molecule.atoms, 
                    'train-coords.xyz', 'w')
            '''

            equiv_atoms = False
            if equiv_atoms:
                print('permutations - sorting of equiv_atoms')
                A = Molecule.find_bonded_atoms(molecule.atoms, 
                        molecule.coords[0])
                indices, pairs_dict = Molecule.find_equivalent_atoms(
                        molecule.atoms, A)
                print('indices', indices)
                print('pairs_dict', pairs_dict)
                all_sorted_list, all_resorted_list = Molecule.get_sorted_pairs(
                        train_input_NRF, pairs_dict)
                print('sorted', all_sorted_list[0])
                print('resorted', all_resorted_list[0])
                sys.stdout.flush()
                #print('\norig input', input)
                input = np.take_along_axis(train_input_NRF, 
                        all_sorted_list, axis=1)
                #print('\nnew input', input)
                output = np.take_along_axis(train_output_matFE, 
                        all_sorted_list, axis=1)
                np.savetxt('orig_input_NRF.txt', train_input_NRF)
                np.savetxt('orig_output_matFE.txt', train_output_matFE)
                np.savetxt('perm_input_NRF.txt', input)
                np.savetxt('perm_output_matFE.txt', output)

                sys.exit()


            train_output_E_postscale = ((train_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])
            test_output_E_postscale = ((test_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])

            val_points = 50 #50
            print('val points: {}'.format(val_points))
            ###get validation from within training
            train2, val = Molecule.make_train_test(train_output_E.flatten(), 
                    n_train=len(train_output_E)-val_points, n_test=val_points)
            train2_input_coords = np.take(train_input_coords, train2, axis=0)
            val_input_coords = np.take(train_input_coords, val, axis=0)
            train2_output_E = np.take(train_output_E, train2, axis=0)
            val_output_E = np.take(train_output_E, val, axis=0)
            train2_output_E_postscale = np.take(train_output_E_postscale, 
                    train2, axis=0)
            val_output_E_postscale = np.take(train_output_E_postscale, 
                    val, axis=0)
            train2_forces = np.take(train_forces, train2, axis=0)
            print('train2 max F: {}, min F {}'.format(
                np.max(train2_forces), np.min(train2_forces)))
            val_forces = np.take(train_forces, val, axis=0)
            train2_output_matFE = np.take(train_output_matFE, train2, axis=0)
            val_output_matFE = np.take(train_output_matFE, val, axis=0)
            #train2_output_atomFE = np.take(train_output_atomFE, train2, 
                    #axis=0)
            #val_output_atomFE = np.take(train_output_atomFE, val, axis=0)
            #train2_output_atomNRF = np.take(train_output_atomNRF, train2, 
                    #axis=0)
            #val_output_atomNRF = np.take(train_output_atomNRF, val, axis=0)
            train2_input_NRF = np.take(train_input_NRF, train2, axis=0)
            val_input_NRF = np.take(train_input_NRF, val, axis=0)


        if get_data == False: #hardcoded
            #prescale1 = [-167314.95257129727, -167288.0645391319, 
                    #-184.62847828601284, 211.23633766510102]
            prescale1 = [-167315.01917961572, -167288.834575252, 
                    -260.631337453833, 284.98286516421035]
            #max_NRF1 = 12304.734510536122
            max_NRF1 = 13010.961100355082 
            #max_FE1 = 211.23633766510105
            max_FE1 = 284.98286516421035
            print('\n!!!!scaled values over-written')
            print('prescale', prescale1)
            print('max_NRF', max_NRF1)
            print('max_FE', max_FE1)
            max_NRF = tf.constant(max_NRF1, dtype=tf.float32)
            max_FE = tf.constant(max_FE1, dtype=tf.float32)
            prescale = tf.constant(prescale1, dtype=tf.float32)
            n_atoms_tf = tf.constant(n_atoms, dtype=tf.int32)


        file_name='best_model'
        monitor_loss='loss'
        set_patience=5000
        print('monitor_loss:', monitor_loss)
        print('set_patience', set_patience)
        mc = ModelCheckpoint(file_name, monitor=monitor_loss, mode='min', 
                save_best_only=True, save_weights_only=True
                )
        es = EarlyStopping(monitor=monitor_loss, patience=set_patience)



        #coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
        #NRF_layer = CoordsToNRF_test(atoms_flat, max_NRF, _NC2, 
                #name='NRF_layer')(coords_layer)
        #model = NRF_layer
        
        max_depth = 1
        print('max_depth', max_depth)
        e_loss = [0, 1, 1, 1, 1, 1, 1] #1
        f_loss = [0, 0, 0, 0, 0, 0, 0] #0
        grad_loss = [0, 1000, 1000, 1000, 1000, 1000, 1000] #1000
        q_loss = [1, 1, 1, 1, 1, 1, 1] #_NC2 
        best_error_all = 1e20
        best_iteration = 0
        best_model = None
        for l in range(0, max_depth):
            print('l', l)
            l2 = str(l)
            print('E loss weight: {} \nrecompF loss: {} '\
                    '\ndE_dx loss weight: {} \nq loss weight: {}'.format(
                    e_loss[l], f_loss[l], grad_loss[l], q_loss[l]))
            #if l > 0:
                #model = concatenate([model2[3], NRF_layer])
            coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
            NRF_layer = CoordsToNRF_test(atoms_flat, max_NRF, _NC2, 
                    name='NRF_layer')(coords_layer)

            net_layers = []
            for x in range(_NC2):
                net_layerA = Dense(units=5, activation='sigmoid', 
                        name='net_layerA_{}'.format(x))(NRF_layer)
                net_layerB = Dense(units=4, activation='sigmoid', 
                        name='net_layerB_{}'.format(x))(net_layerA)
                net_layerC = Dense(units=3, activation='sigmoid', 
                        name='net_layerC_{}'.format(x))(net_layerB)
                net_layerD = Dense(units=1, activation='sigmoid', 
                        name='net_layerD_{}'.format(x))(net_layerC)
                net_layers.append(net_layerD)
            net_layer2 = concatenate(net_layers, name='net_layer2')


            #net_layer = Dense(units=1000, activation='sigmoid', 
                    #name='net_layer')(NRF_layer) #(model)
            #net_layer2 = Dense(units=_NC2, activation='sigmoid', 
                    #name='net_layer2')(net_layer)
            scale_layer = ScaleFE(_NC2, max_FE, name='scale_layer')(net_layer2)
            eij_layer = EijMatrix_test(n_atoms, _NC2, prescale, 
                    name='eij_layer')([coords_layer, scale_layer])
            eij_layer2 = EijMatrix_test2(n_atoms, _NC2, 
                    name='eij_layer2')([coords_layer, scale_layer])
            dE_dx = EnergyGradient(n_atoms, _NC2, 
                    name='dE_dx')([eij_layer, coords_layer])



            #x = tf.constant(train2_input_coords)
            #with tf.GradientTape(watch_accessed_variables=False) as tape:
                #tape.watch(x)
                #y = model(x)[0]
                #y = EijMatrix_test(n_atoms, _NC2, prescale)([x, triangle_layer])

            #dE_dx = tape.gradient(y, x)
            #print('xxx', net_layer.trainable_variables)


            '''
            with tf.GradientTape() as tape:
                y = EijMatrix_test(n_atoms, _NC2, prescale)(
                        [coords_layer, triangle_layer])
            dE_dx = tape.gradient(y, train2_input_coords)

            for var, g in zip(EijMatrix_test(n_atoms, _NC2, prescale
                    ).trainable_variables, dE_dx):
                print(f'{var.name}, shape: {g.shape}')
            '''


            '''
            optimizers = [tf.keras.optimizers.Adam(learning_rate=1e-5),
                    tf.keras.optimizers.Adam(learning_rate=1e-3)]
            optimizers_and_layers = [(optimizers[0], model.layers[0:]), 
                    (optimizers[1], model.layers[-1])]
            custom_optimizer = tfa.optimizers.MultiOptimizer(
                    optimizers_and_layers)

            def get_lr_metric(optimizer):
                def lr(y_true, y_pred):
                    return optimizer.lr
                return lr

            rlrop = ReduceLROnPlateau(monitor='loss', factor=0.8, 
                    patience=1000, min_lr=1e-6)
            optimizer = keras.optimizers.Adam(lr=0.001, 
                    beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True)
            lr_metric = get_lr_metric(optimizer)
            '''



            model = Model(
                    inputs=[coords_layer], 
                    #inputs=[NRF_layer], 
                    outputs=[
                        #coords_layer,
                        #forces_layer, 
                        eij_layer, 
                        eij_layer2, 
                        dE_dx, 
                        scale_layer, 
                        #net_layer2,
                        #eij_layer2
                        ], 
                    )

            #with tf.GradientTape(watch_accessed_variables=False) as tape:
                #tape.watch(x)
                #y = model(x)[0]
            #print(l, 'start grads\n', tape.gradient(y, x)) 
            #print(l, 'start trainable vars\n', model.trainable_variables)
            #print()

            #https://stackoverflow.com/questions/69597055/
                #custom-loss-function-with-gradient-tape-tf2-6
            def custom_loss_pass(model, x_tensor, forces):
                def custom_loss(y_true, y_pred):
                    with tf.GradientTape() as tape:
                        tape.watch(x_tensor)
                        output_E = model(x_tensor)[0]
                    dEdX = tape.gradient(output_E, x_tensor)
                    minus_dEdX = dEdX * -1
                    f_diff = forces - minus_dEdX
                    f_loss = tf.reduce_mean(tf.square(f_diff))
                    #e_diff = energies - output_E
                    #e_loss = tf.reduce_mean(tf.square(e_diff))
                    #_loss = f_loss * f_weight #+ e_loss * e_weight
                    return f_loss
                return custom_loss




            if l == 0:
                model.compile(
                        loss={
                            #'for_forces': custom_loss_pass(model, 
                            #tf.constant(train2_input_coords, 
                                #dtype=tf.float32),
                            #tf.constant(train2_forces, dtype=tf.float32),
                            #),
                            'energy_gradient': 'mse',
                            'eij_matrix_test': 'mse',
                            'eij_matrix_test2': 'mse',
                            'scale_fe': 'mse',
                            #'coords_layer': 'mse'
                            },
                        loss_weights={
                            #'for_forces': 0, #grad_loss - q_loss[l],
                            'energy_gradient': grad_loss[l],
                            'eij_matrix_test': e_loss[l],
                            'eij_matrix_test2': f_loss[l],
                            'scale_fe': q_loss[l],
                            #'coords_layer': 0
                            },
                        optimizer='adam')
            if l != 0:
                model.compile(
                        loss={
                            #'for_forces_'+l2: custom_loss_pass(model, 
                            #tf.constant(train_input_coords, dtype=tf.float32),
                            #tf.constant(train_forces, dtype=tf.float32),
                            #),
                            'energy_gradient_'+l2: 'mse',
                            'eij_matrix_test_'+l2: 'mse',
                            'eij_matrix_test2_'+l2: 'mse',
                            'scale_fe_'+l2: 'mse',
                            #'coords_layer': 'mse'
                            },
                        loss_weights={
                            #'for_forces_'+l2: 0, #grad_loss - q_loss[l],
                            'energy_gradient_'+l2: grad_loss[l],
                            'eij_matrix_test_'+l2: e_loss[l],
                            'eij_matrix_test2_'+l2: f_loss[l],
                            'scale_fe_'+l2: q_loss[l],
                            #'coords_layer': 0
                            },
                        optimizer='adam')

            model.summary()

            print('initial learning rate:', K.eval(model.optimizer.lr))

            #load_first = True
            if load_first and l != 0:
                print('!!!loading previous weights first')
                l3 = str(l-1)
                #model.load_weights('best_model1')
                model.load_weights('best_ever_model_'+l3)

            '''
            if load_first and l == 0:
                print('!!!loading previous weights first')
                model.load_weights('1/best_ever_model_5')
            '''

            #fit = True
            if fit:
                model.fit(train2_input_coords, 
                        #train2_input_NRF,
                        [
                            #train2_input_coords,
                            #train2_output_E_postscale, 
                            train2_output_E_postscale, 
                            train2_forces, 
                            train2_forces, 
                            #train2_output_matFE, 
                            train2_output_matFE, 
                            #train2_forces, 
                            ],
                        validation_data=(val_input_coords, 
                            #val_input_NRF,
                            [
                                #val_input_coords,
                                #val_output_E_postscale, 
                                val_output_E_postscale, 
                                val_forces, 
                                val_forces, 
                                #val_output_matFE, 
                                val_output_matFE, 
                                #val_forces, 
                                ]),
                        epochs=5000,#00, 
                        verbose=2,
                        #callbacks=[mc],
                        callbacks=[es,mc],
                        #callbacks=[es,mc,rlrop],
                        )

                best_error = model.evaluate(train2_input_coords, 
                        #train_input_NRF,
                        [
                            #train_input_coords,
                            #train2_output_E_postscale, 
                            train2_output_E_postscale, 
                            train2_forces, 
                            train2_forces, 
                            #train_output_matFE, 
                            train2_output_matFE, 
                            #train_forces, 
                            ],
                        verbose=0)
                print(l, 'model error train2: ', best_error)

                best_error2 = model.evaluate(val_input_coords, 
                        [
                            val_output_E_postscale, 
                            val_forces, 
                            val_forces, 
                            val_output_matFE, 
                            ],
                        verbose=0)
                print(l, 'model error val: ', best_error2)

                best_error3 = model.evaluate(train_input_coords, 
                        [
                            train_output_E_postscale, 
                            train_forces, 
                            train_forces, 
                            train_output_matFE, 
                            ],
                        verbose=0)
                print(l, 'model error train: ', best_error3)

                best_error4 = model.evaluate(test_input_coords, 
                        [
                            test_output_E_postscale, 
                            test_forces, 
                            test_forces, 
                            test_output_matFE, 
                            ],
                        verbose=0)
                print(l, 'model error test: ', best_error4)

                best_error5 = model.evaluate(input_coords, 
                        [
                            output_E_postscale, 
                            output_F, 
                            output_F, 
                            output_matFE, 
                            ],
                        verbose=0)
                print(l, 'model error all: ', best_error5)

                model.save_weights('best_ever_model_'+l2)

                #with tf.GradientTape(watch_accessed_variables=False) as tape:
                    #tape.watch(x)
                    #y = model(x)[0]
                #print(l, 'end grads\n', tape.gradient(y, x)) 
                #print(l, 'end trainable vars\n', model.trainable_variables)

                '''
                if best_error[0] >= best_error_all:
                    break
                if best_error[0] < best_error_all:
                    model.save_weights('best_ever_model')
                    best_model = model
                    best_iteration = l
                    best_error_all = best_error[0]
                    print('best model was achieved on layer %d' % l)
                    print('its error was: {}'.format(best_error_all))
                '''
            if time.time()-start_time >= 302400: #3.5 days
                print('\n*** Time limit reached, ending training ***')
                break
            #model.trainable = False
            #model2 = model(coords_layer)
            #print(l, 'model2', model2)
            print()

        '''
        if fit:
            print('best_iteration', best_iteration)
            model = best_model #use best model
            print('training completed\n')
        '''

        #load_weights = False
        if load_weights:
            print('load_weights', load_weights)
            model.load_weights('../model/best_ever_model_6')
            #model.load_weights('best_model')

        if fit:
            best_error = model.evaluate(train_input_coords, 
                    [train_output_E_postscale, #train_output_E_postscale, 
                    train_forces, train_forces,
                    train_output_matFE],verbose=0)
            print(l, len(train_input_coords), 'train model error: ', 
                    best_error)




            sys.stdout.flush()
            #sys.exit()

            #E_layer_output = Model(inputs=model.input, 
                    #outputs=model.get_layer('eij_matrix_test').output)

            for i in range(3):
                print('\n\t', i)
                print('coords\n', train2_input_coords[i])
                print('energy\n', train2_output_E_postscale[i])
                print('E\n', train2_output_E[i])
                print('forces\n', train2_forces[i])
                prediction = model.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3)
                        #train2_input_NRF[i].reshape(1,_NC2)
                        )
                print('prediction')
                for count, p in enumerate(prediction):
                    print(count, p)

                '''
                E = layer_output.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3))
                print('pred\n', E[0])
                #print('energy\n', train_output_E[i])
                #E2 = energy_layer_output.predict(
                        #train_input_coords[i].reshape(1,n_atoms,3))
                #print('pred postscale\n', E2[0])
                '''

                x1 = tf.constant(train2_input_coords[i].reshape(1,n_atoms,3), 
                        dtype=tf.float32)
                with tf.GradientTape(watch_accessed_variables=False) as tape2:
                    tape2.watch(x1)
                    y = model(x1)[0]
                gradient1 = tape2.gradient(y, x1) * -1
                print('tape_grad\n', gradient1) 

                #grad = grad_layer_output.predict(
                        #train2_input_coords[i].reshape(1,n_atoms,3))
                #print('grad_lambda\n', grad[0])
                variance, translations, rotations = Network.check_invariance(
                        train2_input_coords[i], prediction[2][0],
                        #gradient1[0]
                        )
                print('variance\n', variance, translations, rotations)
                #print('atomE\n', atomE_layer_output.predict(
                        #train2_input_coords[i].reshape(1,n_atoms,3)))
                #print('mat_FE\n', train2_output_matFE[i])
                #matFE_pred = scale_layer_output.predict(
                        #train2_input_coords[i].reshape(1,n_atoms,3))[0]
                #print('pred matFE', matFE_pred)

                '''
                triangle = triangle_layer_output.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3))[0]
                print('triangle\n', triangle)
                atomE = np.sum(triangle, axis=0)
                print('column sums, atomE\n', atomE)
                print(molecule.atoms)
                print('sum atomE\n', np.sum(atomE)/2)
                print('pred mat_FE\n', prediction[2])
                '''

                '''
                lower_mask = np.tri(n_atoms,dtype=bool, k=-1)
                out = np.zeros((n_atoms, n_atoms))
                out[lower_mask] = prediction[2][0] 
                out3 = out + out.T
                print('upper lower triangle\n', out3)
                atomE = np.sum(out3, axis=0)
                print('column sums, atomE', atomE)
                print(molecule.atoms)
                print('sum atomE', np.sum(atomE)/2)
                '''

                '''
                atomE_pred = scale_atomE_layer_output.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3))[0]
                #atomE_pred = atomE_layer_output.predict(
                        #train2_input_coords[i].reshape(1,n_atoms,3))[0]
                print('pred atomE', atomE_pred)
                print('sum pred atomE', np.sum(atomE_pred))
                sum_atomE_pred = sum_atomE_layer_output.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3))[0]
                print('pred sum_atomE', sum_atomE_pred)
                print('train2 atomE', train2_output_atomFE[i])
                print('sum train2 atomE', np.sum(train2_output_atomFE[i]))
                '''

                '''
                print('atomNRF', train2_output_atomNRF[i] / max_atomNRF1)
                atomNRF_pred = atomNRF_layer_output.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3))[0]
                print('model atomNRF', atomNRF_pred)
                '''
                




                '''
                np_train_input_coords = np.asarray([train_input_coords[i]], 
                        np.float32)
                tf_train_input_coords = tf.convert_to_tensor(
                        np_train_input_coords, np.float32)
                tf_train_input_coords = tf.Variable(tf_train_input_coords)
                gradient = tf.gradients(model(tf_train_input_coords)[0], 
                        tf_train_input_coords, 
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
                #print(model(tf_train_input_coords)[0])
                with tf.Session() as sess: #to get tensor to numpy
                    sess.run(tf.global_variables_initializer())
                    result_output_scaled = sess.run(gradient[0][0])
                    print('grad\n', result_output_scaled * -1) #-tive grad
                    variance, translations, rotations = \
                            Network.check_invariance(
                            train_input_coords[i], result_output_scaled)
                    print('invariance\n', variance, translations, rotations)
                '''
                sys.stdout.flush()



        #inp_out_pred = True
        if inp_out_pred:

            print('Predict all data')
            prediction = model.predict(input_coords)
            prediction_E = prediction[0].flatten()
            #prediction_F = prediction[1].flatten()
            prediction_matFE = prediction[3]
            #prediction_atomFE = prediction[1]
            #prediction_E = prediction[2].flatten()

            x_tensor = tf.constant(input_coords, dtype=tf.float32)
            with tf.GradientTape(watch_accessed_variables=False) as tape3:
                tape3.watch(x_tensor)
                y = model(x_tensor)[0]
            #prediction_F = tape3.gradient(y, x_tensor).numpy() * -1
            prediction_F = prediction[2]
            print(prediction_F.shape)
            print(prediction_F[0], output_F[0])


            Writer.write_csv([output_E_postscale, prediction_E], 
                    'E2', 'actual_E,prediction_E')

            atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                    zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
            atom_pairs = []
            for i in range(len(molecule.atoms)):
                for j in range(i):
                    atom_pairs.append('{}_{}'.format(atom_names[i], 
                        atom_names[j]))
            input_header = ['input_' + s for s in atom_pairs]
            input_header = ','.join(input_header)
            output_header = ['output_' + s for s in atom_pairs]
            output_header = ','.join(output_header)
            prediction_header = ['prediction_' + s for s in atom_pairs]
            prediction_header = ','.join(prediction_header)
            header = input_header + ',' + output_header + ',' + \
                    prediction_header
            #header = 'input,output,prediction'
            print(input_NRF.shape)
            #print(output_atomNRF.shape)
            #print(output_atomFE.shape)
            #print(prediction_atomFE.shape)
            print(prediction_E.shape)
            Writer.write_csv([
                    input_NRF,
                    #output_atomNRF, 
                    #output_atomFE, 
                    output_matFE,
                    #prediction_atomFE
                    prediction_matFE
                    ], 'inp_out_pred2', 
                    header)
            sys.stdout.flush()

            mae, rms, msd = Binner.get_error(output_E_postscale.flatten(), 
                    prediction_E.flatten())
            print('\n{} E All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_E_postscale), mae, rms, 
                    msd))
            bin_edges, hist = Binner.get_scurve(output_E_postscale.flatten(), 
                    prediction_E.flatten(), 'all-hist1.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-E.png')

            mae, rms, msd = Binner.get_error(output_F.flatten(), 
                    prediction_F.flatten())
            print('\n{} grad F All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_F), mae, rms, msd))
            bin_edges, hist = Binner.get_scurve(output_F.flatten(), 
                    prediction_F.flatten(), 'all-hist3.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-F.png')

            #'''
            mae, rms, msd = Binner.get_error(output_matFE.flatten(), 
                    prediction_matFE.flatten())
            print('\n{} mat_(F)E All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_matFE), mae, rms, msd))
            bin_edges, hist = Binner.get_scurve(output_matFE.flatten(), 
                    prediction_matFE.flatten(), 'all-hist2.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-matFE.png')
            #'''

            '''
            mae, rms, msd = Binner.get_error(output_atomFE.flatten(), 
                    prediction_atomFE.flatten())
            print('\n{} atom_(F)E All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_atomFE), mae, rms, msd))
            bin_edges, hist = Binner.get_scurve(output_atomFE.flatten(), 
                    prediction_atomFE.flatten(), 'all-hist4.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-atomFE.png')
            '''

            sys.stdout.flush()
            #sys.exit()



        sys.stdout.flush()

        get_stats = False
        if get_stats:

            train_output_E_postscale = train_output_E_postscale
            test_output_E_postscale = test_output_E_postscale
            train_forces = train_forces
            test_forces = test_forces

            prediction = model.predict(train_input_coords)
            train_prediction1 = prediction[0].flatten()
            train_prediction2 = prediction[1].flatten()
            train_prediction3 = prediction[3].flatten()
            train_prediction4 = prediction[2].flatten()


            train_mae, train_rms, train_msd = Binner.get_error(
                    train_output_E_postscale.flatten(), 
                    train_prediction1.flatten())
            print('\nE Train MAE: {} \nTrain RMS: {} \nTrain MSD: {}'.format(
                    train_mae, train_rms, train_msd))


            prediction = model.predict(test_input_coords)
            test_prediction1 = prediction[0].flatten()
            test_prediction2 = prediction[1].flatten()
            test_prediction3 = prediction[3].flatten()
            test_prediction4 = prediction[2].flatten()

            test_mae, test_rms, test_msd = Binner.get_error(
                    test_output_E_postscale.flatten(), 
                    test_prediction1.flatten())
            print('\nE Test MAE: {} \nTest RMS: {} \nTest MSD: {}'.format(
                    test_mae, test_rms, test_msd))
            train_bin_edges, train_hist = Binner.get_scurve(
                    train_output_E_postscale.flatten(), 
                    train_prediction1.flatten(), 
                    'train-hist1.txt')
            test_bin_edges, test_hist = Binner.get_scurve(
                    test_output_E_postscale.flatten(), #actual
                    test_prediction1.flatten(), #prediction
                    'test-hist1.txt')
            Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                    [train_hist, test_hist], ['train', 'test'], 
                    'Error', '% of points below error', 's-curves1.png')
            sys.stdout.flush()

            train_mae, train_rms, train_msd = Binner.get_error(
                    train_forces.flatten(), 
                    train_prediction2.flatten())
            print('\n{} grad forces Train MAE: {} \nTrain RMS: {} '\
                    '\nTrain MSD: {}'.format(
                    len(train_forces), train_mae, train_rms, train_msd))
            test_mae, test_rms, test_msd = Binner.get_error(
                    test_forces.flatten(), 
                    test_prediction2.flatten())
            print('\n{} grad forces Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(
                    len(test_forces), test_mae, test_rms, test_msd))
            train_bin_edges, train_hist = Binner.get_scurve(
                    train_forces.flatten(), 
                    train_prediction2.flatten(), 
                    'train-hist3.txt')
            test_bin_edges, test_hist = Binner.get_scurve(
                    test_forces.flatten(), #actual
                    test_prediction2.flatten(), #prediction
                    'test-hist3.txt')
            Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                    [train_hist, test_hist], ['train', 'test'], 
                    'Error', '% of points below error', 's-curves3.png')
            sys.stdout.flush()

            train_mae, train_rms, train_msd = Binner.get_error(
                    train_forces.flatten(), 
                    train_prediction3.flatten())
            print('\n{} recomp forces Train MAE: {} \nTrain RMS: {} '\
                    '\nTrain MSD: {}'.format(
                    len(train_forces), train_mae, train_rms, train_msd))
            test_mae, test_rms, test_msd = Binner.get_error(
                    test_forces.flatten(), 
                    test_prediction3.flatten())
            print('\n{} recomp forces Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(
                    len(test_forces), test_mae, test_rms, test_msd))
            train_bin_edges, train_hist = Binner.get_scurve(
                    train_forces.flatten(), 
                    train_prediction3.flatten(), 
                    'train-hist4.txt')
            test_bin_edges, test_hist = Binner.get_scurve(
                    test_forces.flatten(), #actual
                    test_prediction3.flatten(), #prediction
                    'test-hist4.txt')
            Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                    [train_hist, test_hist], ['train', 'test'], 
                    'Error', '% of points below error', 's-curves4.png')
            sys.stdout.flush()

            train_mae, train_rms, train_msd = Binner.get_error(
                    train_output_matFE.flatten(), 
                    train_prediction4.flatten())
            print('\n{} mat_(F)E Train MAE: {} \nTrain RMS: {} '\
                    '\nTrain MSD: {}'.format(
                    len(train_output_matFE), train_mae, train_rms, train_msd))
            test_mae, test_rms, test_msd = Binner.get_error(
                    test_output_matFE.flatten(), 
                    test_prediction4.flatten())
            print('\n{} mat_(F)E Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(
                    len(test_output_matFE), test_mae, test_rms, test_msd))
            train_bin_edges, train_hist = Binner.get_scurve(
                    train_output_matFE.flatten(), 
                    train_prediction4.flatten(), 
                    'train-hist2.txt')
            test_bin_edges, test_hist = Binner.get_scurve(
                    test_output_matFE.flatten(), #actual
                    test_prediction4.flatten(), #prediction
                    'test-hist2.txt')
            Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                    [train_hist, test_hist], ['train', 'test'], 
                    'Error', '% of points below error', 's-curves2.png')
            sys.stdout.flush()


            print('Predict all data')
            prediction = model.predict(input_coords)
            prediction_E = prediction[0].flatten()
            prediction_F = prediction[1].flatten()
            prediction_recompF = prediction[3].flatten()
            prediction_matFE = prediction[2] #- prescale1[5]
            Writer.write_csv([output_E_postscale, prediction_E], 
                    'E', 'actual_E,prediction_E')
            atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                    zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
            atom_pairs = []
            for i in range(len(molecule.atoms)):
                for j in range(i):
                    atom_pairs.append('{}_{}'.format(atom_names[i], 
                        atom_names[j]))
            input_header = ['input_' + s for s in atom_pairs]
            input_header = ','.join(input_header)
            output_header = ['output_' + s for s in atom_pairs]
            output_header = ','.join(output_header)
            prediction_header = ['prediction_' + s for s in atom_pairs]
            prediction_header = ','.join(prediction_header)
            header = input_header + ',' + output_header + ',' + \
                    prediction_header
            #header = 'input,output,prediction'
            Writer.write_csv([input_NRF, output_matFE, 
                    prediction_matFE], 'inp_out_pred', header)
            sys.stdout.flush()

            mae, rms, msd = Binner.get_error(output_E_postscale.flatten(), 
                    prediction_E.flatten())
            print('\n{} E All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_E_postscale), mae, rms, 
                    msd))
            bin_edges, hist = Binner.get_scurve(output_E_postscale.flatten(), 
                    prediction_E.flatten(), 'all-hist1.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-E.png')

            mae, rms, msd = Binner.get_error(output_F.flatten(), 
                    prediction_F.flatten())
            print('\n{} grad F All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_F), mae, rms, msd))
            bin_edges, hist = Binner.get_scurve(output_F.flatten(), 
                    prediction_F.flatten(), 'all-hist3.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-F.png')

            mae, rms, msd = Binner.get_error(output_F.flatten(), 
                    prediction_recompF.flatten())
            print('\n{} recomp F All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_F), mae, rms, msd))
            bin_edges, hist = Binner.get_scurve(output_F.flatten(), 
                    prediction_recompF.flatten(), 'all-hist4.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-recompF.png')

            mae, rms, msd = Binner.get_error(output_matFE.flatten(), 
                    prediction_matFE.flatten())
            print('\n{} mat_(F)E All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_matFE), mae, rms, msd))
            bin_edges, hist = Binner.get_scurve(output_matFE.flatten(), 
                    prediction_matFE.flatten(), 'all-hist2.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-matFE.png')
            sys.stdout.flush()

        #print('Done')
        #sys.exit()
        #######################---TEST---##############################
        #######################---TEST---##############################

        return model


    def dummy_get_coord_FE_model(self, molecule, prescale1):
        with_lambda = False
        if with_lambda:
            print('with_lambda')
            #### WITH LAMBDA LAYERS
            coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
            NRF_layer = Lambda(Network.get_NRF_from_coords, 
                    input_shape=(n_atoms,3), output_shape=(n_atoms,n_atoms), 
                    name='NRF_layer', arguments={'atoms2':atoms2, 
                    'max_NRF':max_NRF})(coords_layer)
            flatten_NRF = Lambda(Network.flatten_triangle, 
                    input_shape=(n_atoms,n_atoms), output_shape=(_NC2,), 
                    name='flatten_NRF')(NRF_layer)
            net_layer = Dense(units=1000, activation='sigmoid', 
                    name='net_layer')(flatten_NRF)
            net_layer2 = Dense(units=_NC2, activation='sigmoid', 
                    name='net_layer2')(net_layer)
            triangle_layer = Lambda(Network.get_triangle, input_shape=(_NC2,), 
                    output_shape=(n_atoms,n_atoms), 
                    name='triangle_layer')(net_layer2)
            eij_layer = Lambda(Network.get_FE_eij_matrix,
                    input_shape=(n_atoms,n_atoms), 
                    #output_shape=(1,),
                    output_shape=(n_atoms*3+1,),
                    name='eij_layer', arguments={'coords':coords_layer, 
                    'max_FE':max_FE}
                    )(triangle_layer)

        else:
            print('with_custom')
            #### WITH CUSTOM LAYERS
            coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
            NRF_layer = CoordsToNRF(atoms2, max_NRF, name='NRF_layer')(
                    coords_layer)
            flatten_NRF = FlatTriangle(_NC2, name='flatten_NRF')(NRF_layer)
            net_layer = Dense(units=1000, activation='sigmoid', 
                    name='net_layer')(flatten_NRF)
            net_layer2 = Dense(units=_NC2, activation='sigmoid', 
                    name='net_layer2')(net_layer)
            triangle_layer = Triangle(n_atoms, name='triangle_layer')(
                    net_layer2)
            eij_layer = EijMatrix(n_atoms, max_FE, name='eij_layer')(
                    [triangle_layer, coords_layer])
            energy_layer = TotalEnergy(prescale, name='energy_layer')(
                    eij_layer)


        #model = Model(inputs=[coords_layer], outputs=[eij_layer, net_layer2])
        model = Model(inputs=[coords_layer], outputs=[energy_layer])
        model.compile(#loss=cl,
                #loss='mse', 
                #loss={'eij_layer': cl, 'net_layer2': 'mse'},
                #loss={'energy_layer': cl, 'net_layer2': 'mse'},
                loss={'energy_layer': 'mse'},
                optimizer='adam', 
                metrics=['mae']) #, 'acc']) #mean abs error, accuracy
        model.summary()



        fit = True
        if fit:
            model.fit(train_input_coords, 
                    #[train_output_FE, train_output_matFE_scaled],
                    #[train_output_E, train_output_matFE_scaled],
                    train_output_E_postscale,
                    epochs=100, 
                    verbose=2,
                    #callbacks=[es],
                    callbacks=[es,mc],
                    )
            model.save_weights('model/model_weights.h5')
        else:
            '''
            model = load_model(file_name, 
                    custom_objects={'CoordsToNRF': CoordsToNRF, 
                        'FlatTriangle': FlatTriangle, 'Traingle': Triangle,
                        'EijMatrix': EijMatrix, 
                        #'custom_loss': custom_loss1(weights)
                        })
            '''
            model.load_weights('model/model_weights.h5')


        best_error = model.evaluate(train_input_coords, 
                #[train_output_FE, train_output_matFE_scaled],
                train_output_E_postscale,
                verbose=0)
        print('model error: ', best_error)

        #last_layer_output = Model(inputs=model.input, 
                #outputs=model.get_layer('eij_layer').output)
        #FE = NRF_layer_output.predict(train_input)

        #'''
        gradients = K.gradients(model.output, model.input)
        print(gradients)
        '''
        with tf.Session() as sess: #to get tensor to numpy
            sess.run(tf.global_variables_initializer())
            result_output_scaled = sess.run(gradients)
            print(result_output_scaled)
        '''



        #####get gradient oof network model
        print('\nget network model gradient')
        for i in range(1):
            print(i)
            np_train_input_coords = np.asarray([train_input_coords[i]], 
                    np.float32)
            tf_train_input_coords = tf.convert_to_tensor(np_train_input_coords, 
                    np.float32)
            tf_train_input_coords = tf.Variable(tf_train_input_coords)
            gradient = tf.gradients(model(tf_train_input_coords), 
                    tf_train_input_coords, 
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)
            #X = tf.placeholder(tf.float32, shape=(n_atoms,3))
            #Y = model([X])
            #fn = K.function([X], K.gradients(Y, [X]))
            #print(fn([np.ones((n_atoms,3), dtype=np.float32)]))
            with tf.Session() as sess: #to get tensor to numpy
                sess.run(tf.global_variables_initializer())
                result_output_scaled = sess.run(gradient[0])
                print(result_output_scaled)

        sys.exit()
        #'''

        prediction = model.predict(train_input_coords)
        #train_prediction = prediction[:,-1].flatten()
        #train_prediction = prediction[0][:,-1].flatten()
        train_prediction = prediction.flatten()

        print('\nprediction')
        print(train_prediction)
        print('actual E')
        print(train_output_E.flatten())
        print('diff')
        print(train_output_E.flatten() - train_prediction)

        train_mae, train_rms = Binner.get_error(train_output_E.flatten(), 
                    train_prediction.flatten())
        print('\nTrain MAE: {} \nTrain RMS: {}'.format(train_mae, train_rms))


        prediction = model.predict(test_input_coords)
        #prediction = prediction[:,-1].flatten()
        #test_prediction = prediction[0][:,-1].flatten()
        test_prediction = prediction.flatten()

        print('\nprediction')
        print(test_prediction)
        print('actual E')
        print(test_output_E.flatten())
        print('diff')
        print(test_output_E.flatten() - test_prediction)

        test_mae, test_rms = Binner.get_error(test_output_E.flatten(), 
                    test_prediction.flatten())
        print('\nTest MAE: {} \nTest RMS: {}'.format(test_mae, test_rms))

        scurves = True
        if scurves:
            train_bin_edges, train_hist = Binner.get_scurve(
                    train_output_E.flatten(), 
                    train_prediction.flatten(), 
                    'train-hist.txt')

            test_bin_edges, test_hist = Binner.get_scurve(
                    test_output_E.flatten(), #actual
                    test_prediction.flatten(), #prediction
                    'test-hist.txt')
           
            Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                    [train_hist, test_hist], ['train', 'test'], 
                    'Error', '% of points below error', 's-curves.png')




    def get_decompE_sum_model(self, molecule, nodes, input, output):
        start_time = time.time()
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        n_nodes = nodes #1000
        n_epochs = 100000 #100000
        sumE_weight = 1
        print('summed E weighted by: {}'.format(sumE_weight))
        input_scale_method = 'A'
        output_scale_method = 'D'

        train_input = input
        train_output = output

        train = True #False
        if train:
            train_input = np.take(input, molecule.train, axis=0)
            train_output = np.take(output, molecule.train, axis=0)


        scaled_input, self.scale_input_max, self.scale_input_min = \
                Network.get_scaled_values(train_input, np.amax(train_input), 
                np.amin(train_input), method=input_scale_method)
        print('INPUT: \nscale_max: {}\nscale_min: {}'\
                '\nnstructures: {}\n'.format(
                self.scale_input_max, self.scale_input_min, len(train_input)))
        scaled_output, self.scale_output_max, self.scale_output_min = \
                Network.get_scaled_values(train_output, 
                np.amax(np.absolute(train_output)), 
                #np.amax(np.absolute(np.sum(train_output, axis=1))), 
                np.amin(np.absolute(train_output)), 
                #np.amin(np.absolute(np.sum(train_output, axis=1))), 
                method=output_scale_method)
        sum_output = np.sum(scaled_output, axis=1).reshape(-1,1)
        all_scaled_output = np.concatenate((scaled_output, sum_output), 
                axis=1)
        #all_scaled_output = sum_output
        print('OUTPUT: \nscale_max: {}\nscale_min: {}'\
                '\nnstructures: {}\n'.format(
                self.scale_output_max, self.scale_output_min, 
                len(train_output)))

        print('input shape: {}'.format(scaled_input.shape))
        print('output shape: {}'.format(scaled_output.shape))


        #setup network with a Lambda layer output concatenated to the output
        #NC2 + 1 outputs

        def sum_layer(x):
            #y = K.sum(x)
            y = K.sum(x,axis=1,keepdims=True)
            return y

        def custom_loss1(weights):
            def custom_loss(y_true, y_pred):
                return K.mean(K.abs(y_true - y_pred) * weights)
            return custom_loss

        #train network
        max_depth = 1 #6
        best_error = 10000
        file_name = 'best_model'
        #file_name = 'best_ever_model'
        mc = ModelCheckpoint(file_name, monitor='loss', 
                mode='min', 
                save_best_only=True)
        es = EarlyStopping(monitor='loss', 
                patience=500)
        model_in = Input(shape=(scaled_input.shape[1],))
        model = model_in

        #weights = np.ones((scaled_input.shape[0], _NC2+1))
        #weights[:,-1] = sumE_weight #weight the total E to be 10x 
                #more important than decompsed Es
        weights = np.zeros((_NC2+1)) #np.ones((_NC2+1)) #
        weights[-1] = sumE_weight
        #print(weights, weights.shape)
        #weights_tensor = Input(shape=(_NC2+1,))
        #cl = partial(custom_loss1, weights=weights_tensor)
        cl = custom_loss1(weights)


        for i in range(0, max_depth):
            print('i', i)
            if i > 0:
                model = concatenate([model,model_in])

            net = Dense(units=n_nodes, activation='sigmoid')(model_in)
            net = Dense(units=scaled_output.shape[1], 
                    activation='sigmoid')(net)
            net2 = Lambda(sum_layer, name='sum_layer')(net)
            net = concatenate([net, net2])

            model = Model(model_in, net)
            #model = Model([model, weights_tensor], net)
            model.summary()

            model.compile(loss=cl,
                    #loss='mse', 
                    optimizer='adam', 
                    metrics=['mae', 'acc']) #mean abs error, accuracy
            model.fit(
                    scaled_input, 
                    #[scaled_input, weights], 
                    all_scaled_output, 
                    epochs=n_epochs, verbose=2, 
                    callbacks=[mc,es]
                    )

            #'''
            model = load_model(file_name, 
                    custom_objects={'custom_loss': custom_loss1(weights)})
            if model.evaluate(scaled_input, all_scaled_output, 
                    verbose=0)[0] < best_error:
                model.save('best_ever_model')
                best_error = model.evaluate(scaled_input, all_scaled_output, 
                        verbose=0)[0]
                print('best model was achieved on layer %d' % i)
                print('its error was: {}'.format(best_error))
                #_, accuracy = model.evaluate(scaled_input, scaled_output)
                #print('accuracy: {}'.format(accuracy * 100))
                #print()
            print()
            #'''
            #end_training = np.loadtxt('end_file', dtype=int)
            #if end_training == 1:
                #break
            #if time.time()-start_time >= 518400: #5 days
            if time.time()-start_time >= 302400: #3.5 days
                print('\n*** Time limit reached, ending training ***')
                break
            model = load_model(file_name, 
                    custom_objects={'custom_loss': custom_loss1(weights)})
            model.trainable = False
            model = model(model_in)




        #model = load_model('best_ever_model')
        model = load_model('best_ever_model', 
                custom_objects={'custom_loss': custom_loss1(weights)})
        print('train', model.evaluate(scaled_input, all_scaled_output, 
                verbose=2))
        train_prediction_scaled = model.predict(scaled_input)
        train_prediction_scaled = train_prediction_scaled[:,:-1]
        train_prediction = Network.get_unscaled_values(
                train_prediction_scaled, 
                self.scale_output_max, self.scale_output_min, 
                method=output_scale_method)
        pred_sumE = np.sum(train_prediction, axis=1).reshape(1,-1)
        print('pred:', pred_sumE)
        #print(train_output)
        actual_sumE = np.sum(train_output, axis=1).reshape(1,-1)
        print('actual', actual_sumE)
        sum_Es = np.concatenate((actual_sumE, pred_sumE))
        np.savetxt('actual-pred-sumE.txt', sum_Es.T)
        train_mae, train_rms = Binner.get_error(actual_sumE.flatten(), 
                    pred_sumE.flatten())
        print('sumE errors MAE: {} RMS: {}'.format(train_mae, train_rms)) 


        atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
        atom_pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                atom_pairs.append('{}_{}'.format(atom_names[i], 
                    atom_names[j]))

        input_header = 'input'
        if len(scaled_input[0]) == _NC2:
            input_header = ['input_' + s for s in atom_pairs]
            input_header = ','.join(input_header)
        output_header = 'output'
        if len(scaled_output[0]) == _NC2:
            output_header = ['output_' + s for s in atom_pairs]
            output_header = ','.join(output_header)
        prediction_header = 'prediction'
        if len(train_prediction[0]) == _NC2:
            prediction_header = ['prediction_' + s for s in atom_pairs]
            prediction_header = ','.join(prediction_header)
        header = input_header + ',' + output_header + ',' + prediction_header
        #header = 'input,output,prediction'
 
        Writer.write_csv([train_input, train_output, train_prediction], 
                'trainset_inp_out_pred', header)


        test_input = np.take(input, molecule.test, axis=0)
        test_output = np.take(output, molecule.test, axis=0)

        weights_test = np.ones((_NC2+1))
        weights_test[-1] = sumE_weight 
        model = load_model('best_ever_model', 
                custom_objects={'custom_loss': custom_loss1(weights_test)})

        scaled_input_test, scale_input_max, scale_input_min = \
                Network.get_scaled_values(test_input, np.amax(train_input), 
                np.amin(train_input), method=input_scale_method)
        scaled_output_test, scale_output_max, scale_output_min = \
                Network.get_scaled_values(test_output, 
                np.amax(np.absolute(train_output)), 
                np.amin(np.absolute(train_output)), 
                method=output_scale_method)
        sum_output_test = np.sum(scaled_output_test, axis=1).reshape(-1,1)
        all_scaled_output_test = np.concatenate((scaled_output_test, 
                sum_output_test), axis=1)
        print('test', model.evaluate(scaled_input_test, 
            all_scaled_output_test, verbose=2))
        test_prediction_scaled = model.predict(scaled_input_test)
        test_prediction_scaled = test_prediction_scaled[:,:-1]
        test_prediction = Network.get_unscaled_values(test_prediction_scaled, 
                self.scale_output_max, self.scale_output_min, 
                method=output_scale_method)

        train_mae, train_rms = Binner.get_error(train_output.flatten(), 
                    train_prediction.flatten())

        test_mae, test_rms = Binner.get_error(test_output.flatten(), 
                    test_prediction.flatten())

        print('\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

        scurves = True
        if scurves:
            train_bin_edges, train_hist = Binner.get_scurve(
                    train_output.flatten(), 
                    train_prediction.flatten(), 
                    'train-hist.txt')

            test_bin_edges, test_hist = Binner.get_scurve(
                    test_output.flatten(), #actual
                    test_prediction.flatten(), #prediction
                    'test-hist.txt')
           
            Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                    [train_hist, test_hist], ['train', 'test'], 
                    'Error', '% of points below error', 's-curves.png')



    def get_variable_depth_model(self, molecule, nodes, input, output):
        start_time = time.time()
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        input_scale_method = 'A'
        output_scale_method = 'B'

        ###sort input and output by equivalent atoms
        equiv_atoms = False
        print('equiv_atoms', equiv_atoms)
        all_resorted_list = []
        if len(input[0]) == _NC2 and equiv_atoms:
            A = Molecule.find_bonded_atoms(molecule.atoms, 
                    molecule.coords[0])
            indices, pairs_dict = Molecule.find_equivalent_atoms(
                    molecule.atoms, A)
            all_sorted_list, all_resorted_list = Molecule.get_sorted_pairs(
                    input, pairs_dict)
            #print('\norig input', input)
            input = np.take_along_axis(input, 
                    all_sorted_list, axis=1)
            #print('\nnew input', input)
            output = np.take_along_axis(output, 
                    all_sorted_list, axis=1)

            print(molecule.atoms)
            print(A)
            print(pairs_dict)
            #print('all_sorted_list', all_sorted_list)
            #print('all_resorted_list', all_resorted_list)

        train_input = input
        train_output = output

        train = True #False
        if train:
            train_input = np.take(input, molecule.train, axis=0)
            #print('\ntrain_input', train_input)
            train_output = np.take(output, molecule.train, axis=0)
            #print('\ntrain_output', train_output)


        with_atom_F = False
        if with_atom_F:
            train_input = train_input.reshape(-1,_NC2)
            train_output = train_output.reshape(-1,_NC2)

        '''
        ###save max and min values for decomp forces to file
        #print(train_output.shape)
        max_NRFs = np.amax(train_input, axis=0)
        np.savetxt('max_inputs.txt', max_NRFs)
        min_NRFs = np.amin(train_input, axis=0)
        np.savetxt('min_inputs.txt', min_NRFs)

        max_Fs = np.amax(train_output, axis=0)
        np.savetxt('max_outputs.txt', max_Fs)
        min_Fs = np.amin(train_output, axis=0)
        np.savetxt('min_outputs.txt', min_Fs)
        #sys.exit()
        '''

        scaled_input, self.scale_input_max, self.scale_input_min = \
                Network.get_scaled_values(train_input, np.amax(train_input), 
                np.amin(train_input), method=input_scale_method)
        print('INPUT: \nscale_max: {}\nscale_min: {}'\
                '\nnstructures: {}\n'.format(
                self.scale_input_max, self.scale_input_min, len(train_input)))
        scaled_output, self.scale_output_max, self.scale_output_min = \
                Network.get_scaled_values(train_output, 
                np.amax(np.absolute(train_output)), 
                np.amin(np.absolute(train_output)), 
                method=output_scale_method)
        print('OUTPUT: \nscale_max: {}\nscale_min: {}'\
                '\nnstructures: {}\n'.format(
                self.scale_output_max, self.scale_output_min, 
                len(train_output)))

        '''
        print('\npairwise network')
        train_prediction_scaled = Network.pairwise_NN(scaled_input, 
                scaled_output, nodes, _NC2)
        '''

        #'''
        print('input shape: {}'.format(scaled_input.shape))
        print('output shape: {}'.format(scaled_output.shape))
        max_depth = 1 #6
        file_name = 'best_model'
        #file_name = 'best_ever_model'
        mc = ModelCheckpoint(file_name, monitor='loss', 
                mode='min', save_best_only=True)
        es = EarlyStopping(monitor='loss', patience=500)
        model_in = Input(shape=(scaled_input.shape[1],))
        model = model_in
        best_error = 1000
        n_nodes = nodes #1000
        n_epochs = 100000 #100000
        print('max depth: {}\nn_nodes: {}\nn_epochs: {}'.format(
                max_depth, n_nodes, n_epochs))

        train_net = True
        if train_net:
            for i in range(0, max_depth):
                if i > 0:
                    model = concatenate([model,model_in])
                net = Dense(units=n_nodes, activation='sigmoid')(model)
                #net = Dense(units=scaled_input.shape[1], 
                        #activation='sigmoid')(net)
                net = Dense(units=scaled_output.shape[1], 
                        activation='sigmoid')(net)
                model = Model(model_in, net)
                model.compile(loss='mse', optimizer='adam', 
                        metrics=['mae', 'acc']) #mean abs error, accuracy
                model.summary()
                model.fit(scaled_input, scaled_output, epochs=n_epochs, 
                        verbose=2, callbacks=[mc,es]
                        )

                #'''
                model = load_model(file_name)
                if model.evaluate(scaled_input, scaled_output, 
                        verbose=0)[0] < best_error:
                    model.save('best_ever_model')
                    best_error = model.evaluate(scaled_input, scaled_output, 
                            verbose=0)[0]
                    print('best model was achieved on layer %d' % i)
                    print('its error was: {}'.format(best_error))
                    #_, accuracy = model.evaluate(scaled_input, scaled_output)
                    #print('accuracy: {}'.format(accuracy * 100))
                    #print()
                    print()
                #'''
                #end_training = np.loadtxt('end_file', dtype=int)
                #if end_training == 1:
                    #break
                #if time.time()-start_time >= 518400: #5 days
                if time.time()-start_time >= 302400: #3.5 days
                    print('\n*** Time limit reached, ending training ***')
                    break
                model = load_model(file_name)
                model.trainable = False
                model = model(model_in)
            self.model = model

        model = load_model('best_ever_model')
        print('train', model.evaluate(scaled_input, scaled_output, verbose=2))
        train_prediction_scaled = model.predict(scaled_input)

        #'''

        train_prediction = Network.get_unscaled_values(
                train_prediction_scaled, 
                self.scale_output_max, self.scale_output_min, 
                method=output_scale_method)

        atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
        atom_pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                atom_pairs.append('{}_{}'.format(atom_names[i], 
                    atom_names[j]))

        input_header = 'input'
        if len(scaled_input[0]) == _NC2:
            input_header = ['input_' + s for s in atom_pairs]
            input_header = ','.join(input_header)
        output_header = 'output'
        if len(scaled_output[0]) == _NC2:
            output_header = ['output_' + s for s in atom_pairs]
            output_header = ','.join(output_header)
        prediction_header = 'prediction'
        if len(train_prediction[0]) == _NC2:
            prediction_header = ['prediction_' + s for s in atom_pairs]
            prediction_header = ','.join(prediction_header)
        header = input_header + ',' + output_header + ',' + prediction_header
        #header = 'input,output,prediction'


        if len(input[0]) == _NC2 and equiv_atoms:

            #input_resorted = np.take_along_axis(input, 
                    #all_resorted_list, axis=1)
            #print('\ninput_resorted', train_input)

            train_resorted_list = np.take(all_resorted_list, 
                    molecule.train, axis=0)
            #print('\ntrained new input', train_input)
            train_input = np.take_along_axis(train_input, 
                    train_resorted_list, axis=1)
            #print('\ntrained orig input', train_input)
            train_output = np.take_along_axis(train_output, 
                    train_resorted_list, axis=1)
            train_prediction = np.take_along_axis(train_prediction, 
                    train_resorted_list, axis=1)
            #print('\ntrained orig pred', train_prediction)
        
        Writer.write_csv([train_input, train_output, train_prediction], 
                'trainset_inp_out_pred', header)

        test_prediction = Network.get_validation(
                molecule, input, output, header, atom_names, 
                all_resorted_list, equiv_atoms, with_atom_F, train_input, 
                train_output, train_prediction)

        return train_prediction, test_prediction


    def get_validation(molecule, input, output, header, atom_names, 
            all_resorted_list, equiv_atoms, with_atom_F, train_input, 
            train_output, train_prediction):
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        input_scale_method = 'A'
        output_scale_method = 'B'
        test_input = np.take(input, molecule.test, axis=0)
        test_output = np.take(output, molecule.test, axis=0)

        if with_atom_F:
            test_input = test_input.reshape(-1,_NC2)
            test_output = test_output.reshape(-1,_NC2)

        scaled_input_test, scale_input_max, scale_input_min = \
                Network.get_scaled_values(test_input, np.amax(train_input), 
                np.amin(train_input), method=input_scale_method)
        scaled_output_test, scale_output_max, scale_output_min = \
                Network.get_scaled_values(test_output, 
                np.amax(np.absolute(train_output)), 
                np.amin(np.absolute(train_output)), 
                method=output_scale_method)


        model = load_model('best_ever_model')

        '''
        print('\npairwise test')
        new_scaled_input_test = []
        new_scaled_output_test = []
        for i in range(_NC2):
            new_scaled_input_test.append(scaled_input_test.T[i])
            new_scaled_output_test.append(scaled_output_test.T[i])
        print('test', model.evaluate(new_scaled_input_test, 
            new_scaled_output_test, verbose=2))
        test_prediction_scaled = model.predict(new_scaled_input_test)
        test_prediction_scaled = np.array(
                test_prediction_scaled).reshape(-1,_NC2)
        '''

        #'''
        print('test', model.evaluate(scaled_input_test, scaled_output_test, 
            verbose=2))
        test_prediction_scaled = model.predict(scaled_input_test)
        #'''


        test_prediction = Network.get_unscaled_values(test_prediction_scaled, 
                scale_output_max, scale_output_min, 
                method=output_scale_method)


        if len(input[0]) == _NC2 and equiv_atoms:
            test_resorted_list = np.take(all_resorted_list, 
                    molecule.test, axis=0)
            test_input = np.take_along_axis(test_input, test_resorted_list,
                    axis=1)
            test_output = np.take_along_axis(test_output, test_resorted_list,
                    axis=1)
            test_prediction = np.take_along_axis(test_prediction, 
                    test_resorted_list, axis=1)

        Writer.write_csv([test_input, test_output, test_prediction], 
                'testset_inp_out_pred', header)

        get_recomp_charges = False
        ##output charges! ##for Lejun
        if len(test_prediction[0]) == _NC2 and get_recomp_charges:
            coords_test = np.take(molecule.coords, molecule.test, axis=0)
            charges_test = np.take(molecule.charges, molecule.test, axis=0)
            test_recomp_charges = Converter.get_recomposed_charges(
                    coords_test, test_prediction, n_atoms, _NC2)
            output_header = ['output_' + s for s in atom_names]
            prediction_header = ['prediction_' + s for s in atom_names]
            Writer.write_csv([charges_test, test_recomp_charges], 
                    'testset_ESP_charges', 
                    ','.join(output_header+prediction_header))

        get_recomp_F = False
        #output forces!
        if len(test_prediction[0]) == _NC2 and get_recomp_F:

            coords_test = np.take(molecule.coords, molecule.test, axis=0)
            #coords = np.take(molecule.coords, molecule.train, axis=0)
            '''
            forces_test = np.take(molecule.forces, molecule.test, axis=0)
            forces = np.take(molecule.forces, molecule.train, axis=0)
            '''
            forces_test = np.take(molecule.forces, molecule.test, axis=0)
            #charges = np.take(molecule.charges, molecule.train, axis=0)
            #test_recomp_forces = Network.get_recomposed_forces(coords_test, 
                    #test_prediction, n_atoms, _NC2)
            #recomp_forces = Network.get_recomposed_forces(coords, 
                    #train_prediction, n_atoms, _NC2)
            test_recomp_forces = Network.get_recomposed_forces(
                    coords_test, test_prediction, n_atoms, _NC2)
            #recomp_charges = Converter.get_recomposed_charges(coords, 
                    #train_prediction, n_atoms, _NC2)

            output_header = ['output_' + s + i for s in atom_names
                    for i in ['x', 'y', 'z']]
            prediction_header = ['prediction_' + s + i for s in atom_names
                    for i in ['x', 'y', 'z']]
            Writer.write_csv([forces_test.reshape(-1,n_atoms*3), 
                    test_recomp_forces.reshape(-1,n_atoms*3)], 
                    'testset_atomic_forces', 
                    ','.join(output_header+prediction_header))
            #Writer.write_xyz(test_recomp_forces, molecule.atoms, 
                #'testset-predicted-forces.xyz', 'w')
            #Writer.write_xyz(recomp_forces, molecule.atoms, 
                #'trainset-predicted-forces.xyz', 'w')
            '''
            Writer.write_xyz(coords_test, molecule.atoms, 
                'testset-coords.xyz', 'w')
            #Writer.write_xyz(forces_test, molecule.atoms, 
                #'testset-forces.xyz', 'w')
            Writer.write_xyz(coords, molecule.atoms, 
                'trainset-coords.xyz', 'w')
            #Writer.write_xyz(forces, molecule.atoms, 
                #'trainset-forces.xyz', 'w')    
            '''

        train_mae, train_rms = Binner.get_error(train_output.flatten(), 
                    train_prediction.flatten())

        test_mae, test_rms = Binner.get_error(test_output.flatten(), 
                    test_prediction.flatten())

        print('\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

        scurves = True
        if scurves:
            train_bin_edges, train_hist = Binner.get_scurve(
                    train_output.flatten(), 
                    train_prediction.flatten(), 
                    'train-hist.txt')

            test_bin_edges, test_hist = Binner.get_scurve(
                    test_output.flatten(), #actual
                    test_prediction.flatten(), #prediction
                    'test-hist.txt')
           
            Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                    [train_hist, test_hist], ['train', 'test'], 
                    'Error', '% of points below error', 's-curves.png')

        return test_prediction


    def get_scaled_values_old(values):
        '''normalise values for NN'''
        #print(np.amax(values), np.amin(values))
        if np.amax(values) > 0 and np.amin(values) <= 0:
            #make zero = 0.5
            scale_max = np.amax(np.absolute(values))
            scale_min = np.amin(np.absolute(values))
            scale_max = 2 * np.amax(np.absolute(values))
            scale_min = 2 * np.amin(np.absolute(values))
            scaled_values = values / scale_max + 0.5
        else:
            #make max = 1 and min = 0
            scale_max = np.amax(values)
            scale_min = np.amin(values)
            scaled_values = values / np.amax(values)
            #scaled_values = (values - np.amin(values)) / \
                    #(np.amax(values) - np.amin(values))
            #if np.amax(values) < 0 and np.amin(values) < 0:
                #scaled_values = -scaled_values
                #print('inp3')
            #else:
                #print('inp2')

        print('scale_max: {}\nscale_min: {}\nnstructures: {}\n'.format(
                scale_max, scale_min, len(values)))

        return scaled_values, scale_max, scale_min


    def get_scaled_values(values, scale_max, scale_min, method):
        '''normalise values for NN'''
        #print(np.amax(values), np.amin(values))

        if method == 'A':
            scaled_values = values / scale_max
        if method == 'B':
            scaled_values = values / (2 * scale_max) + 0.5
        if method == 'C':
            scaled_values = (values - scale_min) / (scale_max - scale_min)
        if method == 'D': #if all values are -tive
            scaled_values = values / -scale_max

        return scaled_values, scale_max, scale_min


    def get_unscaled_values(scaled_values, scale_max, scale_min, method):
        '''normalise values for NN'''
        #print(np.amax(values), np.amin(values))

        if method == 'A':
            values = scaled_values * scale_max
        if method == 'B':
            values = (scaled_values - 0.5) * \
                    (2 * np.absolute(scale_max))
        if method == 'C':
            values = (scaled_values * (scale_max - scale_min)) + scale_min
        if method == 'D': #if all values are -tive
            values = scaled_values * -scale_max

        return values



    def get_NRF_input(coords, atoms, n_atoms, _NC2):
        mat_NRF = np.zeros((len(coords),_NC2))
        for s in range(len(coords)):
            _N = -1
            for i in range(n_atoms):
                zi = atoms[i]
                for j in range(i):
                    _N += 1
                    zj = atoms[j]
                    r = Converter.get_r(coords[s][i], coords[s][j])
                    if i != j:
                        mat_NRF[s,_N] = Converter.get_NRF(zi, zj, r)
        return mat_NRF

    def get_recomposed_forces(all_coords, all_prediction, n_atoms, _NC2):
        '''Take pairwise decomposed forces and convert them back into 
        system forces.'''
        all_recomp_forces = []
        for coords, prediction in zip(all_coords, all_prediction):
            #print(coords.shape)
            #print(prediction.shape)
            rij = np.zeros((3, n_atoms, n_atoms))
                #interatomic vectors from col to row
            eij = np.zeros((3, n_atoms, n_atoms))
                #normalised interatomic vectors
            q_list = []
            for i in range(1,n_atoms):
                for j in range(i):
                    rij[:,i,j] = (coords[i,:] - coords[j,:])
                    rij[:,j,i] = -rij[:,i,j]
                    eij[:,i,j] = rij[:,i,j] / np.reshape(
                            np.linalg.norm(rij[:,i,j], axis=0), (-1,1))
                    eij[:,j,i] = -eij[:,i,j]
                    q_list.append([i,j])
            _T = np.zeros((3*n_atoms, _NC2))
            for i in range(int(_T.shape[0]/3)):
                for k in range(len(q_list)):
                    if q_list[k][0] == i:
                        _T[range(i*3, (i+1)*3), k] = \
                                eij[:,q_list[k][0],q_list[k][1]]
                    if q_list[k][1] == i:
                        _T[range(i*3, (i+1)*3), k] = \
                                eij[:,q_list[k][1],q_list[k][0]]
            recomp_forces = np.zeros((n_atoms, 3))
            recomp_forces = np.reshape(np.dot(_T, prediction.flatten()), 
                    (-1,3))
            all_recomp_forces.append(recomp_forces)
        #print(np.array(all_recomp_forces).shape)
        return np.array(all_recomp_forces).reshape(-1,n_atoms,3)

    def get_scaling_factors(all_coords, all_next_coords, 
            all_prediction, n_atoms, _NC2):
        mat_q_new = []
        delta_q = []
        for coords, next_coords, prediction in zip(all_coords, 
                all_next_coords, all_predicition):
            new_rij = np.zeros((3, n_atoms, n_atoms))
            new_eij = np.zeros((3, n_atoms, n_atoms))
            for i in range(n_atoms):
                for j in range(i):
                    new_rij[:,i,j] = (new_coords[i,:] - coords[j,:])
                    new_rij[:,j,i] = -new_rij[:,i,j]
                    new_eij[:,i,j] = new_rij[:i,j] / np.reshape(
                            np.linalg.norm(new_rij[:,i,j], axis=0), (-1,1))

            new_eij2 = new_eij.reshape(n_atoms*3,_NC2)
            forces = Network.get_recomposed_forces(all_coords, 
                    [prediction], n_atoms, _NC2)
            forces2 = forces.reshape(n_atoms*3)
            q_new = np.matmul(np.linalg.pinv(new_eij2), forces2)
            mat_q_new.append(q_new)

            _N = -1
            for i in range(n_atoms):
                for j in range(i):
                    _N += 1
                    dq = (q_new[_N] - predicition[_N]) / np.reshape(
                            np.linalg.norm(new_rij[:,i,j], axis=0), (-1,1))

        mat_q_new = np.reshape(np.vstack(mat_q_new), (-1,_NC2))



    def get_scaled(a, b, x_min, x_max, x):
        x_norm = (b - a) * (x - x_min) / (x_max - x_min) + a
        return x_norm

    def get_unscaled(a, b, x_min, x_max, x_norm):
        x = (x_norm - a) / (b - a) * (x_max - x_min) + x_min
        return x


    def internal_decomp_NN(self, molecule, nodes, input, output):
        '''Force decomposition is perfomed within the ANN'''

        # setup network inputs
        model_in = Input(shape=(inputs.shape[1],))
        input_layers = []
        input_layers.append(model_in)
        proj_mats = []
        for i in range(0,values_mat.shape[1]):
            proj_mats.append(Input(shape=(NRE_inputs.shape[1],)))
            input_layers.append(proj_mats[i])
        print(len(proj_mats))

        #setup network
        model = Dense(units=1000,activation="relu")(model_in)
        model = Dense(units = NRE_inputs.shape[1],activation="linear")(model)
        nets = []
        for i in range(0,len(proj_mats)):
            my_name = "multiply%d" %i
            nets.append(Multiply(name=my_name)([model,proj_mats[i]]))
            my_name = "sum%d" %i
            nets[i] = Lambda(sum_layer,name=my_name)(nets[i])
        output = concatenate(nets)
        model = Model(input_layers,output)
        model.summary()
        net_inputs = []
        NRE_inputs = NRE_inputs/np.amax(NRE_inputs)
        net_inputs.append(NRE_inputs)
        for i in range(0,values_mat.shape[1]):
            net_inputs.append(values_mat[:,i,:])

        #train network
        file_name = "best_model"
        mc = ModelCheckpoint(file_name, monitor='loss', mode='min', 
                save_best_only=True)
        es = EarlyStopping(monitor='loss',patience=500)
        model.compile(loss='mse',optimizer='adam')
        forces = np.reshape(forces,(-1,len(proj_mats)))
        model.fit(net_inputs,forces,epochs=100000,verbose = 2, 
                callbacks=[mc,es])
        prediction=model.predict(net_inputs)
        np.savetxt('train_prediction',prediction)


    def run_NVE(network, molecule, timestep, nsteps, qAB_factor=0):
        #mm = MM() #initiate MM class

        saved_steps = nsteps #15000
        print('\nsave {} steps from MD'.format(saved_steps))

        scale_coords = False
        if scale_coords:
            print('\ncoordinates being scaled by min and max NRFs')
            scale_NRF_all = np.loadtxt('max_NRFs.txt')
            scale_NRF_min_all = np.loadtxt('min_NRFs.txt')

        scale_forces = False
        if scale_forces:
            print('\nscaling decomposed forces to max values for each pair')
            max_Fs = np.loadtxt('max_decompFs.txt')
            min_Fs = np.loadtxt('min_decompFs.txt')


        conservation = True
        conserve_steps = nsteps/100 #15000 #750 #3000 #150 #15000  #1875
        if conservation:
            dr = 0.001 #0.001
            print('\nforces are scaled to ensure energy conservation, '\
                    'dr = {}, conserve {} steps'.format(dr, conserve_steps))


        NRF_scale_method = 'A'
        print('\nNRF scaling method: {}'.format(NRF_scale_method))

        sys.stdout.flush()

        mm = Molecule()
        mm.atoms = molecule.atoms
        mm.coords = []
        mm.forces = []
        mm.energies = []
        coords_init = molecule.coords[0]
        '''
        first_frame = -1
        print('first frame {}'.format(first_frame))
        coords_init = molecule.coords[first_frame]
        '''
        #sorted_i = np.argsort(molecule.energies.flatten())
        #print('\nfirst structure is index {}'.format(sorted_i[-1]))
        #coords_init = molecule.coords[sorted_i[-1]] #start with highest E
        atoms = molecule.atoms
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)

        '''
        scale_NRF = network.scale_input_max #29.10017376084781# 
        scale_NRF_min = network.scale_input_min # 0.03589677731052864 #
        scale_NRF_all = 0 #network.scale_NRF_all #
        scale_NRF_min_all = 0 #network.scale_NRF_min_all #
        scale_F = network.scale_output_max # 
        scale_F_min = network.scale_output_min #0 #
        #scale_F_all = network.scale_F_all 
        #scale_F_min_all = network.scale_F_min_all 
        #print(scale_NRF, scale_NRF_min, scale_F)

        print('scale_NRF: {}\nscale_NRF_min: {}\nscale_F: {}\n'\
                'scale_F_min: {}\n'.format(
                scale_NRF, scale_NRF_min, scale_F, scale_F_min))
        '''

        '''updated (min-max scaling)
        scale_NRF = 17495.630534482527
        scale_NRF_min = 11.648288117257646
        scale_F = 214.93048169383425/2
        scale_F_min = 0
        '''

        #'''updated3 (orig scaling) and updated2 (min-max scaling)
        scale_NRF = 19911.051275945494 
        scale_NRF_min = 11.648288117257646 
        scale_F = 105.7233972832097
        scale_F_min = 0
        #'''

        '''
        #Ismaeel's aspirin method
        scale_NRF = 13036.551114036025
        scale_NRF_min = 0
        scale_F = 228.19031799443/2
        scale_F_min = 0
        '''

        #mm.coords.append(coords_init)
        masses = np.zeros((n_atoms,3))
        for i in range(n_atoms):
            masses[i,:] = Converter._ZM[atoms[i]]
        #model = network.model
        model = load_model('best_ever_model')
        open('nn-coords.xyz', 'w').close()
        open('nn-forces.xyz', 'w').close()

        open('nn-NRF.txt', 'w')
        open('nn-decomp-force.txt', 'w')
        open('nn-forces.txt', 'w')
        f1 = open('nn-NRF.txt', 'ab')
        f2 = open('nn-decomp-force.txt', 'ab')
        #f3 = open('nn-E.txt', 'ab')
        f4 = open('nn-forces.txt', 'ab')
        open('nn-E.txt', 'w').close()
        open('nn-KE.txt', 'w').close()
        open('nn-T.txt', 'w').close()
        coords_prev = coords_init
        #coords_prev = molecule.coords[first_frame-1]
        #coords_prev = Converter.translate_coords(
                #coords_init, atoms)
        coords_current = coords_init
        _E_prev = 0
        temp = 2
        dt = 0.5
        equiv_atoms = False
        print('equiv_atoms', equiv_atoms)

        if equiv_atoms:
            A = Molecule.find_bonded_atoms(molecule.atoms, 
                    molecule.coords[0])
            indices, pairs_dict = Molecule.find_equivalent_atoms(
                    molecule.atoms, A)

        for i in range(nsteps):
            sys.stdout.flush()
            #print('\t', i)
            #print('NVE coords', coords_current.shape, coords_current)

            #translate coords back to center
            coords_current = Converter.translate_coords(
                    coords_current, atoms)

            mat_NRF = Network.get_NRF_input([coords_current], atoms, 
                    n_atoms, _NC2)

            if equiv_atoms:
                all_sorted_list, all_resorted_list = \
                        Molecule.get_sorted_pairs(
                        mat_NRF, pairs_dict)
                mat_NRF = np.take_along_axis(mat_NRF, 
                        all_sorted_list, axis=1)

            #mat_NRF_scaled = mat_NRF / scale_NRF
            mat_NRF_scaled, max_, min_ = \
                    Network.get_scaled_values(mat_NRF, scale_NRF, 
                    scale_NRF_min, method=NRF_scale_method)
            
            #scale NRF to not go beyond max and min
            if scale_coords:
                #if np.any(np.greater(mat_NRF, scale_NRF)) or \
                        #np.any(np.less(mat_NRF, scale_NRF_min)):
                if np.any(np.greater(mat_NRF, scale_NRF_all)) or \
                        np.any(np.less(mat_NRF, scale_NRF_min_all)):
                    print(i)
                    #print(mat_NRF_scaled)
                    #print(mat_NRF)
                    #print()
                    #rs = Converter.get_r_from_NRF(mat_NRF[0], atoms)
                    #print('\n', rs)
                    #coords_current = Converter.get_coords_from_NRF(
                            #mat_NRF[0], molecule.atoms, 
                            #coords_current, [scale_NRF]*_NC2, 
                            #[scale_NRF_min]*_NC2)
                    coords_current = Converter.get_coords_from_NRF(
                            mat_NRF[0], molecule.atoms, 
                            coords_current, scale_NRF_all, scale_NRF_min_all)
                    mat_NRF = Network.get_NRF_input([coords_current], 
                            atoms, n_atoms, _NC2)
                    #rs = Converter.get_r_from_NRF(mat_NRF[0], atoms)
                    #print('\n', rs)
                    mat_NRF_scaled = mat_NRF / scale_NRF
                    #mat_NRF_scaled = Network.get_scaled(0, 1, scale_NRF_min, 
                            #scale_NRF, mat_NRF)
                    #print(mat_NRF_scaled)
                    #sys.exit()
                    sys.stdout.flush()

            prediction_scaled = model.predict(mat_NRF_scaled)

            prediction = Network.get_unscaled_values(prediction_scaled, 
                    scale_F, scale_F_min, method='B')



            #'''
            if scale_forces:
                print(prediction)
                #print()
                #print(max_Fs)
                hard_coded_F = 0
                #bonded_pairs = [0, 1, 3, 6, 12, 16, 22, 29, 42]
                for n in range(1): #range(_NC2):
                    prediction[0][n] = hard_coded_F
                '''
                #for n in bonded_pairs:
                    #print(abs(prediction[0][n]), max_Fs[n])
                    if prediction[0][n] < min_Fs[n]:
                        prediction[0][n] = min_Fs[n]

                    if min_Fs[n] < -hard_coded_F:
                        if prediction[0][n] < -hard_coded_F:
                            prediction[0][n] = -hard_coded_F
                    if prediction[0][n] > max_Fs[n]:
                        prediction[0][n] = max_Fs[n]
                    if max_Fs[n] > hard_coded_F:
                        if prediction[0][n] > hard_coded_F:
                            prediction[0][n] = hard_coded_F
                '''
                #print()
                print(prediction)
                print()
                #sys.exit()
            #'''


            if equiv_atoms:
                #print(mat_NRF.shape, prediction.shape)
                mat_NRF = np.take_along_axis(mat_NRF, 
                        all_resorted_list, axis=1)
                prediction = np.take_along_axis(prediction, 
                        all_resorted_list, axis=1)


            if conservation and i%(nsteps/conserve_steps) == 0:
                print('conserve frame', i)
                prediction = Conservation.get_conservation(
                        coords_current, prediction, 
                        molecule.atoms, scale_NRF, scale_NRF_min, scale_F, 
                        model, molecule, dr, NRF_scale_method,
                        qAB_factor)


            recomp_forces = Network.get_recomposed_forces([coords_current], 
                    [prediction], n_atoms, _NC2)

            #recomp_forces[0][0] = [0,0,0]
            #recomp_forces[0][1] = [0,0,0]


            #'''
            scale_forces2 = False #True
            if scale_forces2:
                print(recomp_forces)
                recomp_forces = recomp_forces.reshape(1,-1)
                hard_coded_F = 0
                for i in [0,1,2,5,6,9]:
                    for j in range(3):
                        n = i*3+j
                        if recomp_forces[0][n] < -hard_coded_F:
                            recomp_forces[0][n] = -hard_coded_F
                        if recomp_forces[0][n] > hard_coded_F:
                            recomp_forces[0][n] = hard_coded_F
                recomp_forces = recomp_forces.reshape(1,-1,3)
                print(recomp_forces)
                print()
                #sys.exit()
            #'''



            #print('NVE recomp', recomp_forces.shape, recomp_forces)

            '''
            ##rotate forces
            #print('shape c', coords_current.shape)
            #print('shape f', recomp_forces.shape)
            mm.coords = mm.get_3D_array([coords_current])
            mm.forces = mm.get_3D_array([recomp_forces])
            mm.check_force_conservation()
            Converter.get_rotated_forces(mm)

            coords_current = mm.rotated_coords[0]
            #print('shape c', coords_current.shape)
            recomp_forces = mm.rotated_forces
            #print('shape f', recomp_forces.shape)
            '''

            #print(i, temp)
            coords_next, dE, v, current_T, _KE = \
                    MM.calculate_verlet_step(coords_current, 
                    coords_prev, recomp_forces[0], masses, timestep, dt, temp)
            #coords_next = MM.calculate_step(coords_current, coords_prev, 
                    #recomp_forces[0], timestep, masses, timestep)
                    #ismaeel's code
            _E = _E_prev - dE

            #if i%(nsteps/100) == 0 and temp < 1:
                #temp += 0.01
            if i%(nsteps/saved_steps) == 0:
                np.savetxt(f1, mat_NRF)
                np.savetxt(f2, prediction)
                np.savetxt(f4, recomp_forces.reshape(1,-1))
                open('nn-E.txt', 'a').write('{}\n'.format(_E))
                open('nn-KE.txt', 'a').write('{}\n'.format(_KE))
                open('nn-T.txt', 'a').write('{}\n'.format(current_T))
                Writer.write_xyz([coords_current], molecule.atoms, 
                    'nn-coords.xyz', 'a', i)
                Writer.write_xyz(recomp_forces, molecule.atoms, 
                    'nn-forces.xyz', 'a', i)
                Writer.write_xyz([v], molecule.atoms, 
                    'nn-velocities.xyz', 'a', i)
                mm.coords.append(coords_current)
                mm.forces.append(recomp_forces)
                mm.energies.append(_E)
                sys.stdout.flush()

            coords_prev = coords_current
            coords_current = coords_next
            _E_prev = _E
            #print()
        return mm


        
    def check_invariance(coords, forces):
        ''' Ensure that forces in each structure translationally and 
        rotationally sum to zero (energy is conserved) i.e. translations
        and rotations are invariant. '''
        #translations
        translations = []
        trans_sum = np.sum(forces,axis=0) #sum columns
        for x in range(3):
            x_trans_sum = np.abs(np.round(trans_sum[x], 0))
            translations.append(x_trans_sum)
        translations = np.array(translations)
        #rotations
        cm = np.average(coords, axis=0)
        i_rot_sum = np.zeros((3))
        for i in range(len(forces)):
            r = coords[i] - cm
            diff = np.cross(r, forces[i])
            i_rot_sum = np.add(i_rot_sum, diff)
        rotations = np.round(np.abs(i_rot_sum), 0)
        #check if invariant
        variance = False
        if np.all(rotations != 0):
            variance = True
        if np.all(translations != 0):
            variance = True
        return variance, translations, rotations



