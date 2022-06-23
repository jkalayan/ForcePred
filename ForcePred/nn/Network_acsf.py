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
import math

start_time = time.time()

#tf.compat.v1.disable_eager_execution()


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
    def __init__(self, atoms_flat, n_atoms, max_atomNRF, _NC2, **kwargs):
        super(CoordsTo_atomNRF, self).__init__()
        self.atoms_flat = atoms_flat
        self.n_atoms = n_atoms
        self.max_atomNRF = max_atomNRF
        self._NC2 = _NC2
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
        triangle_NRF = Triangle(self.n_atoms)(_NRF)
        #atomNRF = tf.reduce_sum(triangle_NRF, 1, keepdims=True) / 2
        atomNRF = tf.reduce_sum(triangle_NRF, 1) / 2
        atomNRF_scaled = atomNRF / self.max_atomNRF
        return atomNRF_scaled


class CoordsTo_acsf(Layer):
    def __init__(self, atoms_flat, n_atoms, max_atomNRF, _NC2, **kwargs):
        super(CoordsTo_atomNRF, self).__init__()
        self.atoms_flat = atoms_flat
        self.n_atoms = n_atoms
        self.max_atomNRF = max_atomNRF
        self._NC2 = _NC2
        self.au2kcalmola = tf.Variable(627.5095 * 0.529177)
        self.Rc = 5
        self.nu = 4
        self.pi = tf.constant(math.pi)

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
        qqr2 = ((self.atoms_flat) / (r ** 2))
        triangle_qqr2 = Triangle(self.n_atoms)(qqr2)
        triangle_r = Triangle(self.n_atoms)(r)

        Rs = 0
        less_mask = tf.math.less_equal(triangle_r, self.Rc) #Bools inside
        less_r = triangle_r * tf.cast(less_mask, dtype=tf.float32)
        fc_all = 0.5 * tf.math.cos(math.pi * triangle_r / self.Rc) + 0.5
        fc = fc_all * tf.cast(less_mask, dtype=tf.float32) #only keep less rc
        zeros = tf.zeros([fc.shape[0], fc.shape[1]])
        fc = tf.linalg.set_diag(fc, zeros) #make diags zero
        Gij = tf.math.exp(-nu * (triangle_qqr2 - Rs) ** 2) * fc
        Gi = tf.reduce_sum(Gij, 1)

        return Gi


class SumNRF(Layer):
    def __init__(self, _NC2, max_NRF, max_sumNRF, **kwargs):
        super(SumNRF, self).__init__()
        self._NC2 = _NC2
        self.max_NRF = max_NRF
        self.max_sumNRF = max_sumNRF

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, _NRF):
        sum_NRF = tf.reduce_sum(_NRF * self.max_NRF, 1, keepdims=True)
        sum_NRF_scaled = sum_NRF / self.max_sumNRF
        return sum_NRF_scaled


class SumNet(Layer):
    def __init__(self, max_FE, **kwargs):
        super(SumNet, self).__init__()
        self.max_FE = max_FE

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, net):
        decompFE_unscaled = (net - 0.5) * (2 * self.max_FE)
        E = tf.reduce_sum(decompFE_unscaled, 1, keepdims=True)
        return E


class ScaleE1(Layer):
    def __init__(self, max_E, **kwargs):
        super(ScaleE1, self).__init__()
        self.max_E = tf.Variable(max_E)
        #self.name = name

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, E):
        E_scaled = (E - 0.5) * (2 * self.max_E)
        #E3 = E * self.max_E
        return E_scaled


class ScaleE2(Layer):
    def __init__(self, prescale, **kwargs):
        super(ScaleE2, self).__init__()
        self.prescale = prescale
        #self.name = name

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, E):
        E3 = ((E - self.prescale[2]) / 
                (self.prescale[3] - self.prescale[2]) * 
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])
        return E3


class ScaleFE(Layer):
    def __init__(self, _NC2, max_FE, **kwargs):
        super(ScaleFE, self).__init__()
        self._NC2 = _NC2
        self.max_FE = max_FE

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
        self.n_atoms = n_atoms
        self.max_atomE = max_atomE

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


class sum_atomE(Layer):
    def __init__(self, n_atoms, _NC2, max_FE, prescale, **kwargs):
        super(sum_atomE, self).__init__()
        self.max_FE = max_FE
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        self.prescale = prescale
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
        #E_atoms_sum = tf.reduce_sum(E_atoms, 1)
        E_atoms_sum_postscale = ((E_atoms_sum - self.prescale[2]) / 
                (self.prescale[3] - self.prescale[2]) * 
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])

        return E_atoms_sum_postscale


class GetQ(Layer):
    def __init__(self, n_atoms, _NC2, atoms_flat, **kwargs):
        super(GetQ, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        self.atoms_flat = atoms_flat
        self.au2kcalmola = 627.5095 * 0.529177

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, coords_F_E):
        coords, F, E  = coords_F_E
        F_reshaped = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        FE = tf.concat([F_reshaped, E], axis=1)

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

        #_NRF = ((self.atoms_flat * self.au2kcalmola) / (r_flat ** 2))
        _NRF = ((self.atoms_flat) / (r_flat ** 2))

        #FOR ENERGY get energy 1/r_ij eij matrix
        recip_r = 1 / r_flat
        eij_E = tf.expand_dims(recip_r, 1)
        #eij_E = tf.expand_dims(_NRF, 1)

        #### FOR FORCES
        r = Triangle(self.n_atoms)(r_flat)
        r2 = tf.expand_dims(r, 3)
        eij_F2 = diff / r2
        eij_F = tf.where(tf.math.is_nan(eij_F2), tf.zeros_like(eij_F2), 
                eij_F2) #remove nans 

        #put eij_F in format (batch,3N,NC2), some NC2 will be zeros
        new_eij_F = []
        n_atoms = coords.shape.as_list()[1]
        for i in range(n_atoms):
            for x in range(3):
                atom_i = eij_F[:,i,:,x]
                s = []
                _N = -1
                count = 0
                a = [k for k in range(n_atoms) if k != i]
                for i2 in range(n_atoms):
                    for j2 in range(i2):
                        _N += 1
                        if i2 == i:
                            s.append(atom_i[:,a[count]])
                            count += 1
                        elif j2 == i:
                            s.append(atom_i[:,a[count]])
                            count += 1
                        else:
                            s.append(tf.zeros_like(atom_i[:,0]))
                s = tf.stack(s)
                s = tf.transpose(s)
                new_eij_F.append(s)

        eij_F = tf.stack(new_eij_F)
        eij_F = tf.transpose(eij_F, perm=[1,0,2])
        eij_FE = tf.concat([eij_F, eij_E], axis=1)

        inv_eij = tf.linalg.pinv(eij_FE)
        qs = tf.linalg.matmul(inv_eij, tf.transpose(FE))
        qs = tf.transpose(qs, perm=[1,0,2])
        qs = tf.linalg.diag_part(qs) #only choose diags
        qs = tf.transpose(qs)
        #qs = qs * recip_r #remove bias so sum q = E

        return qs 


class EijMatrix_test(Layer):
    def __init__(self, n_atoms, _NC2, prescale, atoms_flat, **kwargs):
        super(EijMatrix_test, self).__init__()
        #self.max_FE = tf.Variable(max_FE, name='3a')
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        self.prescale = prescale
        self.atoms_flat = atoms_flat
        self.au2kcalmola = 627.5095 * 0.529177

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


        #_NRF = ((self.atoms_flat * self.au2kcalmola) / (r_flat ** 2))
        _NRF = ((self.atoms_flat) / (r_flat ** 2))

        #get energy 1/r_ij eij matrix
        recip_r_flat = 1 / r_flat
        #ones = tf.ones_like(recip_r_flat)
        Q3 = Triangle(self.n_atoms)(recip_r_flat)
        #Q3 = Triangle(self.n_atoms)(_NRF)
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

        #E3 = ((E - self.prescale[2]) / 
                #(self.prescale[3] - self.prescale[2]) * 
                #(self.prescale[1] - self.prescale[0]) + self.prescale[0])

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

        return E #3


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
        #gradients = tf.compat.v1.gradients(E, coords, 
                #colocate_gradients_with_ops=True, 
                #unconnected_gradients='zero')
        gradients = tf.gradients(E, coords, 
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
        atoms_flat = tf.convert_to_tensor(atoms_flat, dtype=tf.float32) #_NC2

        training_model = False
        if training_model:
            #for training a model
            get_data = True
            load_first = True
            fit = True
            load_weights = False
            inp_out_pred = True

        if training_model == False:
            #for loading a model to use with openmm
            get_data = True #False #
            load_first = False
            fit = False
            load_weights = True
            inp_out_pred = False

        #get_data = True
        if get_data:
            split = 100 #500 #200 #2
            train = round(len(molecule.coords) / split, 3)
            print('\nget train and test sets, '\
                    'training set is {} points'.format(train))
            Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                    split) #get train and test sets
            #'''
            print('!!!use regularly spaced training')
            molecule.train = np.arange(2, len(molecule.coords), split).tolist() 
            molecule.test = [x for x in range(0, len(molecule.coords)) 
                    if x not in molecule.train]
            #'''

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
           # output_atomNRF = molecule.atomNRF.reshape(-1,n_atoms) #per atom NRFs

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
            max_sumNRF1 = np.max(np.abs(np.sum(train_input_NRF, axis=1)))
            max_E1 = np.max(np.abs(train_output_E))
            max_FE1 = np.max(np.abs(train_output_matFE))
            print('***!!!OVERRIDING MAX_FE1 WITH FE NOT MATFE!!!!')
            max_FE1 = np.max(np.abs(train_output_FE))
            #max_atomFE1 = np.max(np.abs(train_output_atomFE))
            #max_atomNRF1 = np.max(np.abs(train_output_atomNRF))
            train_output_matFE_scaled = train_output_matFE / (2 * max_FE1) + 0.5
            train_input_NRF_scaled = np.take(input_NRF, molecule.train, axis=0)\
                    / max_NRF1
            print(train_input_coords.shape, train_output_FE.shape)
            print('max_NRF: {}, max_FE: {}, max_E: {}'.format(max_NRF1, 
                    max_FE1, max_E1))
           # print('max_atomE: {} max_atomNRF: {}'.format(max_atomFE1, 
                #max_atomNRF1))
            print('max_sumNRF: {} '.format(max_sumNRF1))
            max_NRF = tf.constant(max_NRF1, dtype=tf.float32)
            max_sumNRF = tf.constant(max_sumNRF1, dtype=tf.float32)
            max_E = tf.constant(max_E1, dtype=tf.float32)
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


            '''
            print('permutations - sorting of equiv_atoms')
            A = Molecule.find_bonded_atoms(molecule.atoms, 
                    molecule.coords[0])
            indices, pairs_dict = Molecule.find_equivalent_atoms(
                    molecule.atoms, A)
            print('indices', indices)
            print('pairs_dict', pairs_dict)
            '''

            '''
            sorted_list = []
            for pair, list_N in pairs_dict.items():
                qs = []
                for _N in list_N:
                    qs.append(input_NRF[0][_N])
                sorted_ = tf.argsort(qs)
                sorted_ = tf.unstack(sorted_)
                for i in sorted_:
                    val = tf.gather(list_N, i) #tf.cast(list_N[i], tf.int32)
                    sorted_list.append(val)
            #resorted_list = tf.argsort(sorted_list) #double sort for orig
            sorted_NRF = tf.experimental.numpy.take_along_axis(input_NRF[0], 
                    sorted_list, axis=-1)
            #print(input_NRF[0])
            sess = tf.compat.v1.Session()
            #print(sess.run(sorted_NRF))
            indices = tf.argsort(input_NRF[:3])
            sorted_ = tf.gather(input_NRF[:3], indices, batch_dims=-1)
            print(input_NRF[:3])
            print(sess.run(indices))
            print(sess.run(sorted_))
            sorted_NRFs = []
            sorted_all_list_N = []
            for pair, list_N in pairs_dict.items():
                print(pair, list_N)
                x_NRFs = tf.gather(input_NRF[:3], list_N, batch_dims=-1)
                print(sess.run(x_NRFs))
                indices = tf.argsort(x_NRFs)
                print(sess.run(indices))
                x_sort = tf.gather(x_NRFs, indices, batch_dims=-1)
                sorted_NRFs.append(x_sort)
                print(sess.run(x_sort))


                #indices2 = tf.argsort(indices)
                #print(sess.run(indices2))
                print(sess.run(tf.shape(input_NRF[:3])))
                list_N_tile = tf.tile(list_N, [tf.shape(input_NRF[:3])[0]])
                list_N_tile = tf.reshape(list_N_tile, [tf.shape(input_NRF[:3])[0], -1])
                print(sess.run(list_N_tile))
                list_N2 = tf.gather(list_N_tile, indices, batch_dims=-1)
                print(sess.run(list_N2))
                sorted_all_list_N.append(list_N2)


            print(sess.run(sorted_NRFs))
            concat_sorted_NRFs = tf.concat(sorted_NRFs, 1)
            print(sess.run(concat_sorted_NRFs))


            concat_sorted_all_list_N = tf.concat(sorted_all_list_N, 1)
            print(sess.run(concat_sorted_all_list_N))
            resorted_all_list_N = tf.argsort(concat_sorted_all_list_N) #resort
            print(sess.run(resorted_all_list_N))
            resorted_NRFs = tf.gather(concat_sorted_NRFs, resorted_all_list_N, batch_dims=-1)
            print(sess.run(resorted_NRFs))
            print(input_NRF[:3])
            sys.exit()
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
            #prescale1 = [-167315.01917961572, -167288.834575252, 
                    #-260.631337453833, 284.98286516421035]

            prescael1 = [ -97086.67448670573, -97060.43620928681, 
                    -151.45141221653606, 162.38544000536768]
            #max_NRF1 = 12304.734510536122
            max_NRF1 = 9119.964965436488 #13010.961100355082 
            #max_FE1 = 211.23633766510105
            max_FE1 = 162.3854400053677 #284.98286516421035
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
        e_loss = [0, 0, 1, 1, 1, 1, 1, 1] #1
        e2_loss = [_NC2, 0, 1, 1, 1, 1, 1, 1] #1
        f_loss = [1, 0, 0, 0, 0, 0, 0, 0] #0
        grad_loss = [1000, 0, 10, 1000, 1000, 1000, 1000, 1000] #1000
        q_loss = [0, 1, 10, 1, 1, 1, 1, 1] #_NC2 
        scaleq_loss = [_NC2, 1, 10, 1, 1, 1, 1, 1] #_NC2 
        best_error_all = 1e20
        best_iteration = 0
        best_model = None
        for l in range(0, max_depth):
            print('l', l)
            l2 = str(l)
            print('E loss weight: {} \nrecompF loss weight: {} '\
                    '\ndE_dx loss weight: {} \nq loss weight: {} '\
                    '\nscaleq loss weight: {} \nE2 loss weight: {}'.format(
                    e_loss[l], f_loss[l], grad_loss[l], q_loss[l], 
                    scaleq_loss[l], e2_loss[l]))
            #if l > 0:
                #model = concatenate([model2[3], NRF_layer])
            coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
            NRF_layer = CoordsToNRF_test(atoms_flat, max_NRF, _NC2, 
                    name='NRF_layer')(coords_layer)
            #sumNRF_layer = SumNRF(_NC2, max_NRF, max_sumNRF, 
                    #name='sumNRF_layer')(NRF_layer)
            #atomNRF_layer = CoordsTo_atomNRF(atoms_flat, n_atoms, max_atomNRF, 
                    #_NC2, name='atomNRF_layer')(coords_layer)
            #sorted_layer = SortPairs(_NC2, name='sorted_layer')(
                    #[pairs_dict, NRF_layer])

            '''
            net_layers = []
            #for x in range(n_atoms):
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
            '''

            #resorted_layer = ResortPairs(_NC2, name='resorted_layer')(
                    #[pairs_dict, net_layer2, NRF_layer])

            #scale_atomEx = Scale_atomE(n_atoms, max_atomFE, 
                    #name='scale_atomE')(net_layer2)
            #sum_atomEx = sum_atomE(n_atoms, _NC2, max_FE, prescale, 
                    #name='sum_atomE')(scale_atomEx)
            #dE_dx = EnergyGradient(n_atoms, _NC2, 
                    #name='dE_dx')([sum_atomEx, coords_layer])

            #net_layer = Dense(units=_NC2, activation='swish', 
                    #name='net_layer')(NRF_layer)
            #net_layer1 = Dense(units=500, activation='swish', 
                    #name='net_layer1')(NRF_layer)
            net_layer2 = Dense(units=1000, activation='sigmoid', 
                    name='net_layer2')(NRF_layer)
            net_layer3 = Dense(units=_NC2, activation='sigmoid', 
                    name='net_layer3')(net_layer2)
            scale_layer = ScaleFE(_NC2, max_FE, name='scale_layer')(
                    net_layer3)
            #E_layer = Dense(units=1, activation='sigmoid', 
                    #name='E_layer')(net_layer3)
            #sumNet_layer = SumNet(max_FE, name='sumNet_layer')(net_layer3)
            #scaleE_layer1 = ScaleE1(max_E, name='scaleE_layer1')(sumNet_layer)

            eij_layer = EijMatrix_test(n_atoms, _NC2, prescale, atoms_flat, 
                    #name='eij_layer')([coords_layer, net_layer3])
                    name='eij_layer')([coords_layer, scale_layer])
            scaleE_layer2 = ScaleE2(prescale, name='scaleE_layer2')(
                    eij_layer)

            eij_layer2 = EijMatrix_test2(n_atoms, _NC2, 
                    #name='eij_layer2')([coords_layer, net_layer3])
                    name='eij_layer2')([coords_layer, scale_layer])
            dE_dx = EnergyGradient(n_atoms, _NC2, 
                    name='dE_dx')([scaleE_layer2, coords_layer])
            qpairs = GetQ(n_atoms, _NC2, atoms_flat, name='qpairs')(
                    [coords_layer, dE_dx, eij_layer])
                    #dE_dx, sumNet_layer])



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


            rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, 
                    patience=1000, min_lr=0.001)
            optimizer = keras.optimizers.Adam(lr=0.001,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)


            model = Model(
                    inputs=[coords_layer], 
                    #inputs=[NRF_layer], 
                    outputs=[
                        #coords_layer,
                        #forces_layer, 
                        #eij_layer2, 
                        dE_dx, 
                        scale_layer,
                        #net_layer3,
                        qpairs,
                        #scale_atomEx, 
                        #sum_atomEx, 
                        scaleE_layer2,
                        #net_layer2,
                        eij_layer, 
                        eij_layer2
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
                            'eij_matrix_test2': 'mse',
                            #'scale_atom_e': 'mse',
                            #'sum_atom_e': 'mse',
                            'eij_matrix_test': 'mse',
                            'scale_fe': 'mse',
                            #'net_layer3': 'mse',
                            #'coords_layer': 'mse'
                            'scale_e2': 'mse',
                            'get_q': 'mse',
                            },
                        loss_weights={
                            #'for_forces': 0, #grad_loss - q_loss[l],
                            'energy_gradient': grad_loss[l],
                            'eij_matrix_test': e_loss[l],
                            'scale_e2': e2_loss[l],
                            'eij_matrix_test2': f_loss[l],
                            #'scale_atom_e': q_loss[l],
                            'scale_fe': scaleq_loss[l],
                            #'net_layer3': scaleq_loss[l],
                            'get_q': q_loss[l],
                            #'sum_atom_e': e_loss[l],
                            #'coords_layer': 0
                            },
                        optimizer='adam',
                        #optimizer=optimizer,
                        )
            if l != 0:
                model.compile(
                        loss={
                            #'for_forces_'+l2: custom_loss_pass(model, 
                            #tf.constant(train_input_coords, dtype=tf.float32),
                            #tf.constant(train_forces, dtype=tf.float32),
                            #),
                            'energy_gradient_'+l2: 'mse',
                            #'eij_matrix_test_'+l2: 'mse',
                            'scale_e2_'+l2: 'mse',
                            #'eij_matrix_test2_'+l2: 'mse',
                            #'scale_atom_e_'+l2: 'mse',
                            #'scale_fe_'+l2: 'mse',
                            'get_q_'+l2: 'mse',
                            #'sum_atom_e_'+l2: 'mse',
                            #'coords_layer': 'mse'
                            },
                        loss_weights={
                            #'for_forces_'+l2: 0, #grad_loss - q_loss[l],
                            'energy_gradient_'+l2: grad_loss[l],
                            #'eij_matrix_test_'+l2: e_loss[l],
                            'scale_e2_'+l2: e_loss[l],
                            #'eij_matrix_test2_'+l2: f_loss[l],
                            #'scale_atom_e_'+l2: q_loss[l],
                            #'scale_fe_'+l2: q_loss[l],
                            'get_q_'+l2: q_loss[l],
                            #'sum_atom_e_'+l2: e_loss[l],
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
                model.load_weights('2/best_ever_model_0')
            '''

            #fit = True
            if fit:
                model.fit(train2_input_coords, 
                        [
                            train2_forces, 
                            train2_output_matFE, 
                            train2_output_matFE, 
                            train2_output_E_postscale, 
                            train2_output_E, 
                            train2_forces, 
                            ],
                        validation_data=(val_input_coords, 
                            [
                                val_forces, 
                                val_output_matFE, 
                                val_output_matFE, 
                                val_output_E_postscale, 
                                val_output_E, 
                                val_forces, 
                                ]),
                        epochs=1000000, 
                        verbose=2,
                        #callbacks=[mc],
                        callbacks=[es,mc],
                        #callbacks=[es,mc,rlrop],
                        )

                best_error = model.evaluate(train2_input_coords, 
                        [
                            train2_forces, 
                            train2_output_matFE, 
                            train2_output_matFE, 
                            train2_output_E_postscale, 
                            train2_output_E, 
                            train2_forces, 
                            ],
                        verbose=0)
                print(l, 'model error train2: ', best_error)

                best_error2 = model.evaluate(val_input_coords, 
                        [
                            val_forces, 
                            val_output_matFE, 
                            val_output_matFE, 
                            val_output_E_postscale, 
                            val_output_E, 
                            val_forces, 
                            ],
                        verbose=0)
                print(l, 'model error val: ', best_error2)

                best_error3 = model.evaluate(train_input_coords, 
                        [
                            train_forces, 
                            train_output_matFE, 
                            train_output_matFE, 
                            train_output_E_postscale, 
                            train_output_E, 
                            train_forces, 
                            ],
                        verbose=0)
                print(l, 'model error train: ', best_error3)

                best_error4 = model.evaluate(test_input_coords, 
                        [
                            test_forces, 
                            test_output_matFE, 
                            test_output_matFE, 
                            test_output_E_postscale, 
                            test_output_E, 
                            test_forces, 
                            ],
                        verbose=0)
                print(l, 'model error test: ', best_error4)

                best_error5 = model.evaluate(input_coords, 
                        [
                            output_F, 
                            output_matFE, 
                            output_matFE, 
                            output_E_postscale, 
                            output_E, 
                            output_F, 
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
            model_file='best_model1'#'../model/best_ever_model_6'
            print('load_weights', load_weights, model_file)
            model.load_weights(model_file)
            #model.load_weights('best_model')

        if fit:
            best_error = model.evaluate(train_input_coords, 
                    [
                        train_forces, 
                        train_output_matFE,
                        train_output_matFE,
                        train_output_E_postscale,
                        train_output_E,
                        train_forces, 
                        ], verbose=0)
            print(l, len(train_input_coords), 'train model error: ', 
                    best_error)


            sys.stdout.flush()
            #sys.exit()

            #E_layer_output = Model(inputs=model.input, 
                    #outputs=model.get_layer('eij_matrix_test').output)

            for i in range(3):
                print('\n\t', i)
                print('NRFs\n', train2_input_NRF[i])
                print('coords\n', train2_input_coords[i])
                print('energy\n', train2_output_E_postscale[i])
                print('E\n', train2_output_E[i])
                print('forces\n', train2_forces[i])
                print('mat_FE\n', train2_output_matFE[i])
                prediction = model.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3)
                        #train2_input_NRF[i].reshape(1,_NC2)
                        )
                print('prediction')
                for count, p in enumerate(prediction):
                    print(count, p)

                au2kcalmola = 627.5095 * 0.529177
                print('mat_FE * NRF')
                _E = np.sum(train2_input_NRF[i] / au2kcalmola * prediction[1])
                print(_E)
                print(np.sum(train2_input_NRF[i] / au2kcalmola * train2_output_matFE[i]))

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
                #with tf.GradientTape(watch_accessed_variables=False) as tape2:
                with tf.GradientTape() as tape2:
                    tape2.watch(x1)
                    y = model(x1)[0]
                gradient1 = tape2.gradient(y, x1) * -1
                print('tape_grad\n', gradient1)
                #sess = tf.compat.v1.Session()
                #print(sess.run(gradient1))
                #print(gradient1.eval())
                print(tf.keras.backend.get_value(gradient1))
                #tf.keras.backend.eval(gradient1)


                #grad = grad_layer_output.predict(
                        #train2_input_coords[i].reshape(1,n_atoms,3))
                #print('grad_lambda\n', grad[0])
                variance, translations, rotations = Network.check_invariance(
                        train2_input_coords[i], prediction[0][0],
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
            prediction_E = prediction[3].flatten()
            #prediction_F = prediction[1].flatten()
            prediction_matFE = prediction[1]
            #prediction_atomFE = prediction[1]
            #prediction_E = prediction[2].flatten()

            x_tensor = tf.constant(input_coords, dtype=tf.float32)
            with tf.GradientTape(watch_accessed_variables=False) as tape3:
                tape3.watch(x_tensor)
                y = model(x_tensor)[0]
            #prediction_F = tape3.gradient(y, x_tensor).numpy() * -1
            prediction_F = prediction[0]
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
            #atom_pairs = [str(i) for i in range(n_atoms)]
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

            '''
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
            '''

            mae, rms, msd = Binner.get_error(output_E_postscale.flatten(), 
                    prediction_E.flatten())
            print('\n{} E All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_E_postscale), mae, rms, 
                    msd))
            check_rms = (np.sum((prediction_E.flatten() - 
                    output_E_postscale.flatten()) ** 2) / 
                    len(prediction_E.flatten())) ** 0.5
            print('check rms:', check_rms, len(prediction_E.flatten()))
            bin_edges, hist = Binner.get_scurve(output_E_postscale.flatten(), 
                    prediction_E.flatten(), 'all-hist1.txt')
            Plotter.plot_2d([bin_edges], [hist], ['all'], 
                    'Error', '% of points below error', 
                    'all-s-curves-E.png')

            mae, rms, msd = Binner.get_error(output_F.flatten(), 
                    prediction_F.flatten())
            print('\n{} grad F All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_F), mae, rms, msd))
            check_rms = (np.sum((prediction_F.flatten() - 
                    output_F.flatten()) ** 2) / 
                    len(prediction_F.flatten())) ** 0.5
            print('check rms:', check_rms, len(prediction_F.flatten()))
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
            check_rms = (np.sum((output_matFE.flatten() - 
                    prediction_matFE.flatten()) ** 2) / 
                    len(prediction_matFE.flatten())) ** 0.5
            print('check rms:', check_rms, len(prediction_matFE.flatten()))
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
        return model


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
