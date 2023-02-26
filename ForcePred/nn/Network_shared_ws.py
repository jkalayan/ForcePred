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
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers

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
import os

start_time = time.time()

#tf.compat.v1.disable_eager_execution()


class NuclearChargePairs(Layer):
    def __init__(self, _NC2, n_atoms, **kwargs):
        super(NuclearChargePairs, self).__init__()
        self._NC2 = _NC2
        self.n_atoms = n_atoms

    def call(self, atom_nc):
        a = tf.expand_dims(atom_nc, 2)
        b = tf.expand_dims(atom_nc, 1)
        c = a * b
        tri1 = tf.linalg.band_part(c, -1, 0) #lower
        tri2 = tf.linalg.band_part(tri1, 0, 0) #lower
        tri = tri1 - tri2
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        nc_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(atom_nc)[0], self._NC2)) #reshape to _NC2
        return nc_flat 


class CoordsToNRF(Layer):
    def __init__(self, max_NRF, _NC2, n_atoms, **kwargs):
        super(CoordsToNRF, self).__init__()
        #self.atoms_flat = atoms_flat
        #self.atoms2 = tf.Variable(atoms2)
        self.max_NRF = max_NRF
        self._NC2 = _NC2
        self.n_atoms = n_atoms
        #self.name = name
        self.au2kcalmola = 627.5095 * 0.529177

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        n_atoms = input_shape[1]
        #return (batch_size, n_atoms, n_atoms)
        return (batch_size, n_atoms, self._NC2)
        #return (batch_size, self._NC2)

    def call(self, coords_nc):
        coords, atom_nc = coords_nc
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

        r = diff_flat ** 0.5
        recip_r2 = 1 / r ** 2
        #_NRF = (((self.atoms_flat * self.au2kcalmola) * recip_r2) / 
                #self.max_NRF) #scaled

        # include a cutoff
        #atom_nc = tf.where(r > 4., 0., atom_nc)

        _NRF = (((atom_nc * self.au2kcalmola) * recip_r2) / 
                self.max_NRF) #scaled
        _NRF = tf.reshape(_NRF, 
                shape=(tf.shape(coords)[0], self._NC2)) #reshape to _NC2
        #_NRF = tf.expand_dims(_NRF, -1)
        #_NRF = tf.tile(_NRF, [1, self.n_atoms])
        return _NRF


class TransformNRF(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(TransformNRF, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def call(self, _NRF):
        tri_NRF = Triangle(self.n_atoms)(_NRF) # b,N,N
        tri_NRF = tri_NRF / 2 #removes double counting
        # put eij_F in format (batch,N,NC2), some NC2 will be zeros
        new_NRF = []
        n_atoms = tri_NRF.shape.as_list()[1]
        for i in range(n_atoms):
            atom_i = tri_NRF[:,i,:]
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
            new_NRF.append(s)

        eij_NRF = tf.stack(new_NRF)
        eij_NRF = tf.transpose(eij_NRF, perm=[1,0,2]) # b,N,NC2
        #eij_NRF = tf.expand_dims(eij_NRF, -1)
        eij_NRF = tf.reshape(eij_NRF, 
                shape=(tf.shape(eij_NRF)[0], self.n_atoms, self._NC2, 1)) #reshape to _NC2

        return eij_NRF #_NRF


class UnscaleE(Layer):
    def __init__(self, prescale, **kwargs):
        super(UnscaleE, self).__init__()
        self.prescale = prescale

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, E_scaled):
        E = ((E_scaled - self.prescale[2]) / 
                (self.prescale[3] - self.prescale[2]) * 
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])
        return E


class SumQ(Layer):
    def __init__(self, _NC2, **kwargs):
        super(SumQ, self).__init__()
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, decomp):
        #scale decompFE
        decomp_summed = tf.reduce_sum(decomp, axis=1)
        return decomp_summed


class UnscaleQ(Layer):
    def __init__(self, _NC2, max_Q, **kwargs):
        super(UnscaleQ, self).__init__()
        self._NC2 = _NC2
        self.max_Q = max_Q

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, decomp_scaled):
        #unscale decompFE
        decomp_scaled = tf.reshape(decomp_scaled, 
                shape=(tf.shape(decomp_scaled)[0], -1))
        decomp = (decomp_scaled - 0.5) * (2 * self.max_Q)
        #decomp = decomp_scaled * 1
        decomp = tf.reshape(decomp, 
                shape=(tf.shape(decomp_scaled)[0], self._NC2)) #reshape to _NC2

        return decomp


class ScaleQ(Layer):
    def __init__(self, _NC2, max_Q, **kwargs):
        super(ScaleQ, self).__init__()
        self._NC2 = _NC2
        self.max_Q = max_Q

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, decomp):
        #scale decompFE
        decomp = tf.reshape(decomp, shape=(tf.shape(decomp)[0], -1))
        decomp_scaled = (decomp / (2 * self.max_Q)) + 0.5
        decomp_scaled = tf.reshape(decomp_scaled, 
                shape=(tf.shape(decomp)[0], self._NC2)) #reshape to _NC2
        return decomp_scaled


class E_Recomposition(Layer):
    def __init__(self, n_atoms, _NC2, bias, **kwargs):
        super(E_Recomposition, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        self.bias = bias

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        #return (batch_size, self._NC2)
        return (batch_size, 1)

    def call(self, coords_decompFE):
        coords, decompFE = coords_decompFE
        decompFE = tf.reshape(decompFE, shape=(tf.shape(decompFE)[0], -1))

        eij_F, r_flat = Projection(self.n_atoms, self._NC2)(coords)
        recip_r_flat = 1 
        if self.bias == '1/r':
            recip_r_flat = 1 / r_flat 
        if self.bias == '1/r2':
            recip_r_flat = 1 / r_flat ** 2
        if self.bias == 'r':
            recip_r_flat = r_flat 

        if self.bias != 'sum':
            norm_recip_r = tf.reduce_sum(recip_r_flat ** 2, axis=1, 
                    keepdims=True) ** 0.5
            eij_E = recip_r_flat / norm_recip_r
            #eij_FE = tf.concat([eij_F, eij_E], axis=1)
            recompE = tf.einsum('bi, bi -> b', eij_E, decompFE) # b,NC2 b,NC2

        ##summing decompFE instead
        if self.bias == 'sum':
            recompE = tf.reduce_sum(decompFE, axis=1, keepdims=True)

        recompE = tf.reshape(recompE, shape=(tf.shape(coords)[0], 1)) # b,1

        return recompE


class FE_Decomposition(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(FE_Decomposition, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def call(self, coords_F_E):
        coords, F, E = coords_F_E
        F_reshaped = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        FE = tf.concat([F_reshaped, E], axis=1)
        FE = tf.expand_dims(FE, 2)
        eij_F, r_flat = Projection(self.n_atoms, self._NC2)(coords)
        recip_r = 1 / r_flat
        norm_r = tf.reduce_sum(recip_r ** 2, axis=1, keepdims=True) ** 0.5
        norm = recip_r / norm_r
        eij_E = tf.expand_dims(norm, 1)
        eij_FE = tf.concat([eij_F, eij_E], axis=1) # b,3N+1,NC2+1

        inv_eij = tf.linalg.pinv(eij_FE) # b,NC2+1,3N+1
        qs = tf.linalg.matmul(inv_eij, FE) # b,NC2+1,3N+1 * b,3N+1,1
        qs = tf.reshape(qs, shape=(tf.shape(qs)[0], self._NC2)) # b,NC2+1
        qs = qs * norm # remove bias so sum q = E

        return qs


class Projection(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(Projection, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def call(self, coords):
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) # get sqrd diff
        # flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) # reshape to _NC2
        r_flat = diff_flat**0.5

        r = Triangle(self.n_atoms)(r_flat)
        r2 = tf.expand_dims(r, 3)
        # replace zeros with ones
        safe = tf.where(tf.equal(r2, 0.), 1., r2)
        eij_F = diff / safe

        # put eij_F in format (batch,3N,NC2), some NC2 will be zeros
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
        eij_F = tf.transpose(eij_F, perm=[1,0,2]) # b,3N,NC2

        return eij_F, r_flat


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


class SharedWeights(Layer):
    def __init__(self, n_atoms, _NC2, units=32, **kwargs):
        super(SharedWeights, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, self.units),
                initializer='random_normal', trainable=True, name='weights')
        #self.b = self.add_weight(shape=(self.units,), 
                #initializer='random_normal', trainable=True, name='bias')

    def call(self, _NRF):
        _NRF = tf.expand_dims(_NRF, -1)
        #return tf.reduce_sum((_NRF * self.w) + self.b, axis=1)
        #return tf.reduce_sum((_NRF * self.w), axis=1) #+ self.b
        #return (_NRF * self.w) + self.b
        return _NRF * self.w


class Reduce(Layer):
    def __init__(self, _NC2, **kwargs):
        super(Reduce, self).__init__()
        self._NC2 = _NC2

    def call(self, inp):
        #scale decompFE
        out = tf.reduce_sum(inp, axis=1)
        return out


class EnergyGradient(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(EnergyGradient, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms, 3)
        #return (batch_size, self._NC2)

    def call(self, E_coords):
        E, coords = E_coords
        gradients = tf.gradients(E, coords, 
                unconnected_gradients='zero')
        return gradients[0] * -1


class Network(object):
    '''
    '''
    def __init__(self, molecule):
        self.model = None
        #self.model_name = None
        #self.atoms = molecule.atoms
        #self.n_atoms = len(self.atoms) 
        #self._NC2 = int(self.n_atoms * (self.n_atoms-1)/2) 
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


    def get_coord_FE_model(self, molecule, prescale1, n_nodes=1000, 
            n_layers=1, grad_loss_w=1000, qFE_loss_w=1, E_loss_w=1, 
            extra_cols=0, bias='1/r', training_model=True,
            load_model=None):
        '''Input coordinates and z_types into model to get NRFS which then 
        are used to predict decompFE, which are then recomposed to give
        Cart Fs and molecular E, both of which could be used in the loss
        function, could weight the E or Fs as required.
        '''

        # here we pre-define the variable values depending on whether
        # we are training the model or if we are loading the model
        # for use in MD simulations for example.
        #training_model = True
        if training_model:
            #for training a model
            get_data = True
            load_first = True
            fit = True
            load_weights = False
            inp_out_pred = True

        if training_model == False:
            #for loading a model to use with openmm
            get_data = False #True #
            load_first = False
            fit = False
            load_weights = True
            inp_out_pred = False

        # the get_data variable should be set as true when we are training,
        # here is where we split our data into the training, validation and
        # test sets.
        #get_data = True
        if get_data:
            print('\nSplit data into training/validation and test sets')
            print('N train {}\nN val {}\nN test {}'.format(
                    len(molecule.train), len(molecule.val), 
                    len(molecule.test)))

            n_atoms = len(molecule.atoms)
            _NC2 = int(n_atoms*(n_atoms-1)/2)
            #extra_cols = n_atoms #1 ## !!!!!!!!
            #print('!!!!! extra_cols:', extra_cols)

            # manipulate nuclear charges here
            atoms = np.array([float(i) for i in molecule.atoms], 
                    dtype='float32')
            atoms_ = tf.convert_to_tensor(atoms, dtype=tf.float32)
            atoms_int = tf.convert_to_tensor(atoms, dtype=tf.int32)
            #multiply each z with each other to get a sq matrix
            atoms2 = tf.tensordot(tf.expand_dims(atoms_, 0), 
                    tf.expand_dims(atoms_, 0), axes=[[0],[0]])
            atoms_flat = []
            for i in range(n_atoms):
                for j in range(i):
                    ij = atoms[i] * atoms[j]
                    atoms_flat.append(ij)
            atoms_flat = tf.convert_to_tensor(atoms_flat, 
                    dtype=tf.float32) #_NC2

            # all structures
            input_coords = molecule.coords#.reshape(-1,n_atoms*3)
            input_NRF = molecule.mat_NRF.reshape(-1,_NC2)
            output_matFE = molecule.mat_FE.reshape(-1,_NC2+extra_cols)
            output_matF = molecule.mat_F.reshape(-1,_NC2+extra_cols)
            output_matE = molecule.mat_E.reshape(-1,_NC2+extra_cols)
            output_FE = np.concatenate((molecule.forces.reshape(-1,n_atoms*3), 
                    molecule.energies.reshape(-1,1)), axis=1)
            output_F = molecule.forces.reshape(-1,n_atoms,3)
            output_E = molecule.energies.reshape(-1,1)
            output_E_postscale = ((output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])

            # training structures
            train_input_coords = np.take(input_coords, molecule.train, axis=0)
            train_input_NRF = np.take(input_NRF, molecule.train, axis=0)
            train_output_matFE = np.take(output_matFE, molecule.train, axis=0)
            train_output_matF = np.take(output_matF, molecule.train, axis=0)
            train_output_matE = np.take(output_matE, molecule.train, axis=0)
            train_output_FE = np.take(output_FE, molecule.train, axis=0)
            train_output_E = np.take(output_E, molecule.train, axis=0)
            train_forces = np.take(molecule.forces, molecule.train, axis=0)

            # test structures
            test_input_coords = np.take(input_coords, molecule.test, axis=0)
            test_input_NRF = np.take(input_NRF, molecule.test, axis=0)
            test_output_matFE = np.take(output_matFE, molecule.test, axis=0)
            test_output_matF = np.take(output_matF, molecule.test, axis=0)
            test_output_matE = np.take(output_matE, molecule.test, axis=0)
            test_output_E = np.take(output_E, molecule.test, axis=0)
            test_output_F = np.take(output_F, molecule.test, axis=0)
            test_forces = np.take(molecule.forces, molecule.test, axis=0)

            # scaling values
            max_NRF1 = np.max(np.abs(train_input_NRF))
            max_E1 = np.max(np.abs(train_output_E))
            max_matFE1 = np.max(np.abs(train_output_matFE))
            max_matF1 = np.max(np.abs(train_output_matF))
            max_matE1 = np.max(np.abs(train_output_matE))
            prescale1[4] = max_NRF1
            prescale1[5] = max_matFE1

            isExist = os.path.exists('model')
            if not isExist:
                os.makedirs('model')
            np.savetxt('model/prescale.txt', 
                    (np.array(prescale1)).reshape(-1,1))
            np.savetxt('model/atoms.txt', 
                    (np.array(molecule.atoms)).reshape(-1,1))
            print('prescale1:', prescale1)

            # scaled training data
            train_input_NRF_scaled = np.take(input_NRF, 
                    molecule.train, axis=0)\
                    / max_NRF1
            train_output_matFE_scaled = (train_output_matFE / 
                    (2 * max_matFE1) + 0.5)
            train_output_matF_scaled = (train_output_matF / 
                    (2 * max_matF1) + 0.5)
            train_output_matFE_scaled = (train_output_matE / 
                    (2 * max_matE1) + 0.5)
            train_output_E_postscale = ((train_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])
            test_output_E_postscale = ((test_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])
            print(train_input_coords.shape, train_output_FE.shape)
            print('max_NRF: {:.3f}, max_qFE: {:.3f}, max_E: {}'.format(
                    max_NRF1, max_matFE1, max_E1))

            # convert scaling values to tensors
            max_NRF = tf.constant(max_NRF1, dtype=tf.float32)
            #max_E = tf.constant(max_E1, dtype=tf.float32)
            max_matFE = tf.constant(max_matFE1, dtype=tf.float32)
            #max_matF = tf.constant(max_matF1, dtype=tf.float32)
            #max_matE = tf.constant(max_matE1, dtype=tf.float32)
            prescale = tf.constant(prescale1, dtype=tf.float32)
            #n_atoms_tf = tf.constant(n_atoms, dtype=tf.int32)


            # get validation from within training
            train2_input_coords = np.take(input_coords, 
                    molecule.train2, axis=0)
            val_input_coords = np.take(input_coords, 
                    molecule.val, axis=0)
            train2_output_E = np.take(output_E, molecule.train2, axis=0)
            val_output_E = np.take(output_E, molecule.val, axis=0)
            train2_output_E_postscale = np.take(output_E_postscale, 
                    molecule.train2, axis=0)
            val_output_E_postscale = np.take(output_E_postscale, 
                    molecule.val, axis=0)
            print('train max F: {:.3f}, min F {:.3f}'.format(
                np.max(train_forces), np.min(train_forces)))
            train2_forces = np.take(output_F, molecule.train2, axis=0)
            print('train2 max F: {:.3f}, min F {:.3f}'.format(
                np.max(train2_forces), np.min(train2_forces)))
            val_forces = np.take(output_F, molecule.val, axis=0)
            print('val max F: {:.3f}, min F {:.3f}'.format(
                np.max(val_forces), np.min(val_forces)))
            train2_output_matFE = np.take(output_matFE, 
                    molecule.train2, axis=0)
            val_output_matFE = np.take(output_matFE, 
                    molecule.val, axis=0)
            train2_output_matF = np.take(output_matF, 
                    molecule.train2, axis=0)
            val_output_matF = np.take(output_matF, molecule.val, axis=0)
            train2_output_matE = np.take(output_matE, 
                    molecule.train2, axis=0)
            val_output_matE = np.take(output_matE, 
                    molecule.val, axis=0)
            train2_input_NRF = np.take(input_NRF, 
                    molecule.train2, axis=0)
            val_input_NRF = np.take(input_NRF, 
                    molecule.val, axis=0)

            # create arrays of nuclear charges for different sets
            train2_atoms = np.tile(atoms, (len(train2_input_coords), 1))
            val_atoms = np.tile(atoms, (len(val_input_coords), 1))
            test_atoms = np.tile(atoms, (len(test_input_coords), 1))
            all_atoms = np.tile(atoms, (len(input_coords), 1))
            print('atoms', atoms.shape)


        # building ANN here
        # the variables immediately below don't need to be changed, but have a
        # look at the keras pages to see what each mean and do in more detail
        file_name='model/best_model' # name of model
        monitor_loss='val_loss' # monitor validation loss during training
        set_patience=1000 # how many epochs before no improvement in training
                # is made and training is stopped
        restore_best_weights = True # after each reduction in learning rate,
                # save the weights that gave the lowest loss
        batch_size = 32 # number of structures that are cycled through in each
                # epoch to adjust weights, 32 is the keras default, you may
                # want to reduce this number if structures are very similar
                # or increase this if structures are very different (like
                # in a general ANN)
        print('monitor_loss:', monitor_loss)
        print('set_patience', set_patience)
        print('restore_best_weights', restore_best_weights)
        print('batch_size:', batch_size)
        if load_weights:
            prescale = np.loadtxt(load_model+'/prescale.txt', 
                    dtype=np.float32).reshape(-1)
            atoms = np.loadtxt(load_model+'/atoms.txt', 
                    dtype=np.float32).reshape(-1)
            n_atoms = len(atoms)
            _NC2 = int(n_atoms*(n_atoms-1)/2)
            max_NRF = prescale[4]
            max_matFE = prescale[5]

        best_error_all = 1e20 # used to save the best error for the model
        best_model = None # needs defining before training??
        #grad_loss_w = 1000
        #qFE_loss_w = 0
        #E_loss_w = 1
        #n_layers = 1
        #n_nodes = 1000
        print('dE_dx (gradient/F) loss weight: {} '\
                '\ndecompFE loss weight: {} '\
                '\nE loss weight: {}'.format(
                grad_loss_w, qFE_loss_w, E_loss_w))

        # tell keras your model criteria here
        mc = ModelCheckpoint(file_name, monitor=monitor_loss, mode='min', 
                save_best_only=True, save_weights_only=True
                )
        es = EarlyStopping(monitor=monitor_loss, patience=set_patience, 
                restore_best_weights=restore_best_weights)

        # now we build each keras layer for the ANN
        coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
        nuclear_charge_layer = Input(shape=(n_atoms), 
                name='nuclear_charge_layer')
        nc_pairs_layer = NuclearChargePairs(_NC2, n_atoms)(
                nuclear_charge_layer)
        NRF_layer = CoordsToNRF(max_NRF, _NC2, n_atoms, 
                name='NRF_layer')([coords_layer, nc_pairs_layer])
        #NRF_layer = TransformNRF(n_atoms, _NC2, 
                #name='NRF_tranform')(NRF_layer)
        #shared_layer = SharedWeights(n_atoms, _NC2, units=n_nodes, 
                #activation='silu', 
                #name='shared_w_layer')(NRF_layer)
        connected_layer = NRF_layer #shared_layer #
        for l in range(n_layers):
            net_layer = Dense(units=n_nodes, #768
                    activation='silu', 
                    name='net_layerA{}'.format(l))(connected_layer)
            #net_layer2 = Dense(units=128, activation='silu', 
            #        name='net_layerB{}'.format(l))(net_layer)
            #net_layer3 = Dense(units=128, activation='silu', 
            #        name='net_layerC{}'.format(l))(net_layer2)
            #net_layer4 = Dense(units=64, activation='silu', 
            #        name='net_layerD{}'.format(l))(net_layer)
            connected_layer = net_layer
            '''
            reduce_layer = net_layer#4 #Reduce(_NC2)(net_layer4)
            if l == 0:
                connected_layer = reduce_layer
            else:
                connected_layer = tf.keras.layers.Add()(
                        [reduce_layer, connected_layer])
            '''

            '''
            unscale_qFE_layer1 = UnscaleQ(_NC2, max_matFE, 
                    name='unscale_qF_layer{}'.format(l))(net_layer4)

            E_layer1 = E_Recomposition(n_atoms, _NC2,
                    name='qFE_layer{}'.format(l))(
                    [coords_layer, unscale_qFE_layer1])
            scaleE_layer1 = UnscaleE(prescale, 
                    name='unscaleE_layer{}'.format(l))(E_layer1)
            dE_dx1 = EnergyGradient(n_atoms, _NC2, 
                    name='dE_dx{}'.format(l))([scaleE_layer1, coords_layer])
            qs1 = FE_Decomposition(n_atoms, _NC2, 
                    name='qs{}'.format(l))(
                    [coords_layer, dE_dx1, scaleE_layer1])
            scale_qFE1 = ScaleQ(_NC2, max_matFE, 
                    name='scale_qFE{}'.format(l))(qs1)
            connected_layer = tf.keras.layers.Add()([scale_qFE1, NRF_layer])
            '''
        connected_layer = Dense(units=_NC2+extra_cols, activation='silu', 
                name='net_layerQ')(connected_layer)
        #connected_layer = Reduce(_NC2)(connected_layer)

        unscale_qFE_layer = UnscaleQ(_NC2+extra_cols, max_matFE, 
                name='unscale_qF_layer')(connected_layer) #qs1 #
        E_layer = E_Recomposition(n_atoms, _NC2, bias,
                name='qFE_layer')([coords_layer, unscale_qFE_layer])
                #name='qFE_layer')([coords_layer, connected_layer])
        unscaleE_layer = UnscaleE(prescale, 
                name='unscaleE_layer')(E_layer)
        dE_dx = EnergyGradient(n_atoms, _NC2, 
                name='dE_dx')([unscaleE_layer, coords_layer])
        qs = FE_Decomposition(n_atoms, _NC2, 
                name='qs')([coords_layer, dE_dx, unscaleE_layer])

        rlrop = ReduceLROnPlateau(monitor=monitor_loss, factor=0.5, 
                patience=50, min_lr=1e-4)
        print('rlrop: monitor={}, factor=0.5, patience=50, '\
                'min_lr=1e-4'.format(monitor_loss))
        optimizer = keras.optimizers.Adam(lr=0.001,
                beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)


        # define the model input layers and the output layers used in the
        # loss function
        model = Model(
                inputs=[coords_layer, nuclear_charge_layer], 
                outputs=[
                    dE_dx,
                    #qs,
                    unscale_qFE_layer,
                    unscaleE_layer,
                    ], 
                )


        # compile the model here by providing the type of loss function
        # for each output and the weighting for each loss function
        model.compile(
                loss={
                    #'energy_gradient_{}'.format(n_layers): 'mse',
                    'energy_gradient': 'mse',
                    #'unscale_q_{}'.format(n_layers-1): 'mse',
                    'unscale_q': 'mse',
                    #'fe__decomposition_{}'.format(n_layers-1): 'mse',
                    #'fe__decomposition': 'mse',
                    #'unscale_e_{}'.format(n_layers): 'mse',
                    'unscale_e': 'mse',
                    },
                loss_weights={
                    #'energy_gradient_{}'.format(n_layers): grad_loss_w,
                    'energy_gradient': grad_loss_w,
                    #'unscale_q_{}'.format(n_layers-1): qFE_loss_w,
                    'unscale_q': qFE_loss_w,
                    #'fe__decomposition_{}'.format(n_layers-1): qFE_loss_w,
                    #'fe__decomposition': qFE_loss_w,
                    #'unscale_e_{}'.format(n_layers): E_loss_w,
                    'unscale_e': E_loss_w,
                    },
                #optimizer='adam',
                optimizer=optimizer,
                )

        # print out the model here
        model.summary()

        print('initial learning rate:', K.eval(model.optimizer.lr))

        load_first = False #True #
        if load_first:
            model_file = load_model+'/best_ever_model' #'model/best_ever_model'
            print('load_first', load_weights, model_file)
            model.load_weights(model_file)

        # here is where the training occurs, the fit flag should be True
        # when training and False when the network needs to be loaded
        # before running MD simulations, etc.
        if fit:
            result = model.fit([train2_input_coords, train2_atoms],
                    [
                        train2_forces, 
                        train2_output_matFE, 
                        train2_output_E_postscale, 
                        ],
                    validation_data=([val_input_coords, val_atoms],
                        [
                            val_forces, 
                            val_output_matFE, 
                            val_output_E_postscale, 
                            ]),
                    epochs=1000000, 
                    verbose=2,
                    batch_size=batch_size,
                    #callbacks=[mc],
                    #callbacks=[es,mc],
                    callbacks=[es,mc,rlrop],
                    )


            # get initial test error for the model
            best_error4 = model.evaluate([test_input_coords, test_atoms],
                    [
                        test_forces, 
                        test_output_matFE, 
                        test_output_E_postscale, 
                        ],
                    verbose=0)
            print('model error test: ', best_error4)


            model.save_weights('model/best_ever_model')
        print()


        # the load_weights variable is only set to True when we have already
        # trained the network and we now need to load these weights for the
        # given network above.
        #load_weights = False
        if load_weights:
            model_file = load_model+'/best_ever_model' #'model/best_ever_model'
            print('load_weights', load_weights, model_file)
            model.load_weights(model_file)

        # the inp_out_pred variable should only True when training the network
        # it will print out the test errors
        #inp_out_pred = True
        if inp_out_pred:

            prediction = model.predict([input_coords, all_atoms])
            prediction_E = prediction[2].flatten()
            prediction_matFE = prediction[1]

            #Writer.write_csv([output_E_postscale, prediction_E], 
                    #'all_energies', 'actual_E,prediction_E')

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

            '''
            Writer.write_csv([
                    input_NRF,
                    output_matFE,
                    prediction_matFE
                    ], 'all_decompFE', 
                    header)
            sys.stdout.flush()
            '''

            print('Predict testset data')
            test_prediction = model.predict([test_input_coords, test_atoms])
            test_prediction_E = test_prediction[2].flatten()
            test_prediction_F = test_prediction[0]
            test_prediction_matFE = test_prediction[1]

            print('\npair ave_r min_q max_q mae rms msd')
            ave_rs, max_qs, maes = [], [], []
            largest, info = 0, []
            ave_rs_all = np.average(molecule.mat_r, axis=0)
            min_qs_all = np.amin(molecule.mat_FE, axis=0)
            max_qs_all = np.amax(molecule.mat_FE, axis=0)
            for ij in range(_NC2):
                ave_r = ave_rs_all[ij] 
                min_q = min_qs_all[ij]
                max_q = max_qs_all[ij]
                mae, rms, msd = Binner.get_error(
                        test_output_matFE[:,ij].flatten(), 
                        test_prediction_matFE[:,ij].flatten())
                ave_rs.append(ave_r)
                max_qs.append(max_q)
                maes.append(mae)
                if mae > largest:
                    largest = mae
                    info = [atom_pairs[ij], ave_r, mae]
                print('{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
                        atom_pairs[ij], ave_r, min_q, max_q, mae, rms, msd))
            print('\nlargest MAE pair MAE:', info)
            Plotter.xy_scatter([ave_rs], [maes], [''], ['k'], 
                    '$<r_{ij}> / \AA$', 'MAE / kcal/mol/$\AA$', [10], 
                    'scatter-ave_r-MAE-decompFE.png')
            Plotter.xy_scatter([ave_rs], [max_qs], [''], ['k'], 
                    '$<r_{ij}> / \AA$', 'max($q$) / kcal/mol/$\AA$', [10], 
                    'scatter-ave_r-max_q-decompFE.png')
            Plotter.xy_scatter([max_qs], [maes], [''], ['k'], 
                    'max($q$) / kcal/mol/$\AA$', 'MAE / kcal/mol/$\AA$', [10], 
                    'scatter-max_q-MAE-decompFE.png')

            mae, rms, msd = Binner.get_error(test_output_F.flatten(), 
                    test_prediction_F.flatten())
            print('\n{} testset structures, errors for gradient (F) / '\
                    'kcal/mol/$\AA$:- '\
                    '\nMAE: {:.3f} \nRMS: {:.3f} '\
                    '\nMSD: {:.3f} \nMSE: {:.3f}'.format(
                        len(test_output_F), mae, rms, msd, rms**2))

            mae, rms, msd = Binner.get_error(test_output_matFE.flatten(), 
                    test_prediction_matFE.flatten())
            print('\n{} testset structures, errors for decompFE / '\
                    'kcal/mol/$\AA$:- '\
                    '\nMAE: {:.3f} \nRMS: {:.3f} '\
                    '\nMSD: {:.3f} \nMSE: {:.3f}'.format(
                    len(test_output_matFE), mae, rms, msd, rms**2))

            mae, rms, msd = Binner.get_error(test_output_E_postscale.flatten(), 
                    test_prediction_E.flatten())
            print('\n{} testset structures, errors for E / kcal/mol:- '\
                    '\nMAE: {:.3f} '\
                    '\nRMS: {:.3f} \nMSD: {:.3f} '\
                    '\nMSE: {:.3f}'.format(
                    len(test_output_E_postscale), mae, rms, msd, rms**2))

            '''
            bin_edges, hist = Binner.get_scurve(
                    test_output_E_postscale.flatten(), 
                    test_prediction_E.flatten(), 'testset_hist_E.dat')
            #Plotter.plot_2d([bin_edges], [hist], [''], 
                    #'Error', '% of points below error', 
                    #'testset_s_curve_E.png', log=True)
            bin_edges, hist = Binner.get_scurve(test_output_F.flatten(), 
                    test_prediction_F.flatten(), 'testset_hist_F.dat')
            #Plotter.plot_2d([bin_edges], [hist], [''], 
                    #'Error', '% of points below error', 
                    #'testset_s_curves_F.png', log=True)
            bin_edges, hist = Binner.get_scurve(test_output_matFE.flatten(), 
                    test_prediction_matFE.flatten(), 'testset_hist_decompFE.dat')
            #Plotter.plot_2d([bin_edges], [hist], [''], 
                    #'Error', '% of points below error', 
                    #'testset_s_curves_decompFE.png', log=True)
            '''

            '''
            model_loss = result.history['loss']
            model_val_loss = result.history['val_loss']
            Writer.write_csv([model_loss, model_val_loss], 
                    'loss', 'loss val_loss', delimiter=' ', ext='dat')

            Plotter.plot_2d([list(range(len(model_loss[10:]))), 
                list(range(len(model_val_loss[10:])))], 
                [model_loss[10:], model_val_loss[10:]], 
                    ['loss', 'val_loss'], 
                    'Epoch', 'Loss', 
                    'loss_curves.png')
            '''


            sys.stdout.flush()


        sys.stdout.flush()
        return model


