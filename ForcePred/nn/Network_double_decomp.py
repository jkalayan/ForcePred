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

#tf.compat.v1.disable_eager_execution()


class OneHot(Layer):
    def __init__(self, atoms_int, **kwargs):
        super(OneHot, self).__init__()
        self.atoms_int = atoms_int

    def call(self, coords):
        ones = coords[:,:,0] / coords[:,:,0]
        ones = tf.cast(ones, dtype=tf.int32)
        one_hot = tf.one_hot(self.atoms_int * ones, 100)
        return one_hot


class CoordsToNRF(Layer):
    def __init__(self, atoms_flat, max_NRF, _NC2, n_atoms, **kwargs):
        super(CoordsToNRF, self).__init__()
        self.atoms_flat = atoms_flat
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
        r = diff_flat ** 0.5
        recip_r2 = 1 / r ** 2
        _NRF = (((self.atoms_flat * self.au2kcalmola) * recip_r2) / 
                self.max_NRF) #scaled

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

        return eij_NRF #_NRF


class SplitNRF(Layer):
    def __init__(self, bonded_indices, nb_indices, **kwargs):
        super(SplitNRF, self).__init__()
        self.bonded_indices = bonded_indices
        self.nb_indices = nb_indices

    def call(self, _NRF):
         
        b_NRF = tf.gather(_NRF, self.bonded_indices, axis=1)
        nb_NRF = tf.gather(_NRF, self.nb_indices, axis=1)

        return b_NRF, nb_NRF


class CombineQ(Layer):
    def __init__(self, all_indices, **kwargs):
        super(CombineQ, self).__init__()
        self.all_indices = all_indices

    def call(self, bonded_nb_Q):
        
        b_Q, nb_Q = bonded_nb_Q
        _Q = tf.concat([b_Q, nb_Q], axis=1)
        decompFE = tf.gather(_Q, self.all_indices, axis=1)

        return decompFE


class RBF(Layer):
    def __init__(self, r_cutoff, n_features, gamma, n_atoms, _NC2, **kwargs):
        super(RBF, self).__init__()
        self.r_cutoff = r_cutoff
        self.n_features = n_features
        self.gamma = gamma #hyperparameter
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def call(self, coords):
        eij_F, r_flat = Projection(self.n_atoms, self._NC2)(coords)
        r = Triangle(self.n_atoms)(r_flat)
        # replace zeros with large number
        r = tf.where(tf.equal(r, 0.), 1000., r)
        mu = tf.linspace(0., 30., 300)
        #mu = tf.constant(mu, dtype=tf.float32)
        rbf = tf.exp(-10. * (r[..., tf.newaxis] - mu) ** 2)
        return rbf



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


class TransposeQ(Layer):
    def __init__(self, _NC2, **kwargs):
        super(TransposeQ, self).__init__()
        self._NC2 = _NC2

    def call(self, Ndecomp):
        shape = Ndecomp.shape.as_list()
        #if len(shape) == 3:
        Ndecomp = tf.transpose(Ndecomp, perm=[0,2,1])
        #if len(shape) == 4:
            #Ndecomp = tf.transpose(Ndecomp, perm=[0,1,3,2])
        return Ndecomp


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
        return decomp_scaled


class DoubleDecomp(Layer):
    def __init__(self, _NC2, **kwargs):
        super(DoubleDecomp, self).__init__()
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, Qs):
        qF, qE = Qs
        E = tf.einsum('bi, bi -> b', qF, qE)
        E = tf.reshape(E, shape=(tf.shape(qF)[0], 1)) #need to define shape
        return E


class Linear(keras.layers.Layer):
    def __init__(self, _NC2, units=32, input_dim=32, **kwargs):
        super(Linear, self).__init__()
        self._NC2 = _NC2
        self.w = self.add_weight(shape=(input_dim, units), 
                initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(units,), 
                initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class FE_Decomposition(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(FE_Decomposition, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    #def build(self, input_shape):
        #self.kernel = self.add_weight('kernel', 
                #shape=[None, self._NC2])

    def call(self, coords_F):
        coords, F = coords_F
        #E = tf.linalg.matmul(NRF, self.kernel)
        #E = tf.reshape(E, shape=(tf.shape(E)[0], 1)) # b,NC2
        #E = self.activation(E)
        F = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        #FE = tf.concat([F, E], axis=1)

        eij_F, r_flat = Projection(self.n_atoms, self._NC2)(coords)
        #recip_r_flat = 1 / r_flat
        #norm_recip_r = tf.reduce_sum(recip_r_flat ** 2, axis=1, 
                #keepdims=True) ** 0.5
        #eij_E = recip_r_flat / norm_recip_r
        #eij_E = tf.reshape(eij_E, shape=(tf.shape(eij_E)[0],1,-1)) # b,NC2
        #eij_FE = tf.concat([eij_F, eij_E], axis=1)

        decompFE = tf.einsum('bij, bi -> bj', eij_F, F) # b,3N+1,NC2 b,3N+1 b,NC2
        decompFE = tf.reshape(decompFE, shape=(tf.shape(decompFE)[0], -1)) # b,NC2

        return decompFE


class Decomposition(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(Decomposition, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, coords_F):
        coords, F = coords_F
        F = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        eij_F, r_flat = Projection(self.n_atoms, self._NC2)(coords)
        decompF = tf.einsum('bij, bi -> bj', eij_F, F) # b,3N,NC2 b,3N b,NC2
        decompF = tf.reshape(decompF, shape=(tf.shape(decompF)[0], -1)) # b,NC2

        return decompF


class Recomposition(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(Recomposition, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms, 3)

    def call(self, coords_decompF):
        coords, decompF = coords_decompF
        decompF = tf.reshape(decompF, shape=(tf.shape(decompF)[0], -1))
        eij_F, r_flat = Projection(self.n_atoms, self._NC2)(coords)
        F = tf.einsum('bij, bj -> bi', eij_F, decompF) # b,3N,NC2 b,NC2 b,3N
        F = tf.reshape(F, shape=(tf.shape(F)[0], -1, 3)) # b,N,3

        return F

class E_Recomposition(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(E_Recomposition, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        #return (batch_size, self._NC2)
        return (batch_size, 1)

    def call(self, coords_decompFE):
        coords, decompFE = coords_decompFE
        decompFE = tf.reshape(decompFE, shape=(tf.shape(decompFE)[0], -1))

        eij_F, r_flat = Projection(self.n_atoms, self._NC2)(coords)
        recip_r_flat = 1 / r_flat
        norm_recip_r = tf.reduce_sum(recip_r_flat ** 2, axis=1, 
                keepdims=True) ** 0.5
        eij_E = recip_r_flat / norm_recip_r
        #eij_FE = tf.concat([eij_F, eij_E], axis=1)

        recompE = tf.einsum('bi, bi -> b', eij_E, decompFE) # b,NC2 b,NC2

        ##summing decompFE instead
        recompE = tf.reduce_sum(decompFE, axis=1, keepdims=True)
        recompE = tf.reshape(recompE, shape=(tf.shape(coords)[0], 1)) # b,1

        return recompE


class Projection(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(Projection, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    #def compute_output_shape(self, input_shape):
        #batch_size = input_shape[0]
        #return (batch_size, self._NC2)
        #return (batch_size, self.n_atoms, 3)

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
        extra_cols = 1 #n_atoms #1 ## !!!!!!!!
        print('!!!!! extra_cols:', extra_cols)
        atoms = np.array([float(i) for i in molecule.atoms], dtype='float32')
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
        atoms_flat = tf.convert_to_tensor(atoms_flat, dtype=tf.float32) #_NC2

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
            get_data = False #True #
            load_first = False
            fit = False
            load_weights = True
            inp_out_pred = False

        #get_data = True
        if get_data:
            split = 100 #500 #200 #2

            '''
            train = round(len(molecule.coords) / split, 3)
            print('\nget train and test sets, '\
                    'training set is {} points'.format(train))
            Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                    split) #get train and test sets
            '''

            '''
            print('!!!use regularly spaced training')
            molecule.train = np.arange(2, len(molecule.coords), split).tolist() 
            molecule.test = [x for x in range(0, len(molecule.coords)) 
                    if x not in molecule.train]
            '''

            print('N train {}\nN test {}'.format(
                    len(molecule.train), len(molecule.test)))

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

            train_input_coords = np.take(input_coords, molecule.train, axis=0)
            train_input_NRF = np.take(input_NRF, molecule.train, axis=0)

            train_output_matFE = np.take(output_matFE, molecule.train, axis=0)
            train_output_matF = np.take(output_matF, molecule.train, axis=0)
            train_output_matE = np.take(output_matE, molecule.train, axis=0)
            train_output_FE = np.take(output_FE, molecule.train, axis=0)
            train_output_E = np.take(output_E, molecule.train, axis=0)
            train_forces = np.take(molecule.forces, molecule.train, axis=0)

            test_input_coords = np.take(input_coords, molecule.test, axis=0)
            test_input_NRF = np.take(input_NRF, molecule.test, axis=0)

            test_output_matFE = np.take(output_matFE, molecule.test, axis=0)
            test_output_matF = np.take(output_matF, molecule.test, axis=0)
            test_output_matE = np.take(output_matE, molecule.test, axis=0)
            test_output_E = np.take(output_E, molecule.test, axis=0)
            test_output_F = np.take(output_F, molecule.test, axis=0)
            test_forces = np.take(molecule.forces, molecule.test, axis=0)

            max_NRF1 = np.max(np.abs(train_input_NRF))
            max_E1 = np.max(np.abs(train_output_E))
            max_matFE1 = np.max(np.abs(train_output_matFE))
            max_matFE1_all = np.amax(np.abs(train_output_matFE), axis=0)
            max_matF1 = np.max(np.abs(train_output_matF))
            max_matE1 = np.max(np.abs(train_output_matE))

            train_input_NRF_scaled = np.take(input_NRF, molecule.train, axis=0)\
                    / max_NRF1
            train_output_matFE_scaled = train_output_matFE / (2 * max_matFE1) + 0.5
            train_output_matF_scaled = train_output_matF / (2 * max_matF1) + 0.5
            train_output_matFE_scaled = train_output_matE / (2 * max_matE1) + 0.5
            print(train_input_coords.shape, train_output_FE.shape)
            print('max_NRF: {}, max_matFE: {}, max_matF: {}, max_matE: {}, '\
                    'max_E: {}'.format(max_NRF1, 
                    max_matFE1, max_matF1, max_matE1, max_E1))
            print('max_matFE_all:', max_matFE1_all)
            max_NRF = tf.constant(max_NRF1, dtype=tf.float32)
            max_E = tf.constant(max_E1, dtype=tf.float32)
            max_matFE = tf.constant(max_matFE1, dtype=tf.float32)
            max_matFE_all = tf.constant(max_matFE1_all, dtype=tf.float32)
            max_matF = tf.constant(max_matF1, dtype=tf.float32)
            max_matE = tf.constant(max_matE1, dtype=tf.float32)
            prescale = tf.constant(prescale1, dtype=tf.float32)
            n_atoms_tf = tf.constant(n_atoms, dtype=tf.int32)


            # get indices for bonded and non-bonded pairs here, 
            # use one structure to get this info to apply to all
            print('\nGet indices for bonded, non-bonded and combined pairs')
            r = molecule.mat_r[0]
            if extra_cols == 1:
                r = np.concatenate((r, [10]), axis=0) #make last r greater than bond cutoff
            print('r', r)
            bonded_indices = [i for i in np.where(r < 2)[0]]
            print('b_inx', bonded_indices)
            nb_indices = [i for i in np.where(r >= 2)[0]]
            print('nb_inx', nb_indices)
            if _NC2 in bonded_indices:
                bonded_indices2 = bonded_indices.copy()[-1]
                nb_indices2 = nb_indices.copy()
            else:
                nb_indices2 = nb_indices.copy()[:-1]
                bonded_indices2 = bonded_indices.copy()
            print(bonded_indices2)
            print(nb_indices2)
            all_indices = bonded_indices + nb_indices
            all_indices = np.argsort(all_indices) #to get back orig order
            print('all_inx', all_indices)
            q_test = molecule.mat_FE[0]
            print('q_test', q_test)
            b_q = q_test[bonded_indices]
            nb_q = q_test[nb_indices]
            print('b_q', b_q)
            print('nb_q', nb_q)
            comb_q = np.concatenate((b_q, nb_q), axis=0)
            print('comb_q', comb_q)
            reorder_q = comb_q[all_indices]
            print('reorder_q', reorder_q)
            print()
            #sys.exit()

            train_output_E_postscale = ((train_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])
            test_output_E_postscale = ((test_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])


            val_points = 50
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
            print('train max F: {}, min F {}'.format(
                np.max(train_forces), np.min(train_forces)))
            train2_forces = np.take(train_forces, train2, axis=0)
            print('train2 max F: {}, min F {}'.format(
                np.max(train2_forces), np.min(train2_forces)))
            val_forces = np.take(train_forces, val, axis=0)
            print('val max F: {}, min F {}'.format(
                np.max(val_forces), np.min(val_forces)))
            train2_output_matFE = np.take(train_output_matFE, train2, axis=0)
            val_output_matFE = np.take(train_output_matFE, val, axis=0)
            train2_output_matF = np.take(train_output_matF, train2, axis=0)
            val_output_matF = np.take(train_output_matF, val, axis=0)
            train2_output_matE = np.take(train_output_matE, train2, axis=0)
            val_output_matE = np.take(train_output_matE, val, axis=0)
            train2_input_NRF = np.take(train_input_NRF, train2, axis=0)
            val_input_NRF = np.take(train_input_NRF, val, axis=0)

            # zeros in shape of forces
            train_zeros_F = np.zeros_like(train_forces)
            train2_zeros_F = np.zeros_like(train2_forces)
            val_zeros_F = np.zeros_like(val_forces)
            test_zeros_F = np.zeros_like(test_forces)
            zeros_F = np.zeros_like(output_F)


        if get_data == False: #hardcoded
            prescale1 = [-167315.01917961572, -167288.834575252, 
                    -260.631337453833, 284.98286516421035]
            #max_NRF1 = 12304.734510536122
            max_NRF1 = 13010.961100355082 #9119.964965436488 
            #max_FE1 = 211.23633766510105
            max_matFE1 = 214.86748872253952 #162.3854400053677 #284.98286516421035
            print('\n!!!!scaled values over-written')
            print('prescale', prescale1)
            print('max_NRF', max_NRF1)
            print('max_matFE', max_matFE1)
            max_NRF = tf.constant(max_NRF1, dtype=tf.float32)
            max_matFE = tf.constant(max_matFE1, dtype=tf.float32)
            prescale = tf.constant(prescale1, dtype=tf.float32)
            n_atoms_tf = tf.constant(n_atoms, dtype=tf.int32)


        file_name='best_model'
        monitor_loss='val_loss' #'loss' #
        set_patience=1000
        restore_best_weights = True
        print('monitor_loss:', monitor_loss)
        print('set_patience', set_patience)
        print('restore_best_weights', restore_best_weights)

        mc = ModelCheckpoint(file_name, monitor=monitor_loss, mode='min', 
                save_best_only=True, save_weights_only=True
                )
        es = EarlyStopping(monitor=monitor_loss, patience=set_patience, 
                restore_best_weights=restore_best_weights)



        #coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
        #NRF_layer = CoordsToNRF_test(atoms_flat, max_NRF, _NC2, n_atoms, 
                #name='NRF_layer')(coords_layer)
        #model = NRF_layer
        
        max_depth = 1
        print('max_depth', max_depth)
        grad_loss = [1000]
        qF_loss = [0]
        qE_loss = [0]
        qFE_loss = [1]
        E_loss = [1]
        q_loss = [0]
        force_diff_loss = [0]
        best_error_all = 1e20
        best_iteration = 0
        best_model = None
        batch_size = 32
        print('batch_size:', batch_size)
        for l in range(0, max_depth):
            print('l', l)
            l2 = str(l)
            print('dE_dx loss weight: {} \nqF loss weight: {} '\
                    '\nqE loss weight: {} \nqFE loss weight: {} '\
                    '\nE loss weight: {} \nq loss weight: {}'.format(
                    grad_loss[l], qF_loss[l], qE_loss[l], qFE_loss[l], 
                    E_loss[l], q_loss[l]))
            #if l > 0:
                #model = concatenate([model2[3], NRF_layer])
            coords_layer = Input(shape=(n_atoms,3), name='coords_layer')

            proj_layer, r_layer = Projection(n_atoms, _NC2)(coords_layer)

            NRF_layer = CoordsToNRF(atoms_flat, max_NRF, _NC2, n_atoms, 
                    name='NRF_layer')(coords_layer)


            one_hot_layer = OneHot(atoms_int, name='one_hot_layer')(coords_layer)

            #'''
            r_cutoff1 = 10
            r_cutoff = tf.constant(r_cutoff1, dtype=tf.float32)
            n_features = int(r_cutoff1/0.1)
            gamma = tf.constant(1, dtype=tf.float32)
            RBF_layer = RBF(r_cutoff, n_features, gamma, n_atoms, _NC2,
                    name='RBF_layer')(coords_layer)
            #embedding_layer = tf.keras.layers.Embedding(100, n_atoms)(NRF_layer) 

            for n in range(3):

                if n == 0:
                    split_NRF_layer = SplitNRF(bonded_indices2, nb_indices2,
                            name='split_NRF_layer')(NRF_layer)
                else:
                    split_NRF_layer = SplitNRF(bonded_indices, nb_indices,
                            name='split_NRF_layer')(NRF_layer)

                #NRF_layer2 = TransformNRF(n_atoms, _NC2, 
                        #name='NRF_layer2_{}'.format(n))(NRF_layer)
                #trans_layer1 = TransposeQ(_NC2, 
                        #name='trans_layer1_{}'.format(n))(RBF_layer)
                #trans_layer2 = TransposeQ(_NC2, 
                        #name='trans_layer2_{}'.format(n))(proj_layer)
                #NRF_splits = tf.split(NRF_layer2, n_atoms, axis=1, 
                        #name='NRF_splits')
                #NRF_splits = tf.split(split_NRF_layer, 22, axis=1, 
                        #name='NRF_splits')


                net_layer1A = Dense(units=128, activation='silu', 
                        name='net_layer1A_{}'.format(n))(split_NRF_layer[0])
                        #(NRF_layer2)#(one_hot_layer)#
                net_layer1B = Dense(units=128, activation='silu', 
                        name='net_layer1B_{}'.format(n))(net_layer1A)
                net_layer1C = Dense(units=128, activation='silu', 
                        name='net_layer1C_{}'.format(n))(net_layer1B)
                net_layer1D = Dense(units=len(bonded_indices),#_NC2,#100, 
                        activation='silu', 
                        name='net_layer1D_{}'.format(n))(net_layer1C)

                net_layer2A = Dense(units=128, activation='silu', 
                        name='net_layer2A_{}'.format(n))(split_NRF_layer[1])#(RBF_layer)
                net_layer2B = Dense(units=128, activation='silu', 
                        name='net_layer2B_{}'.format(n))(net_layer2A)
                net_layer2C = Dense(units=128, activation='silu', 
                        name='net_layer2C_{}'.format(n))(net_layer2B)
                net_layer2D = Dense(units=len(nb_indices),#_NC2,#100, 
                        activation='silu', 
                        name='net_layer2D_{}'.format(n))(net_layer2C)

                combineQ_layer = CombineQ(all_indices, 
                        name='combineQ_layer')([net_layer1D, net_layer2D])

                #conv_layer = tf.keras.layers.multiply(
                        #[net_layer1D, net_layer2D], 
                        #name='conv_layer_{}'.format(n))

                #trans_layer3 = TransposeQ(_NC2, 
                        #name='trans_layer3_{}'.format(n))(conv_layer)

                #net_layer3A = Dense(units=_NC2, activation='silu', 
                        #name='net_layer3A_{}'.format(n))(conv_layer)

                #sum_layer1 = SumQ(_NC2, 
                        #name='sum_layer1_{}'.format(n))(conv_layer)
                #sum_layer2 = SumQ(_NC2, 
                        #name='sum_layer2_{}'.format(n))(sum_layer1)

                NRF_layer = combineQ_layer

                #NRF_layer = tf.keras.layers.add([NRF_layer, sum_layer2], 
                    #name='NRF_layer_{}'.format(n))

                #one_hot_layer = tf.keras.layers.add([one_hot_layer, sum_layer2], 
                    #name='one_hot_layer_{}'.format(n))
            #'''




            net_layer4A = Dense(units=_NC2+extra_cols, activation='silu', 
                    name='net_layer4A')(NRF_layer)#(one_hot_layer)
            #net_layer4B = Dense(units=_NC2, activation='sigmoid', 
                    #name='net_layer4B')(net_layer4A)


            '''
            NRF_layer2 = TransformNRF(n_atoms, _NC2, 
                    name='NRF_layer2')(NRF_layer)
            net_layer1A = Dense(units=728, activation='silu', 
                    name='net_layer1A')(NRF_layer2)
            net_layer1B = Dense(units=128, activation='silu', 
                    name='net_layer1B')(net_layer1A)
            net_layer1C = Dense(units=128, activation='silu', 
                    name='net_layer1C')(net_layer1B)
            net_layer1D = Dense(units=64, activation='silu', 
                    name='net_layer1D')(net_layer1C)
            net_layer1E = Dense(units=_NC2, activation='silu', 
                    name='net_layer1E')(net_layer1D)
            '''


            '''
            trans_layer = TransposeQ(_NC2, name='net_layer1')(net_layer1E)

            net_layer2A = Dense(units=728, activation='silu', 
                    name='net_layer2A')(trans_layer)
            net_layer2B = Dense(units=128, activation='silu', 
                    name='net_layer2B')(net_layer2A)
            net_layer2C = Dense(units=128, activation='silu', 
                    name='net_layer2C')(net_layer2B)
            net_layer2D = Dense(units=64, activation='silu', 
                    name='net_layer2D')(net_layer2C)
            net_layer2E = Dense(units=n_atoms, activation='linear', 
                    name='net_layer2E')(net_layer2D)

            trans_layer2 = TransposeQ(_NC2, name='trans_layer2')(net_layer2E)

            net_layer2 = SumQ(_NC2, name='net_layer2')(trans_layer2)
            '''

            #sum_layer1 = SumQ(_NC2, 
                    #name='sum_layer1')(net_layer4A)


            '''
            net_layers = []
            for x in range(n_atoms):
                net_layerA = Dense(units=728, activation='silu', 
                        name='net_layerA_{}'.format(x))(NRF_splits[x])
                net_layerB = Dense(units=128, activation='silu', 
                        name='net_layerB_{}'.format(x))(net_layerA)
                net_layerC = Dense(units=128, activation='silu', 
                        name='net_layerC_{}'.format(x))(net_layerB)
                net_layerD = Dense(units=64, activation='silu', 
                        name='net_layerD_{}'.format(x))(net_layerC)
                net_layerE = Dense(units=_NC2, activation='sigmoid', 
                        name='net_layerE_{}'.format(x))(net_layerD)
                net_layers.append(net_layerE)
            #net_layer2 = concatenate(net_layers, axis=1, name='net_layer2')
            net_layer2 = tf.keras.layers.add(net_layers, 
                    name='net_layer2')
            '''

            '''
            net_layer1A = Dense(units=128, activation='silu',
                    name='net_layer1A')(NRF_layer)
            net_layer1B = Dense(units=128, activation='silu',
                    name='net_layer1B')(net_layer1A)
            net_layer1C = Dense(units=64, activation='silu',
                    name='net_layer1C')(net_layer1B)
            qF_layer = Dense(units=_NC2+extra_cols, activation='sigmoid',
                    name='qF_layer')(net_layer1C)
            lin_layer = Linear(_NC2, units=_NC2+extra_cols, 
                    input_dim=_NC2+extra_cols, name='lin_layer')(qF_layer)
            '''

            unscale_qFE_layer = UnscaleQ(_NC2, max_matFE,#_all, 
                    name='unscale_qF_layer')(net_layer4A)#(NRF_layer) #(sum_layer1) #
            #F_layer = Recomposition(n_atoms, _NC2,
                    #name='F_layer')([coords_layer, unscale_qF_layer])
            #qFE_layer = FE_Decomposition(n_atoms, _NC2, activation='silu',
                    #name='qFE_layer')([coords_layer, F_layer])
            #scale_qFE_layer = ScaleQ(_NC2, max_matF, 
                    #name='scale_qFE_layer')(qFE_layer)
            E_layer = E_Recomposition(n_atoms, _NC2,
                    name='qFE_layer')([coords_layer, unscale_qFE_layer])
            scaleE_layer = UnscaleE(prescale, 
                    name='unscaleE_layer')(E_layer)
            dE_dx = EnergyGradient(n_atoms, _NC2, 
                    name='dE_dx')([scaleE_layer, coords_layer])

            rlrop = ReduceLROnPlateau(monitor=monitor_loss, factor=0.5, 
                    patience=50, min_lr=1e-4)
            print('rlrop: monitor={}, factor=0.5, patience=50, '\
                    'min_lr=1e-4'.format(monitor_loss))
            optimizer = keras.optimizers.Adam(lr=0.001, #0.001,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)


            model = Model(
                    inputs=[coords_layer], 
                    outputs=[
                        dE_dx,
                        unscale_qFE_layer,
                        scaleE_layer,
                        ], 
                    )


            if l == 0:
                model.compile(
                        loss={
                            'energy_gradient': 'mse',
                            'unscale_q': 'mse',
                            'unscale_e': 'mse',
                            },
                        loss_weights={
                            'energy_gradient': grad_loss[l],
                            'unscale_q': qFE_loss[l],
                            'unscale_e': E_loss[l],
                            },
                        #optimizer='adam',
                        optimizer=optimizer,
                        )

            model.summary()
            print(model(train2_input_coords[0:2]))

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
                result = model.fit(train2_input_coords, 
                        [
                            train2_forces, 
                            train2_output_matFE, 
                            train2_output_E_postscale, 
                            ],
                        validation_data=(val_input_coords, 
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


                best_error = model.evaluate(train2_input_coords, 
                        [
                            train2_forces, 
                            train2_output_matFE, 
                            train2_output_E_postscale, 
                            ],
                        verbose=0)
                print(l, 'model error train2: ', best_error)

                best_error2 = model.evaluate(val_input_coords, 
                        [
                            val_forces, 
                            val_output_matFE, 
                            val_output_E_postscale, 
                            ],
                        verbose=0)
                print(l, 'model error val: ', best_error2)

                best_error3 = model.evaluate(train_input_coords, 
                        [
                            train_forces, 
                            train_output_matFE, 
                            train_output_E_postscale, 
                            ],
                        verbose=0)
                print(l, 'model error train: ', best_error3)

                best_error4 = model.evaluate(test_input_coords, 
                        [
                            test_forces, 
                            test_output_matFE, 
                            test_output_E_postscale, 
                            ],
                        verbose=0)
                print(l, 'model error test: ', best_error4)

                best_error5 = model.evaluate(input_coords, 
                        [
                            output_F, 
                            output_matFE, 
                            output_E_postscale, 
                            ],
                        verbose=0)
                print(l, 'model error all: ', best_error5)

                model.save_weights('best_ever_model_'+l2)

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
            #model_file='best_model1'#'../model/best_ever_model_6'
            model_file='best_ever_model_0'#'../model/best_ever_model_6'
            #model_file='best_model'#'../model/best_ever_model_6'
            print('load_weights', load_weights, model_file)
            model.load_weights(model_file)
            #model.load_weights('best_model')

        if fit:
            best_error = model.evaluate(train_input_coords, 
                    [
                        train_forces, 
                        train_output_matFE,
                        train_output_E_postscale,
                        ], verbose=0)
            print(l, len(train_input_coords), 'train model error: ', 
                    best_error)


            sys.stdout.flush()
            #sys.exit()


            for i in range(3):
                print('\n\t', i)
                print('NRFs\n', train2_input_NRF[i])
                print('coords\n', train2_input_coords[i])
                print('energy\n', train2_output_E_postscale[i])
                print('E\n', train2_output_E[i])
                print('forces\n', train2_forces[i])
                print('mat_FE\n', train2_output_matFE[i])
                print('mat_F\n', train2_output_matF[i])
                print('mat_E\n', train2_output_matE[i])
                prediction = model.predict(
                        train2_input_coords[i].reshape(1,n_atoms,3)
                        #train2_input_NRF[i].reshape(1,_NC2)
                        )
                print('prediction')
                for count, p in enumerate(prediction):
                    print(count, p)


                x1 = tf.constant(train2_input_coords[i].reshape(1,n_atoms,3), 
                        dtype=tf.float32)
                with tf.GradientTape() as tape2:
                    tape2.watch(x1)
                    y = model(x1) #[0]
                print('y\n', y[2])
                gradient1 = tape2.gradient(y[2], x1) * -1
                print('tape_grad\n', gradient1)
                print(tf.keras.backend.get_value(gradient1))

                variance, translations, rotations = Network.check_invariance(
                        train2_input_coords[i], prediction[0][0],
                        )
                print('variance\n', variance, translations, rotations)

                sys.stdout.flush()



        #inp_out_pred = True
        if inp_out_pred:

            print('\nShow layer weights')
            lay = 0
            for layer in model.layers:
                weights = layer.get_weights()
                print(layer.name, model.layers[lay].weights)
                lay += 1
                print()

            print('\nPredict all data')
            prediction = model.predict(input_coords)
            prediction_E = prediction[2].flatten()
            #prediction_matF = prediction[1]
            #prediction_matE = prediction[2]
            prediction_matFE = prediction[1]

            x_tensor = tf.constant(input_coords, dtype=tf.float32)
            with tf.GradientTape(watch_accessed_variables=False) as tape3:
                tape3.watch(x_tensor)
                y = model(x_tensor)[0]
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


            print('Predict test data')
            test_prediction = model.predict(test_input_coords)
            test_prediction_E = test_prediction[2].flatten()
            #test_prediction_E = test_prediction[1].flatten()
            test_prediction_F = test_prediction[0]
            #test_prediction_matF = test_prediction[1]
            #test_prediction_matE = test_prediction[2]
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



            mae, rms, msd = Binner.get_error(test_output_E_postscale.flatten(), 
                    test_prediction_E.flatten())
            print('\n{} E Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(len(test_output_E_postscale), 
                    mae, rms, msd))

            mae, rms, msd = Binner.get_error(test_output_F.flatten(), 
                    test_prediction_F.flatten())
            print('\n{} grad F Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(len(test_output_F), mae, rms, msd))

            '''
            mae, rms, msd = Binner.get_error(test_output_matF.flatten(), 
                    test_prediction_matF.flatten())
            print('\n{} mat_F Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(
                    len(test_output_matF), mae, rms, msd))

            mae, rms, msd = Binner.get_error(test_output_matE.flatten(), 
                    test_prediction_matE.flatten())
            print('\n{} mat_E Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(
                    len(test_output_matE), mae, rms, msd))
            '''

            mae, rms, msd = Binner.get_error(test_output_matFE.flatten(), 
                    test_prediction_matFE.flatten())
            print('\n{} mat_FE Test MAE: {} \nTest RMS: {} '\
                    '\nTest MSD: {}'.format(
                    len(test_output_matFE), mae, rms, msd))




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
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'all-s-curves-E.pdf', log=True)

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
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'all-s-curves-F.pdf', log=True)

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
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'all-s-curves-matFE.pdf', log=True)
            #'''

            '''
            mae, rms, msd = Binner.get_error(output_atomFE.flatten(), 
                    prediction_atomFE.flatten())
            print('\n{} atom_(F)E All MAE: {} \nAll RMS: {} '\
                    '\nAll MSD: {}'.format(len(output_atomFE), mae, rms, msd))
            bin_edges, hist = Binner.get_scurve(output_atomFE.flatten(), 
                    prediction_atomFE.flatten(), 'all-hist4.txt')
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'all-s-curves-atomFE.pdf')
            '''

            sys.stdout.flush()
            #sys.exit()

            model_loss = result.history['loss']
            model_val_loss = result.history['val_loss']
            Writer.write_csv([model_loss, model_val_loss], 
                    'loss', 'loss,val_loss')
            Plotter.plot_2d([list(range(len(model_loss[10:]))), 
                list(range(len(model_val_loss[10:])))], 
                [model_loss[10:], model_val_loss[10:]], 
                    ['loss', 'val_loss'], 
                    'Epoch', 'Loss', 
                    'loss_curves.png')



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
