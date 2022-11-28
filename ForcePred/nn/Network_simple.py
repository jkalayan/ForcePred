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

start_time = time.time()

#tf.compat.v1.disable_eager_execution()

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

        r_flat = Projection(self.n_atoms, self._NC2)(coords)
        recip_r_flat = 1 / r_flat
        norm_recip_r = tf.reduce_sum(recip_r_flat ** 2, axis=1, 
                keepdims=True) ** 0.5
        eij_E = recip_r_flat / norm_recip_r
        #eij_FE = tf.concat([eij_F, eij_E], axis=1)

        recompE = tf.einsum('bi, bi -> b', eij_E, decompFE) # b,NC2 b,NC2

        ##summing decompFE instead
        #recompE = tf.reduce_sum(decompFE, axis=1, keepdims=True)

        recompE = tf.reshape(recompE, shape=(tf.shape(coords)[0], 1)) # b,1

        return recompE


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

        return r_flat



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


    def get_coord_FE_model(self, molecule, prescale1, n_nodes=1000, 
            n_layers=1, grad_loss_w=1000, qFE_loss_w=1, E_loss_w=1):
        '''Input coordinates and z_types into model to get NRFS which then 
        are used to predict decompFE, which are then recomposed to give
        Cart Fs and molecular E, both of which could be used in the loss
        function, could weight the E or Fs as required.
        '''

        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms*(n_atoms-1)/2)
        extra_cols = 0 #n_atoms #1 ## !!!!!!!!
        #print('!!!!! extra_cols:', extra_cols)
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

            print('N train {}\nN val {}\nN test {}'.format(
                    len(molecule.train), len(molecule.val), 
                    len(molecule.test)))

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
            max_matF1 = np.max(np.abs(train_output_matF))
            max_matE1 = np.max(np.abs(train_output_matE))

            train_input_NRF_scaled = np.take(input_NRF, molecule.train, axis=0)\
                    / max_NRF1
            train_output_matFE_scaled = train_output_matFE / (2 * max_matFE1) + 0.5
            train_output_matF_scaled = train_output_matF / (2 * max_matF1) + 0.5
            train_output_matFE_scaled = train_output_matE / (2 * max_matE1) + 0.5
            print(train_input_coords.shape, train_output_FE.shape)
            print('max_NRF: {:.3f}, max_qFE: {:.3f}, max_E: {}'.format(
                    max_NRF1, max_matFE1, max_E1))
            max_NRF = tf.constant(max_NRF1, dtype=tf.float32)
            max_E = tf.constant(max_E1, dtype=tf.float32)
            max_matFE = tf.constant(max_matFE1, dtype=tf.float32)
            max_matF = tf.constant(max_matF1, dtype=tf.float32)
            max_matE = tf.constant(max_matE1, dtype=tf.float32)
            prescale = tf.constant(prescale1, dtype=tf.float32)
            n_atoms_tf = tf.constant(n_atoms, dtype=tf.int32)


            train_output_E_postscale = ((train_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])
            test_output_E_postscale = ((test_output_E - prescale1[2]) / 
                    (prescale1[3] - prescale1[2]) * 
                    (prescale1[1] - prescale1[0]) + prescale1[0])



            ###get validation from within training
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



        file_name='model/best_model'
        monitor_loss='val_loss'
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


        #grad_loss_w = 1000
        #qFE_loss_w = 1
        #E_loss_w = 1
        best_error_all = 1e20
        best_model = None
        batch_size = 32
        print('batch_size:', batch_size)

        print('dE_dx (gradient/F) loss weight: {} '\
                '\ndecompFE loss weight: {} '\
                '\nE loss weight: {}'.format(
                grad_loss_w, qFE_loss_w, E_loss_w))

        coords_layer = Input(shape=(n_atoms,3), name='coords_layer')
        NRF_layer = CoordsToNRF(atoms_flat, max_NRF, _NC2, n_atoms, 
                name='NRF_layer')(coords_layer)
        connected_layer = NRF_layer
        for l in range(n_layers):
            net_layer = Dense(units=n_nodes, activation='silu', 
                    name='net_laye_rA{}'.format(l))(connected_layer)
            connected_layer = net_layer
        net_layer2 = Dense(units=_NC2, activation='sigmoid', 
                name='net_layerB1')(connected_layer)
        unscale_qFE_layer = UnscaleQ(_NC2, max_matFE, 
                name='unscale_qF_layer')(net_layer2)
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
        optimizer = keras.optimizers.Adam(lr=0.001,
                beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)


        model = Model(
                inputs=[coords_layer], 
                outputs=[
                    dE_dx,
                    unscale_qFE_layer,
                    scaleE_layer,
                    ], 
                )


        model.compile(
                loss={
                    'energy_gradient': 'mse',
                    'unscale_q': 'mse',
                    'unscale_e': 'mse',
                    },
                loss_weights={
                    'energy_gradient': grad_loss_w,
                    'unscale_q': qFE_loss_w,
                    'unscale_e': E_loss_w,
                    },
                #optimizer='adam',
                optimizer=optimizer,
                )

        model.summary()

        print('initial learning rate:', K.eval(model.optimizer.lr))

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
                    epochs=100,#0000, 
                    verbose=2,
                    batch_size=batch_size,
                    #callbacks=[mc],
                    #callbacks=[es,mc],
                    callbacks=[es,mc,rlrop],
                    )


            best_error4 = model.evaluate(test_input_coords, 
                    [
                        test_forces, 
                        test_output_matFE, 
                        test_output_E_postscale, 
                        ],
                    verbose=0)
            print('model error test: ', best_error4)


            model.save_weights('model/best_ever_model')
        print()



        #load_weights = False
        if load_weights:
            model_file='model/best_ever_model'
            print('load_weights', load_weights, model_file)
            model.load_weights(model_file)


        #inp_out_pred = True
        if inp_out_pred:

            prediction = model.predict(input_coords)
            prediction_E = prediction[2].flatten()
            prediction_matFE = prediction[1]

            Writer.write_csv([output_E_postscale, prediction_E], 
                    'all_energies', 'actual_E,prediction_E')

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
            test_prediction = model.predict(test_input_coords)
            test_prediction_E = test_prediction[2].flatten()
            test_prediction_F = test_prediction[0]
            test_prediction_matFE = test_prediction[1]

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


            model_loss = result.history['loss']
            model_val_loss = result.history['val_loss']
            Writer.write_csv([model_loss, model_val_loss], 
                    'loss', 'loss val_loss', delimiter=' ', ext='dat')
            '''
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


