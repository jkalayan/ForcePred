#!/usr/bin/env python

'''
This module is for running a NN with a training set of data.
'''
from __future__ import print_function #for tf printing
import numpy as np
from keras.layers import Input, Dense, Lambda, concatenate, Layer, \
        initializers, Add, Multiply
from keras.models import Model, load_model                                   
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K                                              
import tensorflow as tf
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

'''
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)
'''


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


    def print_tensor(a):
        with tf.Session() as sess: #to get tensor to numpy
            sess.run(tf.global_variables_initializer())
            x = sess.run(a)
            print(x)

    def return_tensor(a):
        with tf.Session() as sess: #to get tensor to numpy
            sess.run(tf.global_variables_initializer())
            a = sess.run(a)
        return a


    @tf.function
    def get_FE_eij_matrix(decompFE, coords, atoms, n_atoms, _NC2):
        '''2. From network decomposed FEs, need an eij matrix
        to recompose decompFE to 3N CartF + molecular E within the network
        '''
        atoms = tf.convert_to_tensor(atoms, dtype=tf.float32)
        coords = tf.reshape(coords, [-1,n_atoms,3])
        n_structures = 1 #coords.shape[0]
        i = tf.constant(0)
        j = tf.constant(0)
        _N = tf.constant(-1)
        s = tf.constant(0)
        eij = tf.zeros((n_structures,n_atoms*3+1,_NC2))

        def cond3(i, j, _N, x, s, eij, r):
            return tf.less(x, 3)

        def body3(i, j, _N, x, s, eij, r):
            val = tf.divide(tf.subtract(coords[s][i][x], coords[s][j][x]), r)
            minus_val = tf.negative(val)

            index = [[s,i*3+x,_N]]
            value = [val]
            shape = [n_structures,n_atoms*3+1,_NC2]
            delta = tf.SparseTensor(index, value, shape)
            result = eij + tf.sparse_tensor_to_dense(delta)
            eij = result

            index = [[s,j*3+x,_N]]
            value = [minus_val]
            shape = [n_structures,n_atoms*3+1,_NC2]
            delta = tf.SparseTensor(index, value, shape)
            result = eij + tf.sparse_tensor_to_dense(delta)
            eij = result

            x = tf.add(x, 1)
            return i, j, _N, x, s, eij, r

        def cond2(i, j, _N, s, eij):
            return tf.less(j, i)

        def body2(i, j, _N, s, eij):
            _N = tf.add(_N, 1)
            x = tf.constant(0)
            r = tf.linalg.norm(tf.math.subtract(coords[s][i], coords[s][j]))
            #for forces
            i, j, _N, x, s, eij, r = tf.while_loop(cond3, body3, 
                    [i, j, _N, x, s, eij, r])
            #for energies
            recip_r = tf.reciprocal(r)
            index = [[s,n_atoms*3,_N]]
            value = [recip_r]
            shape = [n_structures,n_atoms*3+1,_NC2]
            delta = tf.SparseTensor(index, value, shape)
            result = eij + tf.sparse_tensor_to_dense(delta)
            eij = result

            j = tf.add(j, 1)
            return i, j, _N, s, eij

        def cond1(i, j, _N, s, eij):
            return tf.less(i, n_atoms)

        def body1(i, j, _N, s, eij):
            i, j, _N, s, eij = tf.while_loop(cond2, body2, 
                    [i, j, _N, s, eij])
            i = tf.add(i, 1)
            j = tf.constant(0)
            return [i, j, _N, s, eij]

        def cond(i, j, _N, s, eij):
            return tf.less(s, n_structures)

        def body(i, j, _N, s, eij):
            i, j, _N, s, eij = tf.while_loop(cond1, body1, 
                    [i, j, _N, s, eij])
            s = tf.add(s, 1)
            i = tf.constant(0)
            _N = tf.constant(-1)
            return [i, j, _N, s, eij]

        res = tf.while_loop(cond, body, [i, j, _N, s, eij])
        eij = tf.reshape(res[4], [n_structures,n_atoms*3+1,_NC2])
        decompFE = tf.reshape(decompFE, [n_structures,_NC2])
        recompFE = tf.einsum("ijk, ik -> ij", eij, decompFE)
        #return recompFE.eval(session=tf. compat. v1. Session())
        return recompFE



    @tf.function
    def get_NRF_from_coords(coords, atoms, n_atoms, _NC2):
        '''1. Use tensors to get NRF from coords, can't use for loops in tf,
        so have to use nested while loops to iterate over i and j atoms.
        '''
        atoms = tf.convert_to_tensor(atoms, dtype=tf.float32)
        coords = tf.reshape(coords, [-1,n_atoms,3])
        n_structures = 1 #coords.shape[0]
        i = tf.constant(0)
        j = tf.constant(0)
        _N = tf.constant(-1)
        s = tf.constant(0)
        _NRF = tf.zeros((n_structures,_NC2))

        def cond3(i, j, _N, s, _NRF):
            return tf.less(j, i)

        def body3(i, j, _N, s, _NRF):
            _N = tf.add(_N, 1)

            r = tf.linalg.norm(tf.math.subtract(coords[s][i], coords[s][j]))
            zizj = tf.multiply(atoms[i], atoms[j])
            zizj = tf.multiply(zizj, Converter.au2kcalmola)
            ij_NRF = tf.divide(zizj, tf.square(r))

            index = [[s,_N]]
            value = [ij_NRF]
            shape = [n_structures,_NC2]
            delta = tf.SparseTensor(index, value, shape)
            result = _NRF + tf.sparse_tensor_to_dense(delta)
            _NRF = result

            j = tf.add(j, 1)
            return i, j, _N, s, _NRF

        def cond2(i, j, _N, s, _NRF):
            return tf.less(i, n_atoms)

        def body2(i, j, _N, s, _NRF):
            i, j, _N, s, _NRF = tf.while_loop(cond3, body3, 
                    [i, j, _N, s, _NRF])
            i = tf.add(i, 1)
            j = tf.constant(0)
            return i, j, _N, s, _NRF

        def cond1(i, j, _N, s, _NRF):
            return tf.less(s, n_structures)

        def body1(i, j, _N, s, _NRF):
            i, j, _N, s, _NRF = tf.while_loop(cond2, body2, 
                    [i, j, _N, s, _NRF])
            s = tf.add(s, 1)
            i = tf.constant(0)
            _N = tf.constant(-1)
            return [i, j, _N, s, _NRF]

        res = tf.while_loop(cond1, body1, [i, j, _N, s, _NRF])
        res = tf.reshape(res[4], [n_structures,_NC2])
        return res


    def get_coord_FE_model(self, molecule):
        '''Input coordinates and z_types into model to get NRFS which then 
        are used to predict decompFE, which are then recomposed to give
        Csrt Fs and molecular E, both of which could be used in the loss
        function, could weight the E or Fs as required.
        '''
        n_atoms = len(molecule.atoms)
        #n_structures = len(molecule.coords)
        #print('n_structures', n_structures)
        _NC2 = int(n_atoms*(n_atoms-1)/2)
        atoms = np.array([float(i) for i in molecule.atoms], dtype='float32')
        #atoms = tf.constant(np.array([float(i) for i in molecule.atoms]),
                #dtype=tf.float32)
        coords = tf.constant(np.copy(molecule.coords[0]), dtype=tf.float32)
        _NRF = Network.get_NRF_from_coords(tf.reshape(coords, 
                [-1,3*n_atoms]), atoms, n_atoms, _NC2)
        print('tf NRF:')
        Network.print_tensor(_NRF[0])
        print(_NRF.shape)
        print('NRF:', molecule.mat_NRF[0])

        decompFE = tf.constant(molecule.mat_FE[0], dtype=tf.float32)
        recompFE = Network.get_FE_eij_matrix(tf.reshape(decompFE, 
                [-1,_NC2]), tf.reshape(coords, 
                [-1,3*n_atoms]), atoms, n_atoms, _NC2)
        print('tf recompFE:')
        Network.print_tensor(recompFE[0])
        print(recompFE.shape)
        print()
        sys.stdout.flush()

        sys.exit()


        end=1
        input = molecule.coords[0:end].reshape(-1,n_atoms*3)
        output = np.concatenate((molecule.forces[0:end].reshape(-1,n_atoms*3), 
                molecule.energies[0:end].reshape(-1,1)), axis=1)
        #n_structures = len(input)
        print(input.shape, output.shape)

        mc = ModelCheckpoint(
                'model.h5', 
                monitor='loss', mode='min', 
                save_best_only=True)
        es = EarlyStopping(monitor='loss', patience=500)

        coords_layer = Input(shape=(n_atoms*3,), name='coords_layer')
        NRF_layer = Lambda(Network.get_NRF_from_coords, 
                input_shape=(n_atoms*3,), output_shape=(_NC2,), 
                name='NRF_layer', arguments={'atoms':atoms, 
                'n_atoms':n_atoms, '_NC2':_NC2})(coords_layer)
        net_layer = Dense(units=1000, activation='relu', 
                name='net_layer')(NRF_layer)
        decomp_layer = Dense(units=_NC2, activation='linear', 
                name='decomp_layer')(net_layer)
        recompFE_layer = Lambda(Network.get_FE_eij_matrix,
                input_shape=(_NC2,), output_shape=(n_atoms*3+1,),
                name='recompFE_layer', arguments={'coords':coords_layer, 
                'atoms':atoms, 'n_atoms':n_atoms, '_NC2':_NC2})(decomp_layer)
        model = Model(coords_layer, recompFE_layer)
        model.compile(loss='mse', optimizer='adam', 
                metrics=['mae', 'acc']) #mean abs error, accuracy
        model.summary()
        sys.stdout.flush()


        model.fit(input, output, epochs=1000, verbose=2,
                #callbacks=[es],
                #callbacks=[es,mc],
                )

        #model = load_model('model.h5')
        print('train', model.evaluate(input, output, verbose=2))

        pred_input = molecule.coords[0].reshape(-1,n_atoms*3)
        print(pred_input.shape) 
        prediction = model.predict(pred_input)
        print('\npred', prediction)
        print('\nactual', molecule.forces[0], molecule.energies[0])


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

        train_net = False
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
                        verbose=2, callbacks=[mc,es])

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


    def pairwise_NN(scaled_input, scaled_output, nodes, _NC2):
        '''For each pairwise NRF input, create a 1000 node dense layer and
        then concat these models back into one.'''
        print('scaled_input shape:', scaled_input.shape)
        scaled_input = scaled_input.T
        print('scaled_input shape:', scaled_input.shape)
        scaled_output = scaled_output.T

        file_name = 'best_ever_model'
        mc = ModelCheckpoint(file_name, monitor='loss', 
                mode='min', save_best_only=True)
        es = EarlyStopping(monitor='loss', patience=500)
        n_nodes = nodes #10 #1000
        n_epochs = 1000 #100000
        print('n_nodes: {}\nn_epochs: {}'.format(n_nodes, n_epochs))

        new_scaled_input = []
        new_scaled_output = []
        model_input = []
        model_output = []
        for i in range(_NC2):
            model_in = Input(shape=(1,))
            net = Dense(units=n_nodes, activation='sigmoid')(model_in)
            net = Dense(units=1, activation='sigmoid')(net)
            model_in = Model(model_in, net)
            model_in.summary()
            new_scaled_input.append(scaled_input[i])
            new_scaled_output.append(scaled_output[i])
            model_input.append(model_in.input)
            model_output.append(model_in.output)

        #model_out = concatenate(model_output)
        #model_out = Dense(units=_NC2, activation='linear')(model_out)
        model = Model(inputs=model_input, outputs=model_output)
        model.compile(loss='mse', optimizer='adam', 
            metrics=['mae', 'acc']) #mean abs error, accuracy
        model.summary()
        model.fit(new_scaled_input, new_scaled_output, epochs=n_epochs, 
                verbose=2, callbacks=[es,mc])

        model = load_model(file_name)
        print('train', model.evaluate(new_scaled_input, new_scaled_output, 
                verbose=2))
        train_prediction_scaled = model.predict(new_scaled_input)
        train_prediction_scaled = np.array(
                train_prediction_scaled).reshape(-1,_NC2)
        #print('train_prediction_scaled', train_prediction_scaled)
        #train_prediction_scaled = train_prediction_scaled[0].reshape(-1,_NC2)
        print('train_prediction_scaled shape', train_prediction_scaled.shape)
        #print('scaled_output', scaled_output.T, scaled_output.T.shape)

        print()
        return train_prediction_scaled


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


        

