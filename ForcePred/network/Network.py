#!/usr/bin/env python

'''
This module is for running a NN with a training set of data.
'''

import numpy as np
from keras.layers import Input, Dense, concatenate, Layer, initializers, Add  
from keras.models import Model, load_model                                   
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K                                              
import tensorflow as tf
from ..calculate.MM import MM
from ..calculate.Converter import Converter
from ..write.Writer import Writer
from ..read.Molecule import Molecule
import sys
import time

class Network(object):
    '''
    '''
    def __init__(self):
        self.model = None

    def get_variable_depth_model(self, molecule, input, output):
        start_time = time.time()
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        train_input = input
        train_output = output
        #np.savetxt('input.txt', input)
        #np.savetxt('output.txt', output)
        train = True #False
        if train:
            train_input = np.take(input, molecule.train, axis=0)
            train_output = np.take(output, molecule.train, axis=0)
            #np.savetxt('trainset-input.txt', train_input)
            #np.savetxt('trainset-output.txt', train_output)
            #np.savetxt('train_indices.txt', molecule.train)

        scaled_input, self.scale_input_max, self.scale_input_min = \
                Network.get_scaled_values(train_input)
        scaled_output, self.scale_output_max, self.scale_output_min = \
                Network.get_scaled_values(train_output)

        #np.savetxt('scaled_input.txt', scaled_input)
        #np.savetxt('scaled_output.txt', scaled_output)
        print('input shape: {}'.format(scaled_input.shape))
        print('output shape: {}'.format(scaled_output.shape))
        max_depth = 1 #6
        file_name = 'best_model'
        mc = ModelCheckpoint(file_name, monitor='loss', 
                mode='min', save_best_only=True)
        es = EarlyStopping(monitor='loss', patience=500)
        model_in = Input(shape=(scaled_input.shape[1],))
        model = model_in
        best_error = 1000
        n_nodes = 10 #1000
        n_epochs = 100000 #100000
        print('max depth: {}\nn_nodes: {}\nn_epochs: {}'.format(
                max_depth, n_nodes, n_epochs))
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
            #end_training = np.loadtxt('end_file', dtype=int)
            #if end_training == 1:
                #break
            if time.time()-start_time >= 518400:
                break
            model = load_model(file_name)
            model.trainable = False
            model = model(model_in)
        self.model = model

        model = load_model('best_ever_model')
        print('train', model.evaluate(scaled_input, scaled_output, verbose=2))
        train_prediction_scaled = model.predict(scaled_input)
        #print('\nscaled_input', scaled_input)
        #print('scaled_output', scaled_output)
        #print('train_predicition_scaled', train_prediction_scaled)
        if np.amax(train_output) > 0 and np.amin(train_output) < 0:
            train_prediction = (train_prediction_scaled - 0.5) * \
                    self.scale_output_max
            #print('train_prediction1', train_prediction)
        else:
            #train_prediction = (train_prediction_scaled) * \
                    #self.scale_output_max
            train_prediction = train_prediction_scaled * \
                    (np.amax(train_output) - np.amin(train_output)) \
                    + np.amin(train_output)
            #if np.amax(train_output) < 0 and np.amin(train_output) < 0:
                #train_prediction = -train_prediction
                #print('train_prediction3', train_prediction)
            #else:
            #print('train_prediction2', train_prediction)
        #np.savetxt('trainset-prediction.txt', train_prediction)

        Writer.write_csv([train_input, train_output, train_prediction], 
                'trainset_inp_out_pred', 'input,output,prediction')

        Network.get_validation(molecule, input, output)

    def get_validation(molecule, input, output):
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        test_input = np.take(input, molecule.test, axis=0)
        test_output = np.take(output, molecule.test, axis=0)
        #np.savetxt('testset-input.txt', test_input)
        #np.savetxt('testset-output.txt', test_output)

        scaled_input_test, scale_input_max_test, scale_input_min_test = \
                Network.get_scaled_values(test_input)
        scaled_output_test, scale_output_max_test, scale_output_min_test = \
                Network.get_scaled_values(test_output)

        model = load_model('best_ever_model')
        print('test', model.evaluate(scaled_input_test, scaled_output_test, 
            verbose=2))
        test_prediction_scaled = model.predict(scaled_input_test)
 
        if np.amax(test_output) > 0 and np.amin(test_output) < 0:
        #if len(output[0]) == _NC2:
            test_prediction = (test_prediction_scaled - 0.5) * \
                    scale_output_max_test
        else:
            #test_prediction = (test_prediction_scaled) * \
                    #np.abs(scale_output_max_test)
            test_prediction = test_prediction_scaled * \
                    (np.amax(test_output) - np.amin(test_output)) \
                    + np.amin(test_output)
            #if np.amax(test_output) < 0 and np.amin(test_output) < 0:
                #test_prediction = -test_prediction
        #np.savetxt('testset-prediction.txt', test_prediction)

        Writer.write_csv([test_input, test_output, test_prediction], 
                'testset_inp_out_pred', 'input,output,prediction')

        get_recomp = False
        if get_recomp:
            coords_test = np.take(molecule.coords, molecule.test, axis=0)
            coords = np.take(molecule.coords, molecule.train, axis=0)
            forces_test = np.take(molecule.forces, molecule.test, axis=0)
            forces = np.take(molecule.forces, molecule.train, axis=0)
            charges_test = np.take(molecule.charges, molecule.test, axis=0)
            charges = np.take(molecule.charges, molecule.train, axis=0)
            #test_recomp_forces = Network.get_recomposed_forces(coords_test, 
                    #test_prediction, n_atoms, _NC2)
            #recomp_forces = Network.get_recomposed_forces(coords, 
                    #train_prediction, n_atoms, _NC2)
            test_recomp_charges = Converter.get_recomposed_charges(
                    coords_test, test_prediction, n_atoms, _NC2)
            recomp_charges = Converter.get_recomposed_charges(coords, 
                    train_prediction, n_atoms, _NC2)
            #Writer.write_xyz(test_recomp_forces, molecule.atoms, 
                #'testset-predicted-forces.xyz', 'w')
            #Writer.write_xyz(recomp_forces, molecule.atoms, 
                #'trainset-predicted-forces.xyz', 'w')
            Writer.write_xyz(coords_test, molecule.atoms, 
                'testset-coords.xyz', 'w')
            #Writer.write_xyz(forces_test, molecule.atoms, 
                #'testset-forces.xyz', 'w')
            Writer.write_xyz(coords, molecule.atoms, 
                'trainset-coords.xyz', 'w')
            #Writer.write_xyz(forces, molecule.atoms, 
                #'trainset-forces.xyz', 'w')      


    def get_scaled_values(values):
        '''normalise values for NN'''
        #print(np.amax(values), np.amin(values))
        if np.amax(values) > 0 and np.amin(values) < 0:
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
            #scaled_values = values / np.amax(np.absolute(values))
            scaled_values = (values - np.amin(values)) / \
                    (np.amax(values) - np.amin(values))
            #if np.amax(values) < 0 and np.amin(values) < 0:
                #scaled_values = -scaled_values
                #print('inp3')
            #else:
                #print('inp2')

        print('scale_max: {}\nscale_min: {}\nnstructures: {}\n'.format(
                scale_max, scale_min, len(values)))

        return scaled_values, scale_max, scale_min


    def get_NRF_input(coords, atoms, n_atoms, _NC2):
        mat_NRF = np.zeros((1, _NC2))
        _N = -1
        for i in range(n_atoms):
            zi = atoms[i]
            for j in range(i):
                _N += 1
                zj = atoms[j]
                r = Converter.get_r(coords[i], coords[j])
                if i != j:
                    mat_NRF[:,_N] = Converter.get_NRF(zi, zj, r)
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

    def get_scaled(a, b, x_min, x_max, x):
        x_norm = (b - a) * (x - x_min) / (x_max - x_min) + a
        return x_norm

    def get_unscaled(a, b, x_min, x_max, x_norm):
        x = (x_norm - a) / (b - a) * (x_max - x_min) + x_min
        return x

    def run_NVE(network, molecule, timestep, nsteps):
        #mm = MM() #initiate MM class

        scale_coords = False
        if scale_coords:
            print('\ncoordinates being scaled by min and max NRFs')

        mm = Molecule()
        mm.atoms = molecule.atoms
        mm.coords = []
        mm.forces = []
        mm.energies = []
        coords_init = molecule.coords[0]
        atoms = molecule.atoms
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)

        scale_NRF = network.scale_NRF #29.10017376084781# 
        scale_NRF_min = network.scale_NRF_min # 0.03589677731052864 #
        scale_NRF_all = 0 #network.scale_NRF_all #
        scale_NRF_min_all = 0 #network.scale_NRF_min_all #
        scale_F = network.scale_F # 
        scale_F_min = network.scale_F_min #0 #
        #scale_F_all = network.scale_F_all 
        #scale_F_min_all = network.scale_F_min_all 
        #print(scale_NRF, scale_NRF_min, scale_F)
        print('scale_NRF: {}\nscale_NRF_min: {}\nscale_F: {}\n'\
                'scale_F_min: {}\n'.format(
                scale_NRF, scale_NRF_min, scale_F, scale_F_min))
        #mm.coords.append(coords_init)
        masses = np.zeros((n_atoms,3))
        for i in range(n_atoms):
            masses[i,:] = Converter._ZM[atoms[i]]
        #model = network.model
        model = load_model('best_ever_model')
        open('nn-coords.xyz', 'w').close()
        open('nn-forces.xyz', 'w').close()
        f1 = open('nn-NRF.txt', 'ab')
        f2 = open('nn-decomp-force.txt', 'ab')
        #f3 = open('nn-E.txt', 'ab')
        open('nn-E.txt', 'w').close()
        open('nn-KE.txt', 'w').close()
        open('nn-T.txt', 'w').close()
        #coords_prev = coords_init
        coords_prev = Converter.translate_coords(
                coords_init, atoms)
        coords_current = coords_init
        _E_prev = 0
        temp = 2
        dt = 0.5
        for i in range(nsteps):
            #print('\t', i)
            #print('NVE coords', coords_current.shape, coords_current)

            #translate coords back to center
            #coords_current = Converter.translate_coords(
                    #coords_current, atoms)

            mat_NRF = Network.get_NRF_input(coords_current, atoms, 
                    n_atoms, _NC2)
            mat_NRF_scaled = mat_NRF / scale_NRF
            #mat_NRF_scaled = Network.get_scaled(0, 1, scale_NRF_min, 
                    #scale_NRF, mat_NRF)
            
            #scale NRF to not go beyond max and min
            if scale_coords:
                #if np.any(np.greater(mat_NRF, scale_NRF)) or \
                        #np.any(np.less(mat_NRF, scale_NRF_min)):
                if np.any(np.greater(mat_NRF, scale_NRF_all)) or \
                        np.any(np.less(mat_NRF, scale_NRF_min_all)):
                    #print(mat_NRF_scaled)
                    #print(mat_NRF)
                    #print()
                    #rs = Converter.get_r_from_NRF(mat_NRF[0], atoms)
                    #print('\n', rs)
                    #coords_current = Converter.get_coords_from_NRF(
                            #mat_NRF[0], molecule.atoms, 
                            #coords_current, scale_NRF, scale_NRF_min)
                    coords_current = Converter.get_coords_from_NRF(
                            mat_NRF[0], molecule.atoms, 
                            coords_current, scale_NRF_all, scale_NRF_min_all)
                    mat_NRF = Network.get_NRF_input(coords_current, 
                            atoms, n_atoms, _NC2)
                    #rs = Converter.get_r_from_NRF(mat_NRF[0], atoms)
                    #print('\n', rs)
                    mat_NRF_scaled = mat_NRF / scale_NRF
                    #mat_NRF_scaled = Network.get_scaled(0, 1, scale_NRF_min, 
                            #scale_NRF, mat_NRF)
                    #print(mat_NRF_scaled)
                    sys.stdout.flush()

            prediction_scaled = model.predict(mat_NRF_scaled)
            prediction = (prediction_scaled - 0.5) * scale_F
            #prediction = Network.get_unscaled(0, 1, scale_NRF_min, scale_NRF, 
                    #prediction_scaled - 0.5)

            recomp_forces = Network.get_recomposed_forces([coords_current], 
                    [prediction], n_atoms, _NC2)
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
            #steps = MM.calculate_step(coords_current, coords_prev, 
                    #recomp_forces[0], timestep, masses, timestep)
                    #ismaeel's code
            _E = _E_prev - dE

            #if i%(nsteps/100) == 0 and temp < 1:
                #temp += 0.01
            if i%(nsteps/10000) == 0:
                np.savetxt(f1, mat_NRF)
                np.savetxt(f2, prediction)
                open('nn-E.txt', 'a').write('{}\n'.format(_E))#.close()
                open('nn-KE.txt', 'a').write('{}\n'.format(_KE))#.close()
                open('nn-T.txt', 'a').write('{}\n'.format(current_T))#.close()
                Writer.write_xyz([coords_current], molecule.atoms, 
                    'nn-coords.xyz', 'a', i)
                Writer.write_xyz(recomp_forces, molecule.atoms, 
                    'nn-forces.xyz', 'a', i)
                Writer.write_xyz([v], molecule.atoms, 
                    'nn-velocities.xyz', 'a', i)
                mm.coords.append(coords_current)
                mm.forces.append(recomp_forces)
                mm.energies.append(_E)

            coords_prev = coords_current
            coords_current = coords_next
            _E_prev = _E
            #print()
        return mm


        

