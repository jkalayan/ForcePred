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

class Network(object):
    '''
    '''
    def __init__(self):
        self.scale_NRF = None
        self.scale_F = None
        self.model = None


    def get_variable_depth_model(self, molecule):
        _NRF = molecule.mat_NRF
        _F = molecule.mat_F
        train = True #False
        if train:
            _NRF = np.take(molecule.mat_NRF, molecule.train, axis=0)
            _F = np.take(molecule.mat_F, molecule.train, axis=0)
            np.savetxt('train-actual-decomp-force.txt', _F)
            #np.savetxt('train_indices.txt', molecule.train)
        #sys.exit()
        #normalise inputs and forces
        self.scale_NRF = np.amax(_NRF)
        self.scale_NRF_min = np.amin(_NRF)
        self.scale_NRF_all = np.amax(_NRF, axis=0)
        self.scale_NRF_min_all = np.amin(_NRF, axis=0)
        #print(np.amax(_NRF), np.amin(_NRF))
        #_NRF = _NRF / self.scale_NRF
        _NRF = Network.get_scaled(0, 1, self.scale_NRF_min, 
                self.scale_NRF, _NRF)
        #_NRF = Network.get_scaled(0, 1, self.scale_NRF_min_all, 
                #self.scale_NRF_all, _NRF)

        self.scale_F = 2 * np.amax(np.absolute(_F))
        self.scale_F_min = 2 * np.amin(np.absolute(_F))
        #self.scale_F_all = 2 * np.amax(np.absolute(_F), axis=0)
        #self.scale_F_min_all = 2 * np.amin(np.absolute(_F), axis=0)
        #_F = _F / self.scale_F + 0.5
        _F = Network.get_scaled(0, 1, self.scale_F_min, 
                self.scale_F, _F) + 0.5
        #_F = Network.get_scaled(0, 1, self.scale_F_min_all, 
                #self.scale_F_all, _F) + 0.5
        print('scale_NRF: {}\nscale_NRF_min: {}\nscale_F: {}\n'\
                'scale_F_min: {}\nnstructures: {}\n'.format(self.scale_NRF, 
                self.scale_NRF_min, self.scale_F, self.scale_F_min, 
                len(_NRF)))
        '''
        print('scale_NRF: {}\nscale_NRF_min: {}\nscale_F: {}\n'\
                'scale_F_min: {}\nnstructures: {}\n'.format(
                self.scale_NRF_all, self.scale_NRF_min_all, 
                self.scale_F_all, self.scale_F_min_all, len(_NRF)))
        '''

        max_depth = 6 #6
        file_name = 'best_model'
        mc = ModelCheckpoint(file_name, monitor='loss', 
                mode='min', save_best_only=True)
        es = EarlyStopping(monitor='loss', patience=500)
        model_in = Input(shape=(_NRF.shape[1],))
        model = model_in
        best_error = 1000
        n_nodes = 1000 #1000
        n_epochs = 1000 #100000
        for i in range(0, max_depth):
            if i > 0:
                model = concatenate([model,model_in])
            net = Dense(units=n_nodes, activation='sigmoid')(model)
            net = Dense(units=_NRF.shape[1], activation='sigmoid')(net)         
            model = Model(model_in, net)
            model.compile(loss='mse', optimizer='adam')
            model.summary()
            model.fit(_NRF, _F, epochs=n_epochs, verbose=2, callbacks=[mc,es])
            model = load_model(file_name)
            if model.evaluate(_NRF, _F, verbose=0) < best_error:
                model.save('best_ever_model')
                best_error = model.evaluate(_NRF, _F, verbose=0)
                print('best model was achieved on layer %d' % i)
                print('its error was:')
                print(best_error)
                print()
            #end_training = np.loadtxt('end_file', dtype=int)
            #if end_training == 1:
                #break
            #if time.time()-start_time >= 518400:
                #break
            model = load_model(file_name)
            model.trainable = False
            model = model(model_in)
        self.model = model

        Network.get_validation(molecule, _NRF, _F, self.scale_NRF, 
                self.scale_NRF_min, self.scale_F, self.scale_F_min)

    def get_validation(molecule, _NRF, _F, scale_NRF, scale_NRF_min, 
            scale_F, scale_F_min):
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        test_NRF = np.take(molecule.mat_NRF, molecule.test, axis=0)
        test_F = np.take(molecule.mat_F, molecule.test, axis=0)
        np.savetxt('test-actual-decomp-force.txt', test_F)


        coords_test = np.take(molecule.coords, molecule.test, axis=0)
        coords = np.take(molecule.coords, molecule.train, axis=0)

        forces_test = np.take(molecule.forces, molecule.test, axis=0)
        forces = np.take(molecule.forces, molecule.train, axis=0)

        scale_NRF_test = np.amax(test_NRF)
        scale_NRF_min_test = np.amin(test_NRF)
        test_NRF = Network.get_scaled(0, 1, scale_NRF_min_test, 
                scale_NRF_test, test_NRF)

        scale_F_test = 2 * np.amax(np.absolute(test_F))
        scale_F_min_test = 2 * np.amin(np.absolute(test_F))
        test_F = Network.get_scaled(0, 1, scale_F_min_test, 
                scale_F_test, test_F) + 0.5

        model = load_model('best_ever_model')
        print('test', model.evaluate(test_NRF, test_F, verbose=2))
        print('train', model.evaluate(_NRF, _F, verbose=2))
        test_prediction_scaled = model.predict(test_NRF)
        prediction_scaled = model.predict(_NRF)
        #test_prediction = (test_prediction_scaled - 0.5) * scale_F
        test_prediction = Network.get_unscaled(0, 1, scale_NRF_min_test, 
                scale_NRF_test, test_prediction_scaled - 0.5)
        #train_prediction = (prediction_scaled - 0.5) * scale_F
        train_prediction = Network.get_unscaled(0, 1, scale_NRF_min, 
                scale_NRF, prediction_scaled - 0.5)
        #np.savetxt('test_predictions', test_predictions)
        #np.savetxt('train_predictions', train_predictions)

        test_recomp_forces = Network.get_recomposed_forces(coords_test, 
                test_prediction, n_atoms, _NC2)
        recomp_forces = Network.get_recomposed_forces(coords, 
                train_prediction, n_atoms, _NC2)

        Writer.write_xyz(coords_test, molecule.atoms, 
            'test-coords.xyz', 'w')
        Writer.write_xyz(test_recomp_forces, molecule.atoms, 
            'test-forces.xyz', 'w')
        Writer.write_xyz(forces_test, molecule.atoms, 
            'test-actual-forces.xyz', 'w')
        np.savetxt('test-decomp-force.txt', test_prediction)

        Writer.write_xyz(coords, molecule.atoms, 
            'train-coords.xyz', 'w')
        Writer.write_xyz(recomp_forces, molecule.atoms, 
            'train-forces.xyz', 'w')
        Writer.write_xyz(forces, molecule.atoms, 
            'train-actual-forces.xyz', 'w')      
        np.savetxt('train-decomp-force.txt', train_prediction)

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
        mm = Molecule()
        mm.atoms = molecule.atoms
        coords_init = molecule.coords[0]
        atoms = molecule.atoms
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)

        scale_NRF = network.scale_NRF #13036.561501577185 # 
        scale_NRF_min = network.scale_NRF_min #0.012938465758322835 #
        scale_NRF_all = network.scale_NRF_all 
        scale_NRF_min_all = network.scale_NRF_min_all
        scale_F = network.scale_F #547.4610197022887 # 
        scale_F_min = network.scale_F_min #547.4610197022887 #
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
        f1 = open('nn-NRF-scaled.txt', 'ab')
        f2 = open('nn-decomp-force-scaled.txt', 'ab')
        #f3 = open('nn-E.txt', 'ab')
        coords_prev = coords_init
        coords_current = coords_init
        _E_prev = 0
        for i in range(nsteps):
            #print('\t', i)
            #print('NVE coords', coords_current.shape, coords_current)
            mat_NRF = Network.get_NRF_input(coords_current, atoms, 
                    n_atoms, _NC2)
            #mat_NRF_scaled = mat_NRF / scale_NRF
            mat_NRF_scaled = Network.get_scaled(0, 1, scale_NRF_min, 
                    scale_NRF, mat_NRF)
            
            '''
            #scale NRF to not go beyond max and min
            if np.any((mat_NRF > scale_NRF)) or \
                    np.any((mat_NRF < scale_NRF_min)):
                #print(mat_NRF_scaled)
                #print(mat_NRF)
                #print()
                coords_current = Converter.get_coords_from_NRF(
                        mat_NRF[0], molecule.atoms, 
                        coords_current, scale_NRF_all, scale_NRF_min_all)
                mat_NRF = Network.get_NRF_input(coords_current, 
                        atoms, n_atoms, _NC2)
                mat_NRF_scaled = Network.get_scaled(0, 1, scale_NRF_min, 
                        scale_NRF, mat_NRF)
                #print(mat_NRF_scaled)
            '''

            #open('nn-NRF-scaled.txt', 'a').write(
                    #'{}\n'.format(mat_NRF_scaled))#.close()
            np.savetxt(f1, mat_NRF_scaled)
            prediction_scaled = model.predict(mat_NRF_scaled)
            #open('nn-decomp-force-scaled.txt', 'a').write(
                    #'{}\n'.format(prediction_scaled))#.close()
            np.savetxt(f2, prediction_scaled)

            #prediction = (prediction_scaled - 0.5) * scale_F
            prediction = Network.get_unscaled(0, 1, scale_NRF_min, scale_NRF, 
                    prediction_scaled - 0.5)
            recomp_forces = Network.get_recomposed_forces([coords_current], 
                    [prediction], n_atoms, _NC2)
            #print('NVE recomp', recomp_forces.shape, recomp_forces)

            '''
            mm.coords = mm.get_3D_array([coords_current])
            mm.forces = mm.get_3D_array([recomp_forces])
            mm.check_force_conservation()
            Converter.get_rotated_forces(mm)

            coords_current = mm.rotated_coords[0]
            recomp_forces = mm.rotated_forces[0]
            '''

            coords_next, dE = MM.calculate_verlet_step(coords_current, 
                    coords_prev, recomp_forces[0], masses, timestep)
            _E = _E_prev - dE
            open('nn-E.txt', 'a').write('{}\n'.format(_E))#.close()
            #np.savetxt(f3, _E)

            #if i%(nsteps/100) == 0:
            Writer.write_xyz([coords_current], molecule.atoms, 
                'nn-coords.xyz', 'a', i)
            Writer.write_xyz(recomp_forces, molecule.atoms, 
                'nn-forces.xyz', 'a', i)

            coords_prev = coords_current
            coords_current = coords_next
            _E_prev = _E
            #print()


        

