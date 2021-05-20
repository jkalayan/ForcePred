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
#import sys

class Network(object):
    '''
    '''
    def __init__(self):
        self.scale_NRF = None
        self.scale_F = None
        self.model = None


    def get_variable_depth_model(self, molecule):
        _NRF = np.take(molecule.mat_NRF, molecule.train, axis=0)
        _F = np.take(molecule.mat_F, molecule.train, axis=0)
        #normalise inputs and forces
        self.scale_NRF = np.amax(_NRF)
        _NRF = _NRF / self.scale_NRF
        self.scale_F = 2 * np.amax(np.absolute(_F))
        _F = _F / self.scale_F + 0.5
        print(self.scale_NRF, self.scale_F)

        max_depth = 6
        file_name = 'best_model'
        mc = ModelCheckpoint(file_name, monitor='loss', 
                mode='min', save_best_only=True)
        es = EarlyStopping(monitor='loss', patience=500)
        model_in = Input(shape=(_NRF.shape[1],))
        model = model_in
        best_error = 1000
        for i in range(0, max_depth):
            if i > 0:
                model = concatenate([model,model_in])
            net = Dense(units=1000, activation='sigmoid')(model)
            net = Dense(units=_NRF.shape[1], activation='sigmoid')(net)         
            model = Model(model_in, net)
            model.compile(loss='mse', optimizer='adam')
            model.summary()
            model.fit(_NRF, _F, epochs=100000, verbose=2, callbacks=[mc,es])
            model = load_model(file_name)
            if model.evaluate(_NRF, _F, verbose=0) < best_error:
                model.save('best_ever_model')
                best_error = model.evaluate(_NRF, _F, verbose=0)
                print('best model was achieved on layer %d' % i)
                print('its error was:')
                print(best_error)
            #end_training = np.loadtxt('end_file', dtype=int)
            #if end_training == 1:
                #break
            #if time.time()-start_time >= 518400:
                #break
            model = load_model(file_name)
            model.trainable = False
            model = model(model_in)
        self.model = model

    #def test_model(self, model, molecule):

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

    def get_recomposed_forces(coords, prediction, n_atoms, _NC2):
        '''Take pairwise decomposed forces and convert them back into 
        system forces.'''
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
        recomp_forces = np.reshape(np.dot(_T, prediction[0]), (-1,3))
        return recomp_forces

    def run_NVE(network, molecule, timestep, nsteps):
        mm = MM() #initiate MM class
        coords_init = molecule.coords[0]
        atoms = molecule.atoms
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        scale_NRF = network.scale_NRF #27942.731497548113 #
        scale_F = network.scale_F #0.06989962601741863 #
        mm.coords.append(coords_init)
        #masses = np.array([Converter._ZM[a] for a in atoms])
        masses = np.zeros((n_atoms,3))
        for i in range(n_atoms):
            masses[i,:] = Converter._ZM[atoms[i]]
        #model = network.model
        model = load_model('best_ever_model')
        for i in range(nsteps):
            coords_current = mm.coords[i]
            mat_NRF = Network.get_NRF_input(coords_current, atoms, 
                    n_atoms, _NC2)
            mat_NRF_scaled = mat_NRF / scale_NRF
            prediction_scaled = model.predict(mat_NRF_scaled)
            prediction = (prediction_scaled - 0.5) * scale_F
            #print(prediction)
            #print(prediction.shape)
            recomp_forces = Network.get_recomposed_forces(coords_current, 
                    prediction, n_atoms, _NC2)
            #print(recomp_forces)
            #print(recomp_forces.shape)
            #print(molecule.forces[0])
            #print(molecule.forces[0].shape)
            #print()
            mm.forces.append(recomp_forces)
            coords_prev = mm.coords[i]
            if i != 0:
                coords_prev = mm.coords[i-1]
            coords_next = MM.calculate_verlet_step(coords_current, 
                    coords_prev, recomp_forces, masses, timestep)
            mm.coords.append(coords_next)
        return mm




                

        

