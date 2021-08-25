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
from ..calculate.Converter import Converter
#from ..calculate.Binner import Binner
#from ..calculate.Plotter import Plotter
#from ..write.Writer import Writer
#from ..read.Molecule import Molecule
#from ..nn.Network import Network
#import sys
#import time

class Conservation(object):
    '''
    '''
    #def __init__(self):
        #self.model = None

    def get_conservation(coords, forces, atoms, scale_NRF, scale_F, 
            model_name, molecule, dr):
        '''scale forces to ensure forces and energies are conserved'''
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        #dr = 0.001 #angstrom
        model = load_model(model_name)

        all_plus_coords, all_minus_coords = \
                Conservation.get_displaced_structures(coords, dr)

        #undisplaced structure
        _NRF = Conservation.get_NRF_input([coords], atoms, n_atoms, _NC2)
        _NRF_scaled = _NRF / scale_NRF
        prediction_scaled = model.predict(_NRF_scaled)
        prediction = (prediction_scaled - 0.5) * scale_F

        #positively displaced structures
        plus_NRF = Conservation.get_NRF_input(all_plus_coords, atoms, 
                n_atoms, _NC2)
        plus_NRF_scaled = plus_NRF / scale_NRF
        plus_prediction_scaled = model.predict(plus_NRF_scaled)
        plus_prediction = (plus_prediction_scaled - 0.5) * scale_F

        #negatively displaced structures
        minus_NRF = Conservation.get_NRF_input(all_minus_coords, atoms, 
                n_atoms, _NC2)
        minus_NRF_scaled = minus_NRF / scale_NRF
        minus_prediction_scaled = model.predict(minus_NRF_scaled)
        minus_prediction = (minus_prediction_scaled - 0.5) * scale_F

        ''' ### to compare with Neil's code
        prediction = molecule.mat_F[0]
        plus_prediction = []
        minus_prediction = []
        for i in range(1,len(molecule.mat_F),6):
            #print('i', i)
            for j in range(3):
                #print('j', i+j)
                plus_prediction.append(molecule.mat_F[i+j])
            for k in range(3,6):
                #print('k', i+k)
                minus_prediction.append(molecule.mat_F[i+k])
        plus_prediction = np.array(plus_prediction)
        minus_prediction = np.array(minus_prediction)
        '''

        dqBA_dC = Conservation.get_finite_difference(prediction.flatten(), 
                plus_prediction, minus_prediction, dr, n_atoms, _NC2)

        q0_scaled, scaled_F = Conservation.check_conservation(coords, 
                prediction.flatten(), plus_prediction, minus_prediction, 
                dqBA_dC, n_atoms, _NC2)


        #recomp_forces = Network.get_recomposed_forces([coords], 
                #[prediction], n_atoms, _NC2)

        return q0_scaled

    def get_displaced_structures(coords, dr):
        '''For each atom, displace xyz coords by dr and save as 
        new structure'''
        all_plus_coords = []
        all_minus_coords = []
        for i in range(len(coords)):
            for j in range(3):
                plus_coords = np.array(coords, copy=True)
                minus_coords = np.array(coords, copy=True)
                plus_coords[i,j] = plus_coords[i,j] + dr
                minus_coords[i,j] = minus_coords[i,j] - dr
                all_plus_coords.append(plus_coords)
                all_minus_coords.append(minus_coords)
        return np.array(all_plus_coords), np.array(all_minus_coords)

    def get_displaced_structures_new(coords, dr):
        '''For each atom, displace xyz coords by dr and save as 
        new structures'''
        n_atoms = len(coords)
        all_plus_coords = np.repeat(
                coords[np.newaxis, :, :], n_atoms*3, axis=0)
        all_minus_coords = np.repeat(
                coords[np.newaxis, :, :], n_atoms*3, axis=0)
        c = 0
        for i in range(len(coords)):
            for j in range(3):
                c += 1
                all_plus_coords[c,i,j] = all_plus_coords[c,i,j] + dr
                all_minus_coords[c,i,j] = all_minus_coords[c,i,j] - dr
        return all_plus_coords, all_minus_coords


    def get_finite_difference(q0, q_plus, q_minus, dr, n_atoms, _NC2):
        '''Take predicted pairwise forces (q) for displaced structures and
        find the difference between them'''
        first_order = False
        second_order = True
        dqBA_dC = np.zeros((3, _NC2, n_atoms))
        for i in range(n_atoms):
            for _N in range(_NC2):
                qBA_plus = np.zeros((3))
                qBA_minus = np.zeros((3))
                for j in range(3):
                    qBA_plus[j] = q_plus[i*3+j,_N]
                    qBA_minus[j] = q_minus[i*3+j,_N]
                if first_order:
                    dqBA_dC[:,_N,i] = (qBA_plus - q0[_N]) / dr
                if second_order:
                    dqBA_dC[:,_N,i] = (qBA_plus - qBA_minus) / (2 * dr)
        return dqBA_dC

    def check_conservation(coords, q0, q_plus, q_minus, dqBA_dC, 
            n_atoms, _NC2):
        '''Check if differential is zero thus energy conserved, if not,
        then we can scale forces with the scaling factor to equal zero'''
        alpha = 1
        n_constraints = 2 * 3 * n_atoms

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

        q0_matrix = np.zeros((n_constraints, _NC2))
        q_dr_vector = np.zeros((n_constraints))
        eBAx1 = np.zeros((3))
        eBAxqBA = np.zeros((3))
        sum_dqBA_dA = np.zeros((n_atoms, 3))
        #print('n A B BA')
        #print('Atom Constraint(x) Constraint(y) Constraint(z)')
        for n in range(n_atoms):
            BA = -1
            for a in range(n_atoms):
                A = a
                for b in range(a):
                    B = b
                    BA += 1
                    if a == n or b == n:
                        if b == n:
                            A, B = n, a
                        #print(n, A, B, BA)
                        eBAxqBA = np.cross(eij[:,A,B], dqBA_dC[:,BA,A])
                        sum_dqBA_dA[A,:] += eBAxqBA
                        eBAx1 = np.cross(eij[:,A,B], np.ones((3)))
                        q_plus_dr = np.zeros((3))
                        q_minus_dr = np.zeros((3))
                        for i, x in zip(range(3), range(A*3, A*3+3)):
                            q_plus_dr[i] = q_plus[x,BA]
                            q_minus_dr[i] = q_minus[x,BA]

                        #positively displaced structures
                        eBAxqdr = np.cross(eij[:,A,B], q_plus_dr)
                        for i, x in zip(range(3), range(A*3, A*3+3)):
                            q0_matrix[x,BA] += eBAx1[i] * q0[BA]
                            q_dr_vector[x] += eBAxqdr[i]
                            q_dr_vector[x] -= (alpha * eBAx1[i] * 
                                    q0[BA])
                            #print(q_dr_vector[x])

                        #negatively displaced structures
                        eBAxqdr = np.cross(eij[:,A,B], q_minus_dr)
                        for i, x in zip(range(3), range(A*3, A*3+3)):
                            q0_matrix[x+(3*n_atoms),BA] += eBAx1[i] * q0[BA]
                            q_dr_vector[x+(3*n_atoms)] += eBAxqdr[i]
                            q_dr_vector[x+(3*n_atoms)] -= (alpha * eBAx1[i] * 
                                    q0[BA])
                            #print('\t', q_dr_vector[x])

            #print(n, sum_dqBA_dA[n])
        abs_mean = np.sum(np.absolute(sum_dqBA_dA)) / (3 * n_atoms)
        #print('mean absolute conservation constraint = {}'.format(
                #abs_mean))
        scaling_factors = np.matmul(np.linalg.pinv(q0_matrix), 
                q_dr_vector)

        #rescale original qforces by scaling factors
        q0_scaled = np.diag(scaling_factors+alpha*np.ones((_NC2,1))) * q0
        #for i in range(_NC2):
            #print(i, scaling_factors[i]+alpha, q0[i], q0_scaled[i])

        #recompose forces
        scaled_F = np.zeros((n_atoms, 3))
        _N = -1
        for i in range(n_atoms):
            for j in range(i):
                _N += 1
                scaled_F[i,:] += q0_scaled[_N] * eij[:,i,j].T
                scaled_F[j,:] += q0_scaled[_N] * eij[:,j,i].T
        return q0_scaled, scaled_F


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
