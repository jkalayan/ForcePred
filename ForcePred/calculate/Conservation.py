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

    def get_conservation(coords, forces, atoms, scale_NRF, scale_NRF_min, 
            scale_F, model, molecule, dr, NRF_scale_method,
            qAB_factor=0):
        '''scale forces to ensure forces and energies are conserved'''
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        #dr = 0.001 #angstrom
        #model = load_model(model_name)

        all_plus_coords, all_minus_coords = \
                Conservation.get_displaced_structures(coords, dr)

        #undisplaced structure
        _NRF = Conservation.get_NRF_input([coords], atoms, n_atoms, _NC2)
        _NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                _NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        prediction_scaled = model.predict(_NRF_scaled)
        prediction = Conservation.get_unscaled_values(prediction_scaled, 
                scale_F, 0, method='B')

        #positively displaced structures
        plus_NRF = Conservation.get_NRF_input(all_plus_coords, atoms, 
                n_atoms, _NC2)
        plus_NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                plus_NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        plus_prediction_scaled = model.predict(plus_NRF_scaled)
        plus_prediction = Conservation.get_unscaled_values(
                plus_prediction_scaled, scale_F, 0, method='B')

        #negatively displaced structures
        minus_NRF = Conservation.get_NRF_input(all_minus_coords, atoms, 
                n_atoms, _NC2)
        minus_NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                minus_NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        minus_prediction_scaled = model.predict(minus_NRF_scaled)
        minus_prediction = Conservation.get_unscaled_values(
                minus_prediction_scaled, scale_F, 0, method='B')

        '''
        ##vary some decomposed forces
        open('vary_qAB.txt', 'a').write('{}\n'.format(qAB_factor))
        #print(prediction, qAB_factor)
        #print('NC2 00', prediction[0][0], qAB_factor)
        n_structures = 1
        for i in range(n_structures):
            prediction[0][i] = prediction[0][i] * qAB_factor
        for h in range(n_atoms*3):
            for i in range(n_structures):
                plus_prediction[h][i] = plus_prediction[h][i] * qAB_factor
        for h in range(n_atoms*3):
            for i in range(n_structures):
                minus_prediction[h][i] = minus_prediction[h][i] * qAB_factor
        '''

        ##all structures recomposed forces
        recomp_forces = Conservation.get_recomposed_forces([coords], 
                prediction, n_atoms, _NC2)
        plus_recomp_forces = Conservation.get_recomposed_forces(
                all_plus_coords, plus_prediction, n_atoms, _NC2)
        minus_recomp_forces = Conservation.get_recomposed_forces(
                all_minus_coords, minus_prediction, n_atoms, _NC2)


        ''' ### to compare with Neil's code
        prediction = np.array([molecule.mat_F[0]])
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

        recomp_forces = np.array([molecule.forces[0]])
        plus_recomp_forces = []
        minus_recomp_forces = []
        for i in range(1,len(molecule.forces),6):
            for j in range(3):
                plus_recomp_forces.append(molecule.forces[i+j])
            for k in range(3,6):
                minus_recomp_forces.append(molecule.forces[i+k])
        plus_recomp_forces = np.array(plus_recomp_forces)
        minus_recomp_forces = np.array(minus_recomp_forces)
        '''

        '''
        #conservation check for decomposed pairwise forces
        dqBA_dC = Conservation.get_finite_difference(prediction.flatten(), 
                plus_prediction, minus_prediction, dr, n_atoms, _NC2)

        q0_scaled = Conservation.check_conservation(coords, 
                prediction.flatten(), plus_prediction, minus_prediction, 
                dqBA_dC, n_atoms, _NC2)
        '''

        '''
        #conservation check for recomposed cartesian forces
        delta_conserved = Conservation.check_cartF_conservation(
                recomp_forces[0], plus_recomp_forces, minus_recomp_forces, 
                dr, n_atoms)
        ave_delta_conserved = np.sum(np.absolute(
                delta_conserved)) / (3 * n_atoms)
        decomp_Fscale = Converter.get_decomposition(atoms, 
                coords, delta_conserved)
        print(delta_conserved)
        print(decomp_Fscale)
        print(ave_delta_conserved)
        print()
        open('ave_delta_F.txt', 'a').write('{}\n'.format(ave_delta_conserved))
        '''

        return prediction #q0_scaled

    def get_displaced_structures_old(coords, dr):
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

    def get_displaced_structures(coords, dr):
        '''For each atom, displace xyz coords by dr and save as 
        new structures'''
        n_atoms = len(coords)
        all_plus_coords = np.repeat(
                coords[np.newaxis, :, :], n_atoms*3, axis=0)
        all_minus_coords = np.repeat(
                coords[np.newaxis, :, :], n_atoms*3, axis=0)
        c = -1
        for i in range(len(coords)):
            for j in range(3):
                c += 1
                all_plus_coords[c,i,j] = all_plus_coords[c,i,j] + dr
                all_minus_coords[c,i,j] = all_minus_coords[c,i,j] - dr
        return all_plus_coords, all_minus_coords


    def get_finite_difference(q0, q_plus, q_minus, dr, n_atoms, _NC2):
        '''Take predicted pairwise forces (q) for displaced structures and
        find the difference between them'''
        first_order_plus = False
        second_order = True
        dqBA_dC = np.zeros((3, _NC2, n_atoms))
        #print('q', q0.shape, q_plus.shape, q_minus.shape)
        for i in range(n_atoms):
            for _N in range(_NC2):
                qBA_plus = np.zeros((3))
                qBA_minus = np.zeros((3))
                for j in range(3):
                    qBA_plus[j] = q_plus[i*3+j,_N]
                    qBA_minus[j] = q_minus[i*3+j,_N]
                if first_order_plus:
                    dqBA_dC[:,_N,i] = (qBA_plus - q0[_N]) / dr
                if second_order:
                    dqBA_dC[:,_N,i] = (qBA_plus - qBA_minus) / (2 * dr)
        return dqBA_dC


    def check_cartF_conservation(f0, f_plus, f_minus, dr, n_atoms):
        '''Check derivatives of cartisian forces wrt r are conservative'''
        first_order_plus = False
        second_order = True
        dFAm_dAk = np.zeros((3, n_atoms, 3)) 
                #1st param = k = Ax, Ay or Az; 3rd param = m = Fx, Fy or Fz
        #print('F', f0.shape, f_plus.shape, f_minus.shape)
        for _N in range(n_atoms):
            for i in range(3):
                fA_plus = np.zeros((3))
                fA_minus = np.zeros((3))
                for j in range(3):
                    #print(_N,i,j)
                    fA_plus[j] = f_plus[_N*3+j,_N,i]
                    fA_minus[j] = f_minus[_N*3+j,_N,i]
                #print('fA_plus', fA_plus)
                if first_order_plus:
                    dFAm_dAk[:,_N,i] = (fA_plus - f0[_N]) / dr
                if second_order:
                    dFAm_dAk[:,_N,i] = (fA_plus - fA_minus) / (2 * dr)

        delta_conserved = np.zeros((n_atoms, 3))
        for _N in range(n_atoms):
            delta_conserved[_N,0] = dFAm_dAk[1,_N,2] - dFAm_dAk[2,_N,1] 
                    #dFzdy - dFydz
            delta_conserved[_N,1] = dFAm_dAk[2,_N,0] - dFAm_dAk[0,_N,2] 
                    #dFxdz - dFzdx
            delta_conserved[_N,2] = dFAm_dAk[0,_N,1] - dFAm_dAk[1,_N,0] 
                    #dFydx - dFxdy

        return delta_conserved


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
        #q_list = []
        for i in range(1,n_atoms):
            for j in range(i):
                rij[:,i,j] = (coords[i,:] - coords[j,:])
                rij[:,j,i] = -rij[:,i,j]
                eij[:,i,j] = rij[:,i,j] / np.reshape(
                        np.linalg.norm(rij[:,i,j], axis=0), (-1,1))
                eij[:,j,i] = -eij[:,i,j]
                #q_list.append([i,j])

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
                            q_dr_vector[x] -= (alpha * eBAx1[i] * q0[BA])
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
        '''
        for i in range(_NC2):
            print(i, scaling_factors[i]+alpha, q0[i], q0_scaled[i])
            open('vary_qAB.txt', 'a').write('{} {} {} {}\n'.format(i, 
                    scaling_factors[i]+alpha, q0[i], q0_scaled[i]))
        #print()
        '''


        '''
        #recompose forces
        scaled_F = np.zeros((n_atoms, 3))
        _N = -1
        for i in range(n_atoms):
            for j in range(i):
                _N += 1
                scaled_F[i,:] += q0_scaled[_N] * eij[:,i,j].T
                scaled_F[j,:] += q0_scaled[_N] * eij[:,j,i].T
        '''

        return q0_scaled #, scaled_F





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


    def get_scaled_values(values, scale_max, scale_min, method):
        '''normalise values for NN'''
        #print(np.amax(values), np.amin(values))

        if method == 'A':
            scaled_values = values / scale_max
        if method == 'B':
            scaled_values = values / (2 * scale_max) + 0.5
        if method == 'C':
            scaled_values = (values - scale_min) / (scale_max - scale_min)

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

        return values



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
