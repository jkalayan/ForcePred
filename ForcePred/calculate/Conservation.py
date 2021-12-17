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
import tensorflow as tf
#from ..calculate.MM import MM
from ..calculate.Converter import Converter
#from ..calculate.Binner import Binner
#from ..calculate.Plotter import Plotter
#from ..write.Writer import Writer
from ..read.Molecule import Molecule
#from ..nn.Network import Network
import sys
#import time
import matplotlib.pyplot as plt

class Conservation(object):
    '''
    '''
    #def __init__(self):
        #self.model = None

    def get_conservation(_i, coords, forces, atoms, scale_NRF, scale_NRF_min, 
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
        #print(molecule.coords)
        _NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                _NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        #print(_NRF_scaled)
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

        prediction_orig = np.copy(prediction)
        plus_prediction_orig = np.copy(plus_prediction)
        minus_prediction_orig = np.copy(minus_prediction)

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
        #check if pairwise-energies give back Cart forces with Neil's 
        #structures
        print(molecule.energies)
        print(molecule.forces[0])
        print(np.sum(molecule.forces))
        a, b, c = Conservation.check_invariance(molecule.coords[0], 
                molecule.forces[0])
        print(a, b, c)
        decomp_E = Converter.get_energy_decomposition(molecule.atoms, 
                molecule.coords[0], molecule.energies[0])
        _E = molecule.energies[0]
        plus_decomp_E = []
        minus_decomp_E = []
        plus_E = []
        minus_E = []
        for i in range(1,len(molecule.mat_F),6):
            for j in range(3):
                decomp = Converter.get_energy_decomposition(molecule.atoms, 
                molecule.coords[i+j], molecule.energies[i+j])
                plus_decomp_E.append(decomp)
                plus_E.append(molecule.energies[i+j])
            for k in range(3,6):
                decomp = Converter.get_energy_decomposition(molecule.atoms, 
                molecule.coords[i+k], molecule.energies[i+k])
                minus_decomp_E.append(decomp)
                minus_E.append(molecule.energies[i+k])
        plus_decomp_E = np.array(plus_decomp_E)
        minus_decomp_E = np.array(minus_decomp_E)
        plus_E = np.array(plus_E)
        minus_E = np.array(minus_E)

        conserved, dEAm_dAk = Conservation.check_molecular_E_conservation(
                _E, plus_E, minus_E, dr, n_atoms)
        print(dEAm_dAk)
        print(np.sum(dEAm_dAk))
        a, b, c = Conservation.check_invariance(molecule.coords[0], 
                dEAm_dAk)
        print(a, b, c)
        delta_conserved = Conservation.check_cartF_conservation(
                recomp_forces[0], plus_recomp_forces, minus_recomp_forces, 
                dr, n_atoms)
        print()

        dqBA_dC = Conservation.get_finite_difference(decomp_E, 
                plus_decomp_E, minus_decomp_E, dr, n_atoms, _NC2)

        for i in range(len(dqBA_dC)):
            fig, ax = plt.subplots(figsize=(10, 10), 
                    edgecolor='k') #all in one plot
            ax.imshow(dqBA_dC[i], cmap='hot')
            fig.savefig('dqBA_dC_%s.png' % (i), transparent=True,
                    bbox_inches='tight',)

        print(molecule.mat_F[0])
        np.savetxt('undisplaced.txt', decomp_E)
        np.savetxt('plus.txt', plus_decomp_E)
        np.savetxt('minus.txt', minus_decomp_E)

        q0_scaled = Conservation.check_conservation(coords, decomp_E, 
                plus_decomp_E, minus_decomp_E, dqBA_dC, dr, n_atoms, _NC2)
        
        '''

        #'''
        #conservation check for decomposed pairwise forces
        dqBA_dC = Conservation.get_finite_difference(prediction.flatten(), 
                plus_prediction, minus_prediction, dr, n_atoms, _NC2)

        q0_scaled, scaled_F, plus_scaled_F, minus_scaled_F, \
                scaling_factors = \
                Conservation.check_conservation(coords, 
                prediction.flatten(), plus_prediction, minus_prediction, 
                dqBA_dC, dr, n_atoms, _NC2)

        #print('original cartF pred', recomp_forces)
        #print('actual cartF', molecule.forces[0])
        #print()
        #print(prediction)
        #print(molecule.mat_F[0])
        #print()
        #'''

        #'''
        #conservation check for recomposed cartesian forces
        #print(recomp_forces[0])
        delta_conserved = Conservation.check_cartF_conservation(
                recomp_forces[0], plus_recomp_forces, minus_recomp_forces, 
                dr, n_atoms)
        delta_conserved_scaled = Conservation.check_cartF_conservation(
                scaled_F, plus_scaled_F, minus_scaled_F, dr, n_atoms)
        #ave_delta_conserved = np.sum(np.absolute(
                #delta_conserved)) / (3 * n_atoms)
        #decomp_Fscale = Converter.get_decomposition(atoms, 
                #coords, delta_conserved)
        print('delta_conserved scaled', delta_conserved_scaled)
        print('delta_conserved unscaled', delta_conserved)
        #print(decomp_Fscale)
        #print(ave_delta_conserved)

        #open('ave_delta_F.txt', 'a').write('{}\n'.format(ave_delta_conserved))
        #'''

        #'''
        #check invariance
        a, b, c = Conservation.check_invariance(coords, scaled_F[0])
        print(a, b, c)
        print()
        #'''

        #'''
        Conservation.print_for_neil(_i, molecule, scaling_factors, 
                    all_plus_coords, all_minus_coords, 
                    prediction_orig, plus_prediction_orig, 
                    minus_prediction_orig, 
                    recomp_forces, plus_recomp_forces, minus_recomp_forces)

        kj2kcal = 1/4.184
        au2kcalmola = Converter.Eh2kcalmol / Converter.au2Ang
        prediction_orig = np.divide(prediction_orig, kj2kcal) 
                #in units kJ/mol A
        #'''

        #return prediction #q0_scaled
        return recomp_forces, scaled_F, prediction_orig, q0_scaled, \
                np.add(scaling_factors, 1)

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
                    #print(sum(dqBA_dC[:,_N,i]))
        return dqBA_dC


    def check_cartF_conservation(f0, f_plus, f_minus, dr, n_atoms):
        '''Check derivatives of Cartesian forces wrt r are conservative'''
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

        delta_conserved = np.sum(delta_conserved, axis=1)

        return delta_conserved


    def check_decompE_conservation(decompE0, decompE_plus, decompE_minus, dr, 
            n_atoms):
        '''Get the derivatives of each decompE and sum these using finite
        difference, we can then compare these with decompF, will they match?
        '''
        _NC2 = int((n_atoms * (n_atoms-1)) / 2)
        dEAm_dAk = np.zeros((_NC2, 3))
        for _N in range(n_atoms):
            for i in range(_NC2):
                eA_plus = np.zeros((3))
                eA_minus = np.zeros((3))
                for j in range(3):
                    eA_plus[j] = decompE_plus[_N*3+j,i]
                    eA_minus[j] = decompE_minus[_N*3+j,i]
                dEAm_dAk[i] += -(eA_plus - eA_minus) / (2 * dr)

        #print(dEAm_dAk.shape, np.sum(dEAm_dAk, axis=1))

        #'''
        delta_conserved = np.zeros((_NC2, 3))
        for _N in range(_NC2):
            delta_conserved[_N,0] = dEAm_dAk[_N,1] - dEAm_dAk[_N,2] 
                    #dEzdy - dEydz
            delta_conserved[_N,1] = dEAm_dAk[_N,2] - dEAm_dAk[_N,0] 
                    #dExdz - dEzdx
            delta_conserved[_N,2] = dEAm_dAk[_N,0] - dEAm_dAk[_N,1] 
                    #dEydx - dExdy
        print('decompE delta conserved', np.sum(delta_conserved, axis=1))
        #'''


        return np.sum(dEAm_dAk, axis=1)



    def check_molecular_E_conservation(e0, e_plus, e_minus, dr, n_atoms):
        '''Check derivatives of molecular energies wrt r are conservative'''
        first_order_plus = False
        second_order = True
        dEAm_dAk = np.zeros((n_atoms,3)) 
                #1st param = k = Ax, Ay or Az; 3rd param = m = Fx, Fy or Fz
        #print('F', f0.shape, f_plus.shape, f_minus.shape)
        for _N in range(n_atoms):
            eA_plus = np.zeros((3))
            eA_minus = np.zeros((3))
            for j in range(3):
                #print(_N*3+j)
                eA_plus[j] = e_plus[_N*3+j]
                eA_minus[j] = e_minus[_N*3+j]
            #print('eA_plus,minus', eA_plus, eA_minus)
            if first_order_plus:
                dEAm_dAk[_N] = -(eA_plus - e0) / dr
            if second_order:
                dEAm_dAk[_N] = -(eA_plus - eA_minus) / (2 * dr)
                #print(dEAm_dAk[_N])

        delta_conserved = np.zeros((n_atoms, 3))
        '''
        for _N in range(n_atoms):
            delta_conserved[_N,0] = dEAm_dAk[_N,1] - dEAm_dAk[_N,2] 
                    #dEzdy - dEydz
            delta_conserved[_N,1] = dEAm_dAk[_N,2] - dEAm_dAk[_N,0] 
                    #dExdz - dEzdx
            delta_conserved[_N,2] = dEAm_dAk[_N,0] - dEAm_dAk[_N,1] 
                    #dEydx - dExdy
        '''
        
        #print(dEAm_dAk)
        #print('\n'*3)
        dEAm_dAk = dEAm_dAk.reshape(-1,3)
        return delta_conserved, dEAm_dAk


    def check_curl_of_cart_forces(forces, n_atoms):
        '''Check curl to make sure it's zero'''
        delta_conserved = np.zeros((n_atoms, 3))
        for _N in range(n_atoms):
            delta_conserved[_N,0] = forces[_N,1] - forces[_N,2] 
                    #dEzdy - dEydz
            delta_conserved[_N,1] = forces[_N,2] - forces[_N,0] 
                    #dExdz - dEzdx
            delta_conserved[_N,2] = forces[_N,0] - forces[_N,1] 
                    #dEydx - dExdy
        return np.sum(delta_conserved, axis=1)


    def check_conservation(coords, q0, q_plus, q_minus, dqBA_dC, 
            dr, n_atoms, _NC2):
        '''Check if differential is zero thus energy conserved, if not,
        then we can scale forces with the scaling factor to equal zero'''
        alpha = 1
        scale_all = True
        n_structures = n_atoms * 3 + 1
        if scale_all:
            n_constraints = 3 * n_atoms
        else:
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
                        if scale_all:
                            #SCALES ALL
                            for i, x in zip(range(3), range(A*3, A*3+3)):
                                q0_matrix[x,BA] += eBAx1[i] * q0[BA]
                            #pseudo 1st order fit (to 2nd order derivatives)
                            q_dr = dqBA_dC[:,BA,A] * dr + q0[BA]

                            eBAxqdr = np.cross(eij[:,A,B], q_dr)
                            for i, x in zip(range(3), range(A*3, A*3+3)):
                                q_dr_vector[x] = q_dr_vector[x] 
                                    #do not add qdR term
                                q0_matrix[x,BA] -= eBAxqdr[i]
                                q_dr_vector[x] -= (alpha * eBAx1[i] * q0[BA])
                                q_dr_vector[x] += (alpha * eBAxqdr[i])

                        else:
                            #plus_minus_constraints, doesn't scale all
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

                            #negatively displaced structures
                            eBAxqdr = np.cross(eij[:,A,B], q_minus_dr)
                            for i, x in zip(range(3), range(A*3, A*3+3)):
                                q0_matrix[x+(3*n_atoms),BA] += (eBAx1[i] * 
                                        q0[BA])
                                q_dr_vector[x+(3*n_atoms)] += eBAxqdr[i]
                                q_dr_vector[x+(3*n_atoms)] -= (alpha * 
                                        eBAx1[i] * q0[BA])
                                #print('\t', q_dr_vector[x])

            #print(n, sum_dqBA_dA[n])
        abs_mean = np.sum(np.absolute(sum_dqBA_dA)) / (3 * n_atoms)
        #print('mean absolute conservation constraint = {}'.format(
                #abs_mean))
        scaling_factors = np.matmul(np.linalg.pinv(q0_matrix), 
                q_dr_vector)

        #rescale original qforces by scaling factors
        q0_scaled = np.diag(scaling_factors+alpha*np.ones((_NC2,1))) * q0
        if scale_all:
            #scale all qs for all structures here
            #q0 *= np.diag(
                    #scaling_factors+alpha*np.ones((_NC2,1)))
            for n in range(n_atoms*3):
                q_plus[n,:] *= np.diag(
                        scaling_factors+alpha*np.ones((_NC2,1)))
                q_minus[n,:] *= np.diag(
                        scaling_factors+alpha*np.ones((_NC2,1)))

        #'''
        for i in range(_NC2):
            print(i, scaling_factors[i]+alpha, q0[i], q0_scaled[i])
            #open('vary_qAB.txt', 'a').write('{} {} {} {}\n'.format(i, 
                    #scaling_factors[i]+alpha, q0[i], q0_scaled[i]))
        #print()
        #'''


        #'''
        #recompose forces, undisplaced only
        scaled_F = np.zeros((1, n_atoms, 3))
        _N = -1
        for i in range(n_atoms):
            for j in range(i):
                _N += 1
                scaled_F[0,i,:] += q0_scaled[_N] * eij[:,i,j]
                scaled_F[0,j,:] += q0_scaled[_N] * eij[:,j,i]
        #'''

        #'''
        #recompose forces, displaced included
        plus_scaled_F = np.zeros((30,n_atoms, 3))
        minus_scaled_F = np.zeros((30,n_atoms, 3))
        for k in range(0, n_atoms*3):
            k_offset = k * n_atoms
            _N = -1
            for i in range(n_atoms):
                for j in range(i):
                    _N += 1
                    plus_scaled_F[k,i,:] += q_plus[k,_N] * eij[:,i,j]
                    plus_scaled_F[k,j,:] += q_plus[k,_N] * eij[:,j,i]
                    minus_scaled_F[k,i,:] += q_minus[k,_N] * eij[:,i,j]
                    minus_scaled_F[k,j,:] += q_minus[k,_N] * eij[:,j,i]
        #'''

        '''
        #conservation check for scaled recomposed cartesian forces
        delta_conserved = Conservation.check_cartF_conservation(
                scaled_F, plus_scaled_F, minus_scaled_F, dr, n_atoms)
        print('delta_conserved', delta_conserved)
        '''

        return q0_scaled, scaled_F, plus_scaled_F, minus_scaled_F, \
                scaling_factors

    def old_check_conservation(coords, q0, q_plus, q_minus, dqBA_dC, 
            dr, n_atoms, _NC2):
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
        #print(q0)
        #print(q0_scaled)
        #'''
        for i in range(_NC2):
            print(i, scaling_factors[i]+alpha, q0[i], q0_scaled[i])
            #open('vary_qAB.txt', 'a').write('{} {} {} {}\n'.format(i, 
                    #scaling_factors[i]+alpha, q0[i], q0_scaled[i]))
        #print()
        #'''


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

    def check_invariance(coords, forces):
        ''' Ensure that forces in each structure translationally and 
        rotationally sum to zero (energy is conserved) i.e. translations
        and rotations are invariant. '''
        #translations
        translations = []
        trans_sum = np.sum(forces,axis=0) #sum columns
        for x in range(3):
            x_trans_sum = round(trans_sum[x], 0)
            translations.append(x_trans_sum)
        translations = np.array(translations)
        #rotations
        cm = np.average(coords, axis=0)
        i_rot_sum = np.zeros((3))
        for i in range(len(forces)):
            r = coords[i] - cm
            diff = np.cross(r, forces[i])
            i_rot_sum = np.add(i_rot_sum, diff)
        rotations = np.round(i_rot_sum, 0)
        #check if invariant
        variance = True
        if np.all((rotations == 0)):
            variance = False
        if np.all((translations == 0)):
            variance = False
        return variance, translations, rotations

    def print_for_neil(i, molecule, scaling_factors, 
            plus_coords, minus_coords, 
            pred_q, plus_pred_q, minus_pred_q, 
            pred_F, plus_pred_F, minus_pred_F):

        kj2kcal = 1/4.184
        au2kcalmola = Converter.Eh2kcalmol / Converter.au2Ang

        n_atoms = len(molecule.atoms)
        i = 0 #!!!!!!!!!!

        open('1_coords.txt', 'w')
        open('2_q_pred.txt', 'w')
        open('3_F_pred.txt', 'w')
        f1 = open('1_coords.txt', 'a')
        f2 = open('2_q_pred.txt', 'a')
        f3 = open('3_F_pred.txt', 'a')
        np.savetxt(f1, molecule.coords[i].reshape(-1,3))
        np.savetxt(f2, np.divide(pred_q[i], kj2kcal).reshape(1,-1))
        np.savetxt(f3, np.divide(pred_F[i], au2kcalmola).reshape(-1,3))
        for n in range(n_atoms):
            for x in range(3):
                np.savetxt(f1, plus_coords[n*3+x].reshape(-1,3))
                np.savetxt(f2, np.divide(plus_pred_q[n*3+x], 
                        kj2kcal).reshape(1,-1))
                np.savetxt(f3, np.divide(plus_pred_F[n*3+x], 
                        au2kcalmola).reshape(-1,3))
            for x in range(3):
                np.savetxt(f1, minus_coords[n*3+x].reshape(-1,3))
                np.savetxt(f2, np.divide(minus_pred_q[n*3+x], 
                        kj2kcal).reshape(1,-1))
                np.savetxt(f3, np.divide(minus_pred_F[n*3+x], 
                        au2kcalmola).reshape(-1,3))
            print()
        f1.close()
        f2.close()
        f3.close()

        open('4_pairs.txt', 'w')
        f4 = open('4_pairs.txt', 'a')
        for a in range(1, n_atoms+1):
            for b in range(1, a):
                f4.write('{}{} '.format(a, b))
        f4.close()

        open('5_scaling_factors.txt', 'w')
        f5 = open('5_scaling_factors.txt', 'a')
        np.savetxt(f5, np.add(scaling_factors, 1).reshape(1,-1))
        f5.close()

        open('6_actual_F.txt', 'w')
        f6 = open('6_actual_F.txt', 'a')
        np.savetxt(f6, np.divide(molecule.forces[i], 
                au2kcalmola).reshape(-1,3))
        f6.close()

        open('7_actual_q.txt', 'w')
        f7 = open('7_actual_q.txt', 'a')
        np.savetxt(f7, np.divide(molecule.mat_F[i], kj2kcal).reshape(1,-1))
        f7.close()




    def get_forces_from_energy(coords, atoms, scale_NRF, scale_NRF_min, 
            scale_E, model, dr, bias_type, molecule, prescale):
        '''
        When predicting decomposed energies, get decompE of displaced
        structures and use finite difference to the forces on the
        undisplaced structure.
        '''
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        NRF_scale_method = 'A'
        output_scale_method = 'B' #'B'
        curl = None

        equiv_atoms = False
        #print('equiv_atoms', equiv_atoms)

        if equiv_atoms:
            A = Molecule.find_bonded_atoms(atoms, 
                    coords)
            indices, pairs_dict = Molecule.find_equivalent_atoms(
                    atoms, A)

        #'''
        #From the ML model that predicts decomposed energies:
        #get displaced structures
        all_plus_coords, all_minus_coords = \
                Conservation.get_displaced_structures(coords, dr)

        #undisplaced structure
        _NRF = Conservation.get_NRF_input([coords], atoms, n_atoms, _NC2)
        if equiv_atoms:
            all_sorted_list, all_resorted_list = Molecule.get_sorted_pairs(
                    _NRF, pairs_dict)
            _NRF = np.take_along_axis(_NRF, all_sorted_list, axis=1)
        _NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                _NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        #print(_NRF_scaled)
        prediction_scaled = model.predict(_NRF_scaled)
        prediction = Conservation.get_unscaled_values(prediction_scaled, 
                scale_E, 0, method=output_scale_method)
        if equiv_atoms:
            _NRF = np.take_along_axis(_NRF, all_resorted_list, axis=1)
            prediction = np.take_along_axis(prediction, 
                    all_resorted_list, axis=1)

        #positively displaced structures
        plus_NRF = Conservation.get_NRF_input(all_plus_coords, atoms, 
                n_atoms, _NC2)
        if equiv_atoms:
            all_sorted_list, all_resorted_list = Molecule.get_sorted_pairs(
                    plus_NRF, pairs_dict)
            plus_NRF = np.take_along_axis(plus_NRF, all_sorted_list, axis=1)
        plus_NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                plus_NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        plus_prediction_scaled = model.predict(plus_NRF_scaled)
        plus_prediction = Conservation.get_unscaled_values(
                plus_prediction_scaled, scale_E, 0, 
                method=output_scale_method)
        if equiv_atoms:
            plus_NRF = np.take_along_axis(plus_NRF, all_resorted_list, axis=1)
            plus_prediction = np.take_along_axis(plus_prediction, 
                    all_resorted_list, axis=1)

        #negatively displaced structures
        minus_NRF = Conservation.get_NRF_input(all_minus_coords, atoms, 
                n_atoms, _NC2)
        if equiv_atoms:
            all_sorted_list, all_resorted_list = Molecule.get_sorted_pairs(
                    minus_NRF, pairs_dict)
            minus_NRF = np.take_along_axis(minus_NRF, all_sorted_list, axis=1)
        minus_NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                minus_NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        minus_prediction_scaled = model.predict(minus_NRF_scaled)
        minus_prediction = Conservation.get_unscaled_values(
                minus_prediction_scaled, scale_E, 0, 
                method=output_scale_method)
        if equiv_atoms:
            minus_NRF = np.take_along_axis(minus_NRF, all_resorted_list, axis=1)
            minus_prediction = np.take_along_axis(minus_prediction, 
                    all_resorted_list, axis=1)
        #'''

        sum_E_included = False
        if sum_E_included:
            prediction = prediction[:,:-1]
            prediction_sum = ((prediction_scaled[:,-1] - 0.5 * _NC2) * 
                    2 * scale_E)
            minus_prediction = minus_prediction[:,:-1]
            minus_prediction_sum = ((minus_prediction_scaled[:,-1] - 0.5 * 
                    _NC2) * 2 * scale_E)
            plus_prediction = plus_prediction[:,:-1]
            plus_prediction_sum = ((plus_prediction_scaled[:,-1] - 0.5 * 
                    _NC2) * 2 * scale_E)
            print(np.sum(prediction))
            print('pred_sum_E', prediction_sum)


        ''' ### to compare with Neil's code
        prediction = np.array([molecule.mat_E[0]])
        prediction_sum = molecule.energies[0]
        plus_prediction = []
        minus_prediction = []
        plus_prediction_sum = []
        minus_prediction_sum = []
        for i in range(1,len(molecule.mat_E),6):
            for j in range(3):
                plus_prediction.append(molecule.mat_E[i+j])
                plus_prediction_sum.append(molecule.energies[i+j])
            for k in range(3,6):
                minus_prediction.append(molecule.mat_E[i+k])
                minus_prediction_sum.append(molecule.energies[i+k])
        plus_prediction = np.array(plus_prediction)
        minus_prediction = np.array(minus_prediction)
        plus_prediction_sum = np.array(plus_prediction_sum)
        minus_prediction_sum = np.array(minus_prediction_sum)
        '''

        mat_FE = True
        if mat_FE:
            f0, e0 = Converter.get_recomposed_FE([coords], 
                    prediction, atoms, n_atoms, _NC2, bias_type)
            plus_f, plus_e = Converter.get_recomposed_FE(all_plus_coords, 
                    plus_prediction, atoms, n_atoms, _NC2, bias_type)
            minus_f, minus_e = Converter.get_recomposed_FE(all_minus_coords, 
                    minus_prediction, atoms, n_atoms, _NC2, bias_type)
            sum_E = np.subtract(e0[0][0]/prescale[1], prescale[0])
            plus_sum_E = np.subtract(plus_e.flatten()/prescale[1], 
                    prescale[0])
            minus_sum_E = np.subtract(minus_e.flatten()/prescale[1], 
                    prescale[0])
            conserved = Conservation.check_cartF_conservation(f0, plus_f, 
                    minus_f, dr, n_atoms)
            #print('e0', e0)
            #print('sum_E', sum_E)
            #print('f0 (recomped from FE)', f0)
            #print('f0 conserved', conserved)
            #a, b, c = Conservation.check_invariance(coords, f0[0])
            #print(a, b, c)


        mat_F = False
        if mat_F:
            f0 = Conservation.get_recomposed_forces([coords], prediction, 
                    n_atoms, _NC2)
            #print(f0.shape, f0)
            f_plus = Conservation.get_recomposed_forces(all_plus_coords, 
                    plus_prediction, n_atoms, _NC2)
            f_minus = Conservation.get_recomposed_forces(all_minus_coords, 
                    minus_prediction, n_atoms, _NC2)
            curl = Conservation.check_cartF_conservation(f0, f_plus, 
                    f_minus, dr, n_atoms)
            #print(f0, curl)
            curl = np.sum(curl)
            #sys.exit()


        ###get the sum of decompEs for each structure
        #sum_E = np.sum(prediction)
        #plus_sum_E = np.sum(plus_prediction, axis=1)
        #minus_sum_E = np.sum(minus_prediction, axis=1)
        '''
        print('pred', sum_E)
        print(prediction)
        #print(plus_sum_E)
        #print(minus_sum_E)
        print()
        print('actual', molecule.energies[0])
        print(molecule.mat_E[0])
        print()
        print('diff decompE', np.subtract(molecule.mat_E[0], prediction))
        print('diff sumE', np.subtract(molecule.energies[0], sum_E))
        print()
        '''

        '''
        #np.set_printoptions(precision=4)

        plus_energies = []
        minus_energies = []
        for i in range(1,len(molecule.energies),6):
            for j in range(3):
                plus_energies.append(molecule.energies[i+j])
            for k in range(3,6):
                minus_energies.append(molecule.energies[i+k])

        print(sum_E, molecule.energies[0][0])
        for e, e2 in zip(plus_sum_E, plus_energies): #e2 = actual
            print(e, e2[0])
        for e, e2 in zip(minus_sum_E, minus_energies):
            print(e, e2[0])
        '''


        #'''
        ##get Cart force from energies via finite diff
        if sum_E_included:
            #use predicted sum
            conserved, dEAm_dAk = Conservation.check_molecular_E_conservation(
                    prediction_sum, plus_prediction_sum, minus_prediction_sum, 
                    dr, n_atoms)
        else:
            #sum all decompE
            conserved, dEAm_dAk = Conservation.check_molecular_E_conservation(
                    sum_E, plus_sum_E, minus_sum_E, dr, n_atoms)

        dEAm_dAk = dEAm_dAk.reshape(1,-1,3)
        #print('predF', dEAm_dAk.shape, dEAm_dAk)
        #print('conserved', np.sum(conserved, axis=1))
        #a, b, c = Conservation.check_invariance(coords, dEAm_dAk[0])
        #print(a, b, c)
        #print()
        #'''

        #check energy conservation from Cart forces
        all_plus_dEAm_dAk = []
        all_minus_dEAm_dAk = []
        for i in range(len(all_plus_coords)):
            plus_dEAm_dAk = Conservation.get_displaced_forces(
                    all_plus_coords[i], 
                    dr, atoms, n_atoms, _NC2, scale_NRF, scale_NRF_min, 
                    scale_E, model, bias_type, prescale)
            minus_dEAm_dAk = Conservation.get_displaced_forces(
                    all_minus_coords[i], 
                    dr, atoms, n_atoms, _NC2, scale_NRF, scale_NRF_min, 
                    scale_E, model, bias_type, prescale)
            all_plus_dEAm_dAk.append(plus_dEAm_dAk)
            all_minus_dEAm_dAk.append(minus_dEAm_dAk)
        all_plus_dEAm_dAk = np.array(all_plus_dEAm_dAk)
        all_minus_dEAm_dAk = np.array(all_minus_dEAm_dAk)

        curl = Conservation.check_cartF_conservation(dEAm_dAk[0], 
                all_plus_dEAm_dAk, all_minus_dEAm_dAk, dr, n_atoms)



        '''
        print()
        print('actualF', molecule.forces[0])
        a, b, c = Conservation.check_invariance(molecule.coords[0], 
                molecule.forces[0])
        print(a, b, c)
        '''


        '''
        #####get gradient oof network model
        print('get network model gradient')
        np_NRF_scaled = np.asarray(_NRF_scaled, np.float32)
        tf_NRF_scaled = tf.convert_to_tensor(np_NRF_scaled, np.float32)
        gradient = tf.gradients(model(tf_NRF_scaled), tf_NRF_scaled)
        with tf.Session() as sess: #to get tensor to numpy
            sess.run(tf.global_variables_initializer())
            result_output_scaled = sess.run(gradient[0])
            print(result_output_scaled)
            result_output = Conservation.get_unscaled_values(
                    result_output_scaled, scale_E, 0, 
                    method=output_scale_method)
        '''

        
        '''
        ##doesn't work
        print('test to get decompF finite difference of decompE')
        qs = Conservation.check_decompE_conservation(
                prediction, plus_prediction, minus_prediction, dr, n_atoms)
        print(qs, qs.shape)
        print(molecule.mat_F[0])
        print()
        recomp_forces = Conservation.get_recomposed_forces([coords], 
                qs.reshape(1,-1), n_atoms, _NC2)
        print('recomp_forces from derived qs', recomp_forces)
        print(molecule.forces[0])
        print()

        for q1, q2 in zip(molecule.mat_F[0], qs):
            print(q1, q2)
        '''

        #print()
        #print(dEAm_dAk)
        #print(curl)
        curl = np.sum(np.abs(curl))

        return dEAm_dAk, curl #  f0




    def get_displaced_forces(coords, dr, atoms, n_atoms, _NC2, 
            scale_NRF, scale_NRF_min, scale_E, model, bias_type, prescale):
        '''Get forces from energies
        '''
        NRF_scale_method = 'A'
        output_scale_method = 'B' #'B'
        #From the ML model that predicts decomposed energies:
        #get displaced structures
        all_plus_coords, all_minus_coords = \
                Conservation.get_displaced_structures(coords, dr)

        #undisplaced structure
        _NRF = Conservation.get_NRF_input([coords], atoms, n_atoms, _NC2)
        _NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                _NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        prediction_scaled = model.predict(_NRF_scaled)
        prediction = Conservation.get_unscaled_values(prediction_scaled, 
                scale_E, 0, method=output_scale_method)

        #positively displaced structures
        plus_NRF = Conservation.get_NRF_input(all_plus_coords, atoms, 
                n_atoms, _NC2)
        plus_NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                plus_NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        plus_prediction_scaled = model.predict(plus_NRF_scaled)
        plus_prediction = Conservation.get_unscaled_values(
                plus_prediction_scaled, scale_E, 0, 
                method=output_scale_method)

        #negatively displaced structures
        minus_NRF = Conservation.get_NRF_input(all_minus_coords, atoms, 
                n_atoms, _NC2)
        minus_NRF_scaled, min_, max_ = Conservation.get_scaled_values(
                minus_NRF, scale_NRF, scale_NRF_min, method=NRF_scale_method) 
        minus_prediction_scaled = model.predict(minus_NRF_scaled)
        minus_prediction = Conservation.get_unscaled_values(
                minus_prediction_scaled, scale_E, 0, 
                method=output_scale_method)
        #'''


        mat_FE = True
        if mat_FE:
            f0, e0 = Converter.get_recomposed_FE([coords], 
                    prediction, atoms, n_atoms, _NC2, bias_type)
            plus_f, plus_e = Converter.get_recomposed_FE(all_plus_coords, 
                    plus_prediction, atoms, n_atoms, _NC2, bias_type)
            minus_f, minus_e = Converter.get_recomposed_FE(all_minus_coords, 
                    minus_prediction, atoms, n_atoms, _NC2, bias_type)
            sum_E = np.subtract(e0[0][0]/prescale[1], prescale[0])
            plus_sum_E = np.subtract(plus_e.flatten()/prescale[1], 
                    prescale[0])
            minus_sum_E = np.subtract(minus_e.flatten()/prescale[1], 
                    prescale[0])


        conserved, dEAm_dAk = Conservation.check_molecular_E_conservation(
                sum_E, plus_sum_E, minus_sum_E, dr, n_atoms)

        dEAm_dAk = dEAm_dAk.reshape(-1,3)

        return dEAm_dAk


