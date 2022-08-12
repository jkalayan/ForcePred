#!/usr/bin/env python

'''
This module is used to store information required to predict forces:
    Nuclear charges, coordinates, forces and molecule energies.
    Coords units are in Angstrom
    Energy units are in kcal/mol
    Force units are in kcal/(mol Ang)
    Z units are in units of elementary charge (e)
    Addtional variables added are:
        NC2 pairwise forces shaped (Nstructures, NC2)

'''

import math
import numpy as np
from ..calculate.Plotter import Plotter
from ..calculate.Converter import Converter
import sys

class Molecule(object):
    '''
    Base class for coords, forces and energies of an array of
    structures of N atoms.
    '''
    def __str__(self):
        return ('\nInput files: %s, \natoms: %s,\nN atoms: %s, ' \
                '\nN structures: %s\n' % 
                (self.filenames, ' '.join(map(str, self.atoms)), 
                len(self.atoms), len(self.coords)))

    def get_ZCFE(self, other):
        self.filenames = other.filenames
        self.atoms = other.atoms
        if len(other.coords) > 0:
            self.coords = self.get_3D_array(other.coords)
        if len(other.forces) > 0:
            self.forces = self.get_3D_array(other.forces)
        if hasattr(other, 'energies'):
            if len(other.energies) > 0:
                self.energies = self.get_2D_array(other.energies)
        '''
        if hasattr(other, 'charges'):
            if len(other.charges) > 0:
                self.charges = self.get_2D_array(other.charges).reshape(
                        -1,len(self.atoms))
            #self.charges = self.get_3D_array(other.charges)
        '''

    def check_force_conservation(self):
        ''' Ensure that forces in each structure translationally and 
        rotationally sum to zero (energy is conserved) i.e. translations
        and rotations are invariant. '''
        unconserved = []
        translations = []
        rotations = []
        for s in range(len(self.forces)):
            cm = np.average(self.coords[s], axis=0)
            #masses = np.array([Converter._ZM[a] for a in self.atoms])
            #cm = Converter.get_com(self.coords[s], masses)
            for x in range(3):
                #translations
                x_trans_sum = np.sum(self.forces[s],axis=0)[x]
                #print(x_trans_sum)
                x_trans_sum = abs(round(x_trans_sum, 0))
                if abs(x_trans_sum) != 0 and s not in unconserved:
                    unconserved.append(s)
                    translations.append(x_trans_sum)
            #rotations
            i_rot_sum = np.zeros((3))
            for i in range(len(self.forces[s])):
                r = self.coords[s][i] - cm
                diff = np.cross(r, self.forces[s][i])
                i_rot_sum = np.add(i_rot_sum, diff)
            i_rot_sum = np.round(np.abs(i_rot_sum), 0)
            if np.all(i_rot_sum != 0) and s not in unconserved:
                unconserved.append(s)
                rotations.append(i_rot_sum)
        if len(unconserved) != 0:
            print('Warning: structures %s are variant, '\
                    'from dataset shaped %s.' % 
                    (unconserved, self.coords.shape))
            print('translations {}'.format(translations))
            print('rotations {}'.format(rotations))
            Molecule.remove_variants(self, unconserved)
            print('New dataset shape is', self.coords.shape, 
                    self.forces.shape)
        return np.array(unconserved)


    def permutation_sort(atoms, ref_coords, ref_groups, coords):
        '''As per Neil's PairPermSort, reorder atoms based on reference
        structure.
        1. Choose the correct atom order, this is based on (a) nuclear charge
        (b) largest NRE. This ordering will prioritise fitting of atoms to
        the reference structure.
        2. 
        '''


        #####################################
        #NRE of reference molecule
        n_atoms = len(atoms)
        atoms = np.array(atoms)
        a = np.expand_dims(atoms, 1)
        b = np.expand_dims(atoms, 0)
        atoms_mul = a * b

        a = np.expand_dims(ref_coords, 2)
        b = np.expand_dims(ref_coords, 1)
        diff = a - b
        diff2 = np.sum(diff**2, axis=-1) #get sqrd diff
        r_ref = diff2**0.5
        _NRE = (atoms_mul * Converter.au2kcalmola) / r_ref
        _NRE[_NRE == np.inf] = 0
        atomNRE = np.sum(_NRE, 1) / 2
        #####################################


        #reference RDF info
        z_tile = np.tile(atoms, (n_atoms,1)).T
        print(z_tile)
        print(r_ref)
        sigma_ref = 0.05
        v3 = 1 / (2 * sigma_ref ** 2)
        print(v3)


        #interatomic distances for all structures
        a = np.expand_dims(coords, 2)
        b = np.expand_dims(coords, 1)
        diff = a - b
        diff2 = np.sum(diff**2, axis=-1) #get sqrd diff
        r_all = diff2**0.5

        print(r_all.shape)
        print(r_all[0])


        ##ignoring perm groups for now


        #scoring atoms against ref

        def calc_gaussian(z, r_ref, v3, r_all):
            y_gaussian = (
                    z * (v3 / np.pi) ** 0.5 * 
                    np.exp(-v3 * (r_all - r_ref) * (r_all - r_ref)))
            return y_gaussian

        def calc_sigmoid(r_all, Scutoff, Srate):
            sigmoid = 1.0 - (1.0 / (1.0 + np.exp(Srate * (-r_all + Scutoff))))
            return sigmoid

        Scutoff, Srate = 10, 10

        for ref_i in range(n_atoms):
            for n in range(n_atoms):
                score = 0
                for i in range(n_atoms):
                    for j in range(n_atoms):
                        #print(atoms[ref_i], z_tile[ref_i][j], atoms[i], r_ref[0][ref_i][j], 
                                #r_all[0][n][i])
                        score += (
                                calc_gaussian(z_tile[ref_i][j], 
                                    r_ref[0][ref_i][j], v3, r_all[0][n][i]) * 
                                atoms[i] * 
                                calc_sigmoid(r_all[0][n][i], Scutoff, Srate))
                        #print(calc_gaussian(z_tile[ref_i][j], 
                                    #r_ref[0][ref_i][j], v3, r_all[0][n][i]))
                        #print(calc_sigmoid(r_all[0][n][i], Scutoff, Srate))
                print(score)
        #print(score.shape)
        #print(score[0])
        #print(np.sum(score[0], axis=1))


        score_tolerance = 0.1 #empirically chosen based on observations

        '''
        Z_sorted_ind = np.expand_dims(np.argsort(atoms)[::-1], (0,2)) #descending
        print(Z_sorted_ind)
        print(Z_sorted_ind.shape)
        print(ref_coords.shape)
        NRE_sorted_ind = np.expand_dims(np.argsort(atomNRE)[::-1], (0,2))
        print(NRE_sorted_ind.shape)
        ref_coords_sorted = np.take(ref_coords, Z_sorted_ind, 
                axis=1).reshape(1,-1,3)
        print(ref_coords_sorted)
        ref_coords_sorted = np.take(ref_coords_sorted, NRE_sorted_ind, axis=1)
        print(ref_coords)
        print(ref_coords_sorted)
        '''


    def find_bonded_atoms(atoms, coords):
        n_atoms = len(atoms)
        A = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i):
                r = Converter.get_r(coords[i], coords[j])
                #print(i+1,j+1,r)
                if r < 1.6:
                    A[i,j] = 1
                    A[j,i] = 1
                else:
                    continue
        return A


    def find_equivalent_atoms(atoms, A):
        '''atoms = list of nuclear charges and A is an adjacency matrix
        returns a list of indices of atoms that are equivalent and a
        dictionary of equivalent atom pairs.
        X is a representation of each atom type in the molecule.'''
        n_atoms = len(atoms)
        Z_types = list(set(atoms))
        N_atomTypes = len(Z_types)
        N = np.eye(N_atomTypes)    
        #print('N', N)
        X = []
        for z in atoms:
            ind = Z_types.index(z)
            X.append(N[ind])
        X = np.array(X)
        #print('X', X)
        E = []
        n = -1
        for row in A:
            n += 1
            r = np.transpose(np.array([row,]*N_atomTypes))
            #print('r', r)
            h1 = r * X
            sum1 = np.sum(h1, axis=0)
            all_sum = 0
            for i in range(len(h1)):
                h = h1[i]
                is_zero = np.all(h == 0)
                if is_zero == False:
                    row2 = A[i]
                    r2 = np.transpose(np.array([row2,]*N_atomTypes))
                    h2 = r2 * X
                    all_sum += np.sum(h2, axis=0)
            cross = np.cross(sum1, all_sum.T) * X[n] #Z_types[n] #amended here
            E.append(cross)
        E = np.array(E)
        #print('E', E)
        u, indices = np.unique(E, axis=0, return_inverse=True)

        pairs = []
        pairs_dict = {}
        _N = -1
        for i in range(n_atoms):
            for j in range(i):
                _N += 1
                pair = (indices[i], indices[j])
                pairs.append(pair)
                if pair not in pairs_dict:
                    pairs_dict[pair] = []
                pairs_dict[pair].append(_N)

        return indices, pairs_dict



    def get_sorted_pairs(all_q_list, pairs_dict):
        '''Input list of decomposed forces and pairs_dict and output 
        a sorted list of decomposed forces and original indices'''
        _NC2 = len(all_q_list[0])

        ##q_list = np.random.choice(1000, _NC2+1)
        #print('\nq_list', q_list, len(q_list))

        all_sorted_list = []
        all_resorted_list = []

        for q_list in all_q_list:
            sorted_N_list = []
            for pair, list_N in pairs_dict.items():
                qs = []
                for _N in list_N:
                    qs.append(q_list[_N])
                    #print(_N, pair, q_list[_N])
                qs = np.array(qs)
                sorted_ = np.argsort(qs)
                #print('\tsorted', sorted_)
                for i in sorted_:
                    sorted_N_list.append(list_N[i])

            resorted_N_list = np.argsort(sorted_N_list) #double sort for orig
            all_sorted_list.append(np.array(sorted_N_list))
            all_resorted_list.append(np.array(resorted_N_list))

        all_sorted_list = np.array(all_sorted_list)
        all_resorted_list = np.array(all_resorted_list)

        '''
        #print('all_sorted_list', all_sorted_list)
        #print('all_resorted_list', all_resorted_list)

        q_list_sorted = np.take_along_axis(all_q_list, 
                all_sorted_list, axis=1)
        q_list_check = np.take_along_axis(q_list_sorted, all_resorted_list,
                axis=1)

        #print('\nq_list_sorted', q_list_sorted)
        #print('\nq_list_check', q_list_check)
        #print('\nall_q_list', all_q_list)
        '''

        return all_sorted_list, all_resorted_list

    def remove_variants(self, unconserved):
        s = unconserved
        #for s in unconserved:
        self.coords = np.delete(self.coords, s, axis=0)
        self.forces = np.delete(self.forces, s, axis=0)
        if hasattr(self, 'energies'):
            if len(self.energies) > 0:
                self.energies = np.delete(self.energies, s, axis=0)
        if hasattr(self, 'charges'):
            self.charges = np.delete(self.charges, s, axis=0)

    def make_train_test_old(molecule, variable, split):                       
        '''Split input data into training and test sets. Data is          
        ordered by a particular variable, outputted are lists of          
        indices for each set.'''                                          
        sorted_i = np.argsort(variable)                                   
        _Nstructures = len(variable)                                      
        _Ntrain = math.floor(_Nstructures / split) #round down            
        print('total points:', _Nstructures, 'number of training points:',
                _Ntrain, 'ratio:', round(split, 3))                       
        #a = np.array(range(0,_Nstructures))                              
        a = sorted_i                                                      
        b = np.where(a % int(_Nstructures/_Ntrain),-1,a)                
        c = np.where(a % int(_Nstructures/_Ntrain),a,-1)                
        molecule.train = b[(b>=0)]                                        
        #np.savetxt('train_indices.txt', b)                               
        #print(molecule.train.shape)                                      
        molecule.test = c[(c>0)]                                          
        #print(molecule.test.shape)


    def make_train_test(variable, n_train, n_test):
        '''Split input data into training and test sets. Data is 
        ordered by a particular variable, outputted are lists of
        indices for each set.'''
        sorted_i = np.argsort(variable)
        n_structures = len(variable)
        np.random.seed(22) ##selected for reproducability
        test = np.random.choice(n_structures, n_test)
        train = np.random.choice(np.setdiff1d(range(n_structures), test), 
                n_train) #select train from points not in test

        print('total points:', n_structures, '\nnumber of training points:', 
                train.shape, 'fraction: 1 /', round(n_structures/n_train, 3), 
                '\nnumber of test points:', test.shape, 
                'fraction: 1 /', round(n_structures/n_test, 3))
        return train, test

    def make_train_test_random_chunks(molecule, variable, split):
        '''Split data into n even chunks, then randomly sample 1/n
        points within each chunk'''
        sorted_i = np.argsort(variable)


    def sort_by_energy(self):
        self.energies = np.array(self.energies)
        self.sorted_i = np.argsort(self.energies)
        self.energies = self.energies[self.sorted_i]
        self.coords = self.order_array(self.coords, self.sorted_i)
        self.forces = self.order_array(self.forces, self.sorted_i)
        return self

    def order_array(self, np_list, sorted_i):
        np_list = self.get_3D_array(np_list)
        np_list = np_list[sorted_i]
        return np_list

    def get_3D_array(self, np_list):
        np_list = np.reshape(np.vstack(np_list), (-1,len(self.atoms),3))
        return np_list

    def get_2D_array(self, np_list):
        return np.vstack(np_list)
