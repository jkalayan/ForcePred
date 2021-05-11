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

import numpy as np
from ..calculate.Plotter import Plotter
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
        self.coords = self.get_3D_array(other.coords)
        self.forces = self.get_3D_array(other.forces)
        if hasattr(other, 'energies'):
            self.energies = self.get_2D_array(other.energies)

    def check_force_conservation(self):
        ''' Ensure that forces in each structure translationally and 
        rotationally sum to zero (energy is conserved) i.e. translations
        and rotations are invariant. '''
        unconserved = []
        for s in range(len(self.forces)):
            cm = np.average(self.coords[s], axis=0)
            for x in range(3):
                #translations
                x_trans_sum = abs(round(np.sum(self.forces[s],axis=0)[x], 1))
                #print(x_trans_sum)
                if abs(x_trans_sum) != 0 and s not in unconserved:
                    unconserved.append(s)
            #rotations
            i_rot_sum = np.zeros((3))
            for i in range(len(self.forces[s])):
                r = self.coords[s][i] - cm
                diff = np.cross(r, self.forces[s][i])
                i_rot_sum = np.add(i_rot_sum, diff)
            i_rot_sum = np.round(np.abs(i_rot_sum), 1)
            #print(i_rot_sum)
            if np.all(i_rot_sum != 0) and s not in unconserved:
                unconserved.append(s)
        if len(unconserved) != 0:
            print('Warning: structures %s are variant, '\
                    'removing from dataset shaped %s.' % 
                    (unconserved, self.coords.shape))
            Molecule.remove_variants(self, unconserved)
            print('New dataset shape is', self.coords.shape)

    def remove_variants(self, unconserved):
        for s in unconserved:
            self.coords = np.delete(self.coords, s, axis=0)
            self.forces = np.delete(self.forces, s, axis=0)
            if hasattr(self, 'energies'):
                self.energies = np.delete(self.energies, s, axis=0)

    def make_train_test(molecule, variable):
        '''Split input data into training and test sets. Data is 
        ordered by a particular variable, outputted are lists of
        indices for each set.'''
        sorted_i = np.argsort(variable)
        _Nstructures = len(variable)
        _Ntrain = int(_Nstructures / 4)
        #a = np.array(range(0,_Nstructures))
        a = sorted_i
        b = np.where(a % int(_Nstructures/_Ntrain+1),-1,a)
        c = np.where(a % int(_Nstructures/_Ntrain+1),a,-1)
        molecule.train = b[(b>=0)]
        molecule.test = c[(c>0)]

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
