#!/usr/bin/env python

'''
This module is used to store information required to predict forces:
    Nuclear charges, coordinates, forces and molecule energies.

'''

import numpy as np

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
        self.coords = other.coords
        self.forces = other.forces
        self.energies = other.energies

    def check_force_conservation(self):
        '''
        Ensure that forces in each structure translationally and 
        rotationally sum to zero (energy is conserved).
        '''
        unconserved = []
        for s in range(len(self.forces)):
            cm = np.average(self.coords[s], axis=0)
            for x in range(3):
                #translations
                x_trans_sum = round(np.sum(self.forces[s],axis=0)[x], 5)
                if abs(x_trans_sum) != 0 and s not in unconserved:
                    unconserved.append(s)
            #rotations
            i_rot_sum = 0
            for i in range(len(self.forces[s])):
                r = self.coords[s][i] - cm
                i_rot_sum += np.cross(r, self.forces[s][i])
            if np.all(np.round(i_rot_sum, 5) != 0) and s not in unconserved:
                unconserved.append(s)
        if len(unconserved) != 0:
            print('Warning: structures %s are variant, '\
                    'removing from dataset shaped %s.' % 
                    (unconserved, self.coords.shape))
            self.remove_variants(unconserved)
            print('New dataset shape is', self.coords.shape)

    def remove_variants(self, unconserved):
        for s in unconserved:
            self.coords = np.delete(self.coords, s, axis=0)
            self.forces = np.delete(self.forces, s, axis=0)
            self.energies = np.delete(self.energies, s, axis=0)

