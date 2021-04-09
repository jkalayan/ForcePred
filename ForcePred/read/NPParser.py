#!/usr/bin/env python

'''
This module is for reading in forces, coordinates and energies from 
numpy output files. 
'''

import numpy as np


class NPParser(object):
    '''
    '''

    def __init__(self, file_atoms, files_coords, files_forces):
        self.file_atoms = file_atoms
        self.files_coords = files_coords
        self.files_forces = files_forces
        self.atoms = []
        self.energies = []
        self.coords = []
        self.forces = []
        self.sorted_i = None
        self.iterate_atoms_file(file_atoms)
        self.coords = self.iterate_files(files_coords, self.coords)
        self.forces = self.iterate_files(files_forces, self.forces)

    def __str__(self):
        return ('\nnumpy files: %s, %s, %s, \natoms: %s, N atoms: %s, ' \
                '\nN structures: %s' % 
                (self.file_atoms, ', '.join(self.files_coords),
                ', '.join(self.files_forces),
                ' '.join(map(str, self.atoms)), 
                len(self.atoms), len(self.coords)))

    def iterate_atoms_file(self, file_atoms):
        input_ = open(file_atoms, 'r')
        for atom in input_:
            self.atoms.append(int(atom))

    def iterate_files(self, filename, var):
        n_atoms = len(self.atoms)
        for filename in filename:
            v = np.reshape(np.loadtxt(filename), (-1,n_atoms,3))
            var.append(v)
        var = self.get_3D_array(var, n_atoms)
        return var

    def get_3D_array(self, np_list, n_atoms):
        np_list = np.reshape(np.vstack(np_list), (-1,n_atoms,3))
        return np_list
       
