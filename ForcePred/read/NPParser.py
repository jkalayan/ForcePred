#!/usr/bin/env python

'''
This module is for reading in forces, coordinates and energies from 
numpy output files. 
'''

import numpy as np


class NPParser(object):
    '''
    '''

    def __init__(self, file_atoms, files_coords, files_forces, 
            files_energies, molecule):
        self.file_atoms = file_atoms
        self.files_coords = files_coords
        self.files_forces = files_forces
        self.filenames = [file_atoms, files_coords, files_forces]
        self.atoms = []
        self.energies = []
        self.coords = []
        self.forces = []
        self.sorted_i = None
        self.iterate_atoms_file(file_atoms)
        self.energies = self.iterate_energies(files_energies, self.energies)
        self.coords = self.iterate_files(files_coords, self.coords)
        self.forces = self.iterate_files(files_forces, self.forces)
        molecule.get_ZCFE(self) #populate molecule class

    def __str__(self):
        return ('\nnumpy files: %s, \natoms: %s, N atoms: %s, ' \
                '\nN structures: %s' % 
                (self.filenames,
                ' '.join(map(str, self.atoms)), 
                len(self.atoms), len(self.coords)))

    def iterate_atoms_file(self, file_atoms):
        input_ = open(file_atoms, 'r')
        for atom in input_:
            self.atoms.append(int(atom))

    def iterate_energies(self, filename, var):
        n_atoms = len(self.atoms)
        for filename in filename:
            v = np.reshape(np.loadtxt(filename), (-1,1))
            var.append(v)
        return var

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
       
