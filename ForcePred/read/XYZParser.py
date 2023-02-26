#!/usr/bin/env python

'''
This module is for reading in forces, coordinates and energies from 
xyz and txt output files. 
'''

import numpy as np
from itertools import islice

class XYZParser(object):
    '''
    '''

    def __init__(self, file_atoms, files_coords, files_forces, 
            files_energies, molecule):
        self.file_atoms = file_atoms
        self.files_coords = files_coords
        self.files_forces = files_forces
        self.filenames = [file_atoms, files_coords, files_forces, 
                files_energies]
        self.atoms = []
        self.coords = []
        self.forces = []
        self.energies = []
        self.sorted_i = None
        self.iterate_atoms_file(file_atoms)
        self.natoms = len(self.atoms)
        self.coords = self.iterate_xyz_files(files_coords, self.coords)
        self.forces = self.iterate_xyz_files(files_forces, self.forces)
        self.energies = self.iterate_files(files_energies, self.energies)
        molecule.get_ZCFE(self) #populate molecule class

    def __str__(self):
        return ('\nxyz files: %s, \natoms: %s, N atoms: %s, ' \
                '\nN structures: %s' % 
                (self.filenames,
                ' '.join(map(str, self.atoms)), 
                len(self.atoms), len(self.coords)))

    def iterate_atoms_file(self, file_atoms):
        input_ = open(file_atoms, 'r')
        for atom in input_:
            self.atoms.append(int(atom))

    def iterate_files(self, filenames, var):
        n_atoms = len(self.atoms)
        for f in filenames:
            v = np.reshape(np.loadtxt(f), (-1,1))
            var.append(v)
        if len(filenames) > 0:
            var = self.get_2D_array(var)
        return var

    def iterate_xyz_files(self, filenames, var):
        for f in filenames:
            input_ = open(f, 'r')
            for line in input_:
                xyz = XYZParser.clean(self, XYZParser.extract(self, 1, input_))
                var.append(xyz)
        if len(filenames) > 0:
            var = XYZParser.get_3D_array(self, var, self.natoms)
        return var

    def extract(self, padding, input_):
        return (list(islice(input_, padding + 
                self.natoms))[-self.natoms:])

    def clean(self, raw):
        cleaned = np.empty(shape=[self.natoms, 3])
        for i, atom in enumerate(raw):
            cleaned[i] = atom.strip('\n').split()[-3:]
        #print(cleaned.shape)
        return np.array(cleaned)

    def get_3D_array(self, np_list, n_atoms):
        return np.reshape(np.vstack(np_list), (-1,n_atoms,3))

    def get_2D_array(self, np_list):
        return np.vstack(np_list)
