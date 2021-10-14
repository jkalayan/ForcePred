#!/usr/bin/env python

'''
This module is for reading in forces, coordinates and energies from 
LAMMPS data files. 
'''

import MDAnalysis       
from MDAnalysis import *
import numpy as np
#from ..calculate.Converter import Converter


class AMBERParser(object):
    '''
    Ensure unwrapped atom coordinates 9xu,yu,zu are used from 
    LAMMPS MD simulations.
    '''

    def __init__(self, amb_top, amb_coords, amb_forces, molecule):
        self.amb_top = amb_top
        self.amb_coords = amb_coords
        self.amb_forces = amb_forces
        self.filenames = [amb_top, amb_coords, amb_forces]
        self.atoms = []
        self.n_all_atoms = 0
        self.coords = []
        self.forces = []
        self.sorted_i = None
        self.read_files()
        molecule.get_ZCFE(self) #populate molecule class


    def __str__(self):
        return ('\nMD files: %s, \natoms: %s, N atoms: %s, ' \
                '\nN structures: %s, \ncoords: %s, \nforces: %s' % 
                (self.filenames,
                ' '.join(map(str, self.atoms)), 
                len(self.atoms), len(self.coords), self.coords.shape,
                self.forces.shape))

    def read_files(self):
        Z_ = {'H':1, 'C':6, 'O':8}
        '''
        read_coords = Universe(self.amb_top, self.amb_coords, format='TRJ')
        read_forces = Universe(self.amb_top, self.amb_forces, format='TRJ')
        self.atoms = [Z_[tp.name[0]] for tp in read_coords.atoms]

        for c, f in zip(read_coords.trajectory, read_forces.trajectory):
            self.coords.append(np.array(c.positions))
            self.forces.append(np.array(f.positions))
        '''

        topology = Universe(self.amb_top, self.amb_coords[0], format='TRJ')
        self.atoms = [Z_[tp.name[0]] for tp in topology.atoms]

        for _C, _F in zip(self.amb_coords, self.amb_forces):
            read_coords = Universe(self.amb_top, _C, format='TRJ')
            read_forces = Universe(self.amb_top, _F, format='TRJ')

            self.n_all_atoms += len(read_coords.trajectory)
            n_atoms = self.n_all_atoms
            for c, f in zip(read_coords.trajectory, read_forces.trajectory):
                self.coords.append(np.array(c.positions))
                self.forces.append(np.array(f.positions))


        self.coords = self.get_3D_array(self.coords, len(self.atoms))
        self.forces = self.get_3D_array(self.forces, len(self.atoms))

        '''
        n_atoms = len(read_coords.trajectory)
        #print(self.atoms, n_atoms, len(forces.trajectory))

        self.coords = self.get_3D_array(self.coords, n_atoms) #/ \
                #Converter.au2Ang
        self.forces = self.get_3D_array(self.forces, n_atoms) #/ \
                #Converter.au2kcalmola
        '''

    def get_3D_array(self, np_list, n_atoms):
        return np.reshape(np.vstack(np_list), (-1,n_atoms,3))

    def get_2D_array(self, np_list):
        return np.vstack(np_list)

       
