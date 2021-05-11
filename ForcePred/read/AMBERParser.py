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
        coords = Universe(self.amb_top, self.amb_coords, format='TRJ')
        forces = Universe(self.amb_top, self.amb_forces, format='TRJ')
        self.atoms = [Z_[tp.name[0]] for tp in coords.atoms]

        n_atoms = len(coords.trajectory)
        #print(self.atoms, n_atoms, len(forces.trajectory))

        self.coords = self.get_3D_array(coords.trajectory, n_atoms) #/ \
                #Converter.au2Ang
        self.forces = self.get_3D_array(forces.trajectory, n_atoms) #/ \
                #Converter.au2kcalmola

    def get_3D_array(self, np_list, n_structures):
        return np.reshape(np.vstack(np_list), (n_structures,-1,3))

    def get_2D_array(self, np_list):
        return np.vstack(np_list)

       
