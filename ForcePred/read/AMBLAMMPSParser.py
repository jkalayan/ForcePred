#!/usr/bin/env python

'''
This module is for reading in forces, coordinates and energies from 
LAMMPS data files. 
'''

import MDAnalysis       
from MDAnalysis import *
import MDAnalysis.transformations as trans
import numpy as np
#from ..calculate.Converter import Converter


class AMBLAMMPSParser(object):
    '''
    Ensure unwrapped atom coordinates 9xu,yu,zu are used from 
    LAMMPS MD simulations.
    '''

    def __init__(self, amb_top, amb_coords, lammps_coords, 
            lammps_forces, lammps_ens, molecule):
        self.amb_top = amb_top
        self.amb_coords = amb_coords
        self.lammps_coords = lammps_coords
        self.lammps_forces = lammps_forces
        self.lammps_ens = lammps_ens
        self.filenames = [amb_top, amb_coords, lammps_coords, 
                lammps_forces, lammps_ens]
        self.atoms = []
        self.coords = []
        self.forces = []
        self.energies = []
        self.atom_potentials = []
        self.atom_kinetics = []
        self.dimensions = []
        self.sorted_i = None
        self.read_files()
        molecule.get_ZCFE(self) #populate molecule class
        #molecule.sort_by_energy()

    def __str__(self):
        return ('\nMD files: %s, \natoms: %s, N atoms: %s, ' \
                '\nN structures: %s, \ncoords: %s, \nforces: %s, ' \
                '\natom potentials: %s, \natom kinetics: %s, '\
                '\nenergies: %s, \ndimensions: %s' % 
                (self.filenames,
                ' '.join(map(str, self.atoms)), 
                len(self.atoms), len(self.coords), self.coords.shape,
                self.forces.shape, self.atom_potentials.shape, 
                self.atom_kinetics.shape, self.energies.shape,
                self.dimensions.shape))

    def read_files(self):
        Z_ = {'H':1, 'C':6, 'O':8}
        topology = Universe(self.amb_top, self.amb_coords, format='TRJ')
        self.atoms = [Z_[tp.name[0]] for tp in topology.atoms]

        coords = Universe(self.amb_top, self.lammps_coords, 
                atom_style='id type x y z', format='LAMMPSDUMP')
        forces = Universe(self.amb_top, self.lammps_forces, 
                atom_style='id type x y z', format='LAMMPSDUMP')
        energies = Universe(self.amb_top, self.lammps_ens, 
                atom_style='id type x y z', format='LAMMPSDUMP') 
                #new file type, where x=PE, y=KE

        n_atoms = len(coords.trajectory)
        for f in range(len(coords.trajectory)):
            t = coords.trajectory[f]
            self.dimensions.append(np.array(t.dimensions[0:3]))

        self.dimensions = self.get_2D_array(self.dimensions)
        self.coords = (self.get_3D_array(coords.trajectory, 
                len(self.dimensions)) /  self.dimensions[:, None]) #/ \
                #Converter.au2Ang
        self.forces = (self.get_3D_array(forces.trajectory, 
                len(self.dimensions)) /  self.dimensions[:, None]) #/ \
                #Converter.au2kcalmola
        atom_energies = (self.get_3D_array(energies.trajectory, 
                len(self.dimensions)) /  self.dimensions[:, None]) #/ \
                #Converter.Eh2kcalmol
        self.atom_potentials = \
                atom_energies.reshape((-1,3))[:,0].reshape(n_atoms,-1)
        self.atom_kinetics = \
                atom_energies.reshape((-1,3))[:,1].reshape(n_atoms,-1)
        self.energies = np.sum(self.atom_potentials, axis=1)

    def get_3D_array(self, np_list, n_structures):
        return np.reshape(np.vstack(np_list), (n_structures,-1,3))

    def get_2D_array(self, np_list):
        return np.vstack(np_list)

    def centre_in_box(coords):
        '''
        Not used and unwrap + centre doesn't work for lammps... 
        '''
        all_ = coords.atoms
        transforms = [trans.unwrap(all_),
                      trans.center_in_box(all_, center='mass'),]
        coords.trajectory.add_transformations(*transforms)

        with MDAnalysis.Writer('test.xyz', all_.n_atoms) as W:
            for ts in coords.trajectory:
                W.write(all_)
       
