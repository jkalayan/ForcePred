#!/usr/bin/env python

'''
This module is for converting forces and coordinates/nuclear charges 
into pair-wise and nuclear repulsive forces respectively.
'''

from ..read.GaussianParser import OPTParser
import numpy as np

class Converter(object):
    '''
    Takes coords, forces and converts into desired format
    e.g. interatomic foces, nuclear repulsive forces,
    '''

    #constants
    au2Ang=0.529177 #bohr to Angstrom
    Eh2kJmol=627.5095*4.184 #hartree to kJ/mol
    au2kJmola=Eh2kJmol*au2Ang #convert from Eh/a_0 to kJ/(mol Ang)

    
    def __init__(self, coords, forces, atoms):
        self.coords = coords
        self.forces = forces
        self.atoms = atoms
        self.mat_r = None
        self.mat_NRE = None
        self.get_interatomic_forces()

    def __str__(self):
        return ('\nN structures: %s, ' % (len(self.coords)))

    def get_interatomic_forces(self):
        if len(self.atoms) == len(self.coords[0]):
            n_atoms = len(self.atoms)
            _NC2 = int(n_atoms * (n_atoms-1)/2)
            n_structures = len(self.coords)
            self.mat_r = np.zeros((n_structures, n_atoms, n_atoms))
            self.mat_NRF = np.zeros((n_structures, n_atoms, n_atoms))
            mat_vals = np.zeros((n_structures, n_atoms, 3, _NC2))
            mat_F = [] 
            for s in range(0, n_structures):
                _N = -1
                for i in range(0, n_atoms):
                    zi = self.atoms[i]
                    for j in range(0, i):
                        _N += 1
                        zj = self.atoms[j]
                        r = self.get_r(self.coords[s][i], self.coords[s][j])
                        self.mat_r[s,i,j] = r
                        self.mat_r[s,j,i] = r
                        if i != j:
                            self.mat_NRF[s,i,j] = self.get_NRF(zi, zj, r)
                        for x in range(0, 3):
                            val = ((self.coords[s][i][x] - 
                                    self.coords[s][j][x]) /
                                    self.mat_r[s,i,j]) #* self.mat_NRF[s,i,j]
                            mat_vals[s,i,x,_N] = val
                            mat_vals[s,j,x,_N] = -val

                mat_vals2 = mat_vals[s].reshape(n_atoms*3,_NC2)
                forces2 = self.forces[s].reshape(n_atoms*3)
                _F = np.matmul(np.linalg.pinv(mat_vals2), forces2)
                mat_F.append(_F)

            ##if using scale factor, need to remove.
            mat_F = np.reshape(np.vstack(mat_F), (n_structures,_NC2))

        else:
            raise ValueError('Number of atoms does not match '\
                    'number of coords.')

    def get_NRF(self, zA, zB, r):
        return zA * zB * Converter.au2kJmola / (r ** 2)
        #return zA * zB / (r ** 2) 

    def get_r(self, coordsA, coordsB):
        return np.linalg.norm(coordsA-coordsB)




