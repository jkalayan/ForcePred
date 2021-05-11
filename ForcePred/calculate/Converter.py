#!/usr/bin/env python

'''
This module is for converting forces and coordinates/nuclear charges 
into pair-wise and nuclear repulsive forces respectively.
'''

import numpy as np

class Converter(object):
    '''
    Takes coords, forces, atoms from Molecule class 
    and converts into desired format
    e.g. interatomic foces, nuclear repulsive forces,
    '''

    #constants
    au2Ang = 0.529177 #bohr to Angstrom
    kcal2kj = 4.184
    Eh2kcalmol = 627.5095 #hartree to kcal/mol
    Eh2kJmol = Eh2kcalmol * kcal2kj #hartree to kJ/mol
    au2kJmola = Eh2kJmol * au2Ang #convert from Eh/a_0 to kJ/(mol Ang)
    au2kcalmola = Eh2kcalmol * au2Ang #convert from Eh/a_0 to kcal/(mol Ang)
    rad2deg = float(180) / float(np.pi) #radians to degrees
    ang2m = 1e-10
    m2ang = 1e10
    fsec2sec = 1e-15
    _NA = 6.02214076e23
    au2kg = 1.66053907e-27
    _ZM = {1:1.008, 6:12.011, 8:15.999}
 
    def __init__(self, molecule):
        molecule.check_force_conservation()
        self.coords = molecule.coords
        self.forces = molecule.forces
        self.atoms = molecule.atoms
        self.mat_r = None
        self.mat_NRF = None
        self.get_interatomic_forces(molecule)

    def __str__(self):
        return ('\nN structures: %s,' % (len(self.coords)))

    def get_interatomic_forces(self, molecule):
        if len(self.atoms) == len(self.coords[0]):
            n_atoms = len(self.atoms)
            _NC2 = int(n_atoms * (n_atoms-1)/2)
            n_structures = len(self.coords)
            self.mat_r = np.zeros((n_structures, _NC2))
            self.mat_NRF = np.zeros((n_structures, _NC2))
            mat_vals = np.zeros((n_structures, n_atoms, 3, _NC2))
            mat_F = [] 
            for s in range(n_structures):
                _N = -1
                for i in range(n_atoms):
                    zi = self.atoms[i]
                    for j in range(i):
                        _N += 1
                        zj = self.atoms[j]
                        r = Converter.get_r(self.coords[s][i], 
                                self.coords[s][j])
                        self.mat_r[s,_N] = r
                        self.mat_r[s,_N] = r
                        if i != j:
                            self.mat_NRF[s,_N] = Converter.get_NRF(zi, zj, r)
                        for x in range(0, 3):
                            val = ((self.coords[s][i][x] - 
                                    self.coords[s][j][x]) /
                                    self.mat_r[s,_N]) * self.mat_NRF[s,_N]
                            mat_vals[s,i,x,_N] = val
                            mat_vals[s,j,x,_N] = -val

                mat_vals2 = mat_vals[s].reshape(n_atoms*3,_NC2)
                forces2 = self.forces[s].reshape(n_atoms*3)
                _F = np.matmul(np.linalg.pinv(mat_vals2), forces2)
                ##### if using scale factor, need to remove.
                _N2 = -1
                for i in range(n_atoms):
                    for j in range(i):
                        _N2 += 1
                        _F[_N2] = _F[_N2] * self.mat_NRF[s,_N2]
                mat_F.append(_F)
                #####

            mat_F = np.reshape(np.vstack(mat_F), (n_structures,_NC2))
            molecule.mat_F = mat_F
            molecule.mat_NRF = self.mat_NRF
            #print(molecule.mat_F.shape)
            #print(molecule.mat_NRF.shape)

        else:
            raise ValueError('Number of atoms does not match '\
                    'number of coords.')

    def get_NRF(zA, zB, r):
        return zA * zB * Converter.au2kcalmola / (r ** 2)
        #return zA * zB / (r ** 2) 

    def get_r(coordsA, coordsB):
        return np.linalg.norm(coordsA-coordsB)




