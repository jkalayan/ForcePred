#!/usr/bin/env python

'''
This module is for binning variables against energy to see how well
sampled configurations are.
'''

import numpy as np
from .Converter import Converter
from .Plotter import Plotter
from ..write.Writer import Writer

import matplotlib.pyplot as plt

class Binner(object):
    '''
    '''
 
    def __init__(self):
        self.dihs = []
        self.phis = []
        self.confs = []
        self.angles = []
        self.thetas = []
        self.bonds = []
        self.rs = []

    def __str__(self):
        return ('\nN structures: %s,' % (len(self.coords)))

    def get_bond_pop(self, molecule, bonds):
        for i in range(len(molecule.coords)):
            for bond in bonds:
                bond = np.array(bond)-1 #index correctly
                v = molecule.coords[i][bond[1]] - molecule.coords[i][bond[0]]
                r = np.linalg.norm(v)
                self.rs.append(r)
        self.rs = np.array(self.rs).reshape(-1,len(bonds))

    def get_angle_pop(self, molecule, angles):
        '''angles is a list of lists containing atom indices involved 
        in angles of interest'''
        self.angles = np.array(angles)
        for i in range(len(molecule.coords)):
            for angle in angles:
                angle = np.array(angle)-1 #index correctly
                theta = Binner.get_angles(molecule.coords[i], angle)
                self.thetas.append(theta)
        self.thetas = np.array(self.thetas).reshape(-1,len(angles))

    def get_dih_pop(self, molecule, dihs):
        ''' dihs is a list of lists. Each list is of the atom indices
        involved in the dihedral of interest. '''
        self.dihs = np.array(dihs)
        for i in range(len(molecule.coords)):
            for dih in dihs:
                dih = np.array(dih)-1 #index correctly
                phi = Binner.get_dih_angles(molecule.coords[i], dih)
                self.phis.append(phi)
                self.confs.append(Binner.get_conf(phi))
        self.phis = np.array(self.phis).reshape(-1,len(dihs))
        self.confs = np.array(self.confs).reshape(-1,len(dihs))
        molecule.phis = self.phis

        '''
        sorted_i = np.argsort(molecule.energies)
        #print(sorted_i)
        #np_list = np.vstack(molecule.energies)
        #np_list = np_list[sorted_i]
        #print(np_list)
        print(self.dihs)

        #print(self.phis[:200])
        #print(self.confs[:200])

        for i in range(1, 50):
            print(sorted_i[-i], molecule.energies[sorted_i[-i]], 
                    self.phis[sorted_i[-i]], 
                    self.confs[sorted_i[-i]])

        phis_1 = [i[0] for i in self.phis]
        phis_2 = [i[1] for i in self.phis]
        phis_3 = [i[2] for i in self.phis]
        _Es = molecule.energies
        Plotter.xyz_scatter(phis_1, phis_2, _Es, 
                '$\phi_1$', '$\phi_2$', '$U$', 'xyz_12')
        Writer.write_xyz([molecule.coords[sorted_i[-1]]], molecule.atoms, 
                'TS-1-2.xyz')
        '''

    def get_angles(coords, angle):
        v1 = coords[angle[0]] - coords[angle[1]]
        v2 = coords[angle[2]] - coords[angle[1]]
        r1 = np.linalg.norm(v1)
        r2 = np.linalg.norm(v2)
        val = np.divide(np.dot(v1, v2), np.multiply(r1, r2))
        angle = np.arccos(val)
        return np.degrees(angle)

    def get_dih_angles(coords, dih):
        #get normalised vectors
        norms = []
        for i in range(len(dih)-1):
            v = coords[dih[i]] - coords[dih[i+1]]
            r = np.linalg.norm(v)
            v_norm = v / r
            norms.append(v_norm)
        ## get normals
        n1 = np.cross(norms[0], norms[1])
        n2 = np.cross(norms[1], norms[2])
        m = np.cross(n1, norms[1])
        #get dot products
        x = np.dot(n1, n2)
        y = np.dot(m, n2)
        phi = Converter.rad2deg * np.arctan2(y, x)
        return int(phi)

    def get_conf(phi):
        if phi >= 120 or phi < -120:
            conf = 't'
        if phi >= 0 and phi < 120:
            conf = 'g-'
        if phi >= -120 and phi < 0:
            conf = 'g+'
        return conf


