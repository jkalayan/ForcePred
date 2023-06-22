#!/usr/bin/env python

'''
This module is used for binning variables against energy to see how well
sampled configurations are.
'''

import numpy as np
from .Converter import Converter
from .Plotter import Plotter
from ..write.Writer import Writer

import matplotlib.pyplot as plt

class Binner(object):
    '''
    This class is used to calculate bond properties, such as angles,
    dihedrals, etc. And then bin this data into a histogram.
    There are also some functions to calculate errors.
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

    def get_bond_pop(self, coords, bonds):
        '''
        For a given set of 3D coordinates and a list of atom pair indices,
        find the bond length between each atom pair. Save this in an array 
        of N structures * N pairs
        '''
        for i in range(len(coords)):
            for bond in bonds:
                bond = np.array(bond)#-1 #index correctly
                #v = coords[i][bond[1]] - coords[i][bond[0]]
                #r = np.linalg.norm(v)
                r = np.linalg.norm(coords[i][bond[1]] - 
                        coords[i][bond[0]])
                self.rs.append(r)
        self.rs = np.array(self.rs).reshape(-1,len(bonds))

    def get_angle_pop(self, coords, angles):
        '''
        For a given set of 3D coords and 3 atoms in the molecule, 
        calculate angles. Output shape = N structures * N angles
        Angles is a list of lists containing atom indices involved 
        in angles of interest
        '''
        self.angles = np.array(angles)
        for i in range(len(coords)):
            for angle in angles:
                angle = np.array(angle)#-1 #index correctly
                theta = Binner.get_angles(coords[i], angle)
                self.thetas.append(theta)
        self.thetas = np.array(self.thetas).reshape(-1,len(angles))

    def get_dih_pop(self, coords, dihs):
        '''
        as with bonds and angles but here for dihedrals.
        dihs is a list of lists. Each list is of the atom indices
        involved in the dihedral of interest. 
        '''
        self.dihs = np.array(dihs)
        for i in range(len(coords)):
            for dih in dihs:
                dih = np.array(dih)#-1 #index correctly
                phi = Binner.get_dih_angles(coords[i], dih)
                self.phis.append(phi)
                self.confs.append(Binner.get_conf(phi))
        self.phis = np.array(self.phis).reshape(-1,len(dihs))
        self.confs = np.array(self.confs).reshape(-1,len(dihs))
        #molecule.phis = self.phis


    def get_angles(coords, angle):
        '''
        Calculate the angle between three 3D coordinates.
        '''
        v1 = coords[angle[0]] - coords[angle[1]]
        v2 = coords[angle[2]] - coords[angle[1]]
        r1 = np.linalg.norm(v1)
        r2 = np.linalg.norm(v2)
        val = np.divide(np.dot(v1, v2), np.multiply(r1, r2))
        angle = np.arccos(val)
        return np.degrees(angle)

    def get_dih_angles(coords, dih):
        '''
        Calculate the dihedral angle.
        '''
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
        #return int(phi)
        return round(phi, 2)

    def get_conf(phi):
        '''
        Find the conformer based on what the dihedral angle value is.
        '''
        if phi >= 120 or phi < -120:
            conf = 't'
        if phi >= 0 and phi < 120:
            conf = 'g-'
        if phi >= -120 and phi < 0:
            conf = 'g+'
        return conf

    def get_scurve(baseline, values, filename):
        RSE = np.sqrt((baseline-values)**2)
        #print(np.amax(RSE))
        #print(np.sqrt(np.sum((baseline-values)**2)/values.shape[0]))
        #hist, bin_edges = np.histogram(RSE,1000,(-1,100))
        hist, bin_edges = np.histogram(RSE,1000,(-1,np.amax(RSE)))
        hist = np.cumsum(hist)
        #print(bin_edges.shape)
        bin_edges = bin_edges[range(1,bin_edges.shape[0])]
        #print(hist.shape)
        hist = hist/values.shape[0]*100
        np.savetxt(filename, np.column_stack((bin_edges,hist)))
        return bin_edges, hist

    def get_error(all_actual, all_prediction):
        '''Get total errors for array values.'''
        _N = np.size(all_actual)
        mae = 0
        rms = 0
        msd = 0 #mean signed deviation
        for actual, prediction in zip(all_actual, all_prediction):
            diff = prediction - actual
            mae += np.sum(abs(diff))
            rms += np.sum(diff ** 2)
            msd += np.sum(diff)
        mae = mae / _N
        rms = (rms / _N) ** 0.5
        msd = msd / _N
        return mae, rms, msd


    def get_each_error(actual, prediction):
        '''Get error for each array'''
        diff = prediction - actual
        mae = abs(diff)
        rms = diff ** 2
        return mae, rms

    def get_L1(all_actual, all_prediction, _N, threshold):
        '''get L1 values - the percentage of points above a threshold value'''
        all_actual = all_actual.flatten()
        all_prediction = all_actual.flatten()
        mae = np.zeros((all_actual.shape))
        rms = np.zeros((all_actual.shape))
        for actual, prediction, i in zip(all_actual, all_prediction, 
                range(len(all_actual))):
            diff = prediction - actual
            mae[i] = abs(diff)
            rms[i] = diff ** 2

    def get_hist(values, n_bins):
        '''Put values into a histogram based on n_bins'''
        hist, bin_edges = np.histogram(values, n_bins, density=True)
        bin_edges = bin_edges[range(1, bin_edges.shape[0])]
        return bin_edges, hist

