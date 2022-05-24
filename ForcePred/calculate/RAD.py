#!/usr/bin/env python

'''
These functions calculate RAD shells. 
'''

import MDAnalysis       
from MDAnalysis import *
import numpy as np
import sys


class RAD(object):
    '''
    '''
 
    def __init__(self):
        RAD_all = RAD.getUALevelRAD(bonded_heavy.indices[0], 
                bonded_heavy.positions[0], neighbours, system, dimensions)


    def getUALevelRAD(atom_index, atom_coords, neighbours, system, dimensions):
        '''
        '''
        sorted_indices, sorted_distances = RAD.getNeighbourList(atom_coords, 
                neighbours, dimensions)
        range_limit = len(sorted_distances)
        if range_limit > 30:
            range_limit = 30
        RAD_heavy = []
        i = atom_index
        i_coords = atom_coords
        count = -1
        for y in sorted_indices[:range_limit]:
            count += 1
            y_index = np.where(sorted_indices == y)[0][0]
            j = system.indices[y]
            j_coords = system.positions[y]
            rij = sorted_distances[y_index]
            blocked = False
            for z in sorted_indices[:count]: #only closer UAs can block
                z_index = np.where(sorted_indices == z)[0][0]
                k = system.indices[z]
                k_coords = system.positions[z]
                rik = sorted_distances[z_index]
                costheta_jik = RAD.angle(j_coords, i_coords, k_coords, 
                        dimensions[:3])
                if np.isnan(costheta_jik):
                    break
                LHS = (1/rij) ** 2 
                RHS = ((1/rik) ** 2) * costheta_jik
                if LHS < RHS:
                    blocked = True
                    break
            if blocked == False:
                RAD_heavy.append(j)

        RAD_all = []
        for A in RAD_heavy:
            RAD_all.append(A)
            bonded = system.select_atoms('bonded index {}'.format(A))
            for b in bonded:
                if b.mass < 1.1:
                    RAD_all.append(b.index)
                else:
                    continue
        return RAD_all


    #def get_r(coordsA, coordsB):
        #return np.linalg.norm(coordsA-coordsB)


    def getNeighbourList(atom, neighbours, dimensions):
        '''
        Use MDAnalysis to get distances between an atom and neighbours within
        a given cutoff. Each atom index pair sorted by distance are outputted.

        Parameters
        __________
        atom : (3,) array of an atom coordinates.
        neighbours : MDAnalysis array of heavy atoms in the system, 
            not the atom itself and not bonded to the atom.
        dimensions : (6,) array of system box dimensions.
        '''
        pairs, distances = MDAnalysis.lib.distances.capped_distance(atom, 
                neighbours.positions, max_cutoff=9e9, min_cutoff=None, 
                box=dimensions, method=None, return_distances=True)
        sorted_distances, sorted_indices = \
                zip(*sorted(zip(distances, neighbours.indices), 
                key=lambda x: x[0]))
        return np.array(sorted_indices), np.array(sorted_distances)


    def angle(a, b, c, dimensions):
        '''
        Get the angle between three atoms, taking into account PBC.

        Parameters
        __________
        a, b, c : (3,) arrays of atom cooordinates
        dimensions : (3,) array of system box dimensions.
        '''
        ba = np.abs(a - b)
        bc = np.abs(c - b)
        ac = np.abs(c - a)
        ba = np.where(ba > 0.5 * dimensions, ba - dimensions, ba)
        bc = np.where(bc > 0.5 * dimensions, bc - dimensions, bc)
        ac = np.where(ac > 0.5 * dimensions, ac - dimensions, ac)
        dist_ba = np.sqrt((ba ** 2).sum(axis=-1))
        dist_bc = np.sqrt((bc ** 2).sum(axis=-1))
        dist_ac = np.sqrt((ac ** 2).sum(axis=-1))
        cosine_angle = ((dist_ac ** 2 - dist_bc ** 2 - dist_ba ** 2) /
                (-2 * dist_bc * dist_ba))
        return cosine_angle
