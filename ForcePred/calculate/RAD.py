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
        self.RAD = []
        self.HB = []
        self.neighbour_index = None
        self.neighbour_r = None


def getHB(atom_index, atom_coords, neighbours, system, dimensions):
    '''
    Foor
    '''

    print(system.masses[atom_index], system.charges[atom_index])
    bonded_heavy = system.select_atoms('mass 2 to 999 and not index {} and '\
            'bonded index {}'.format(atom_index, atom_index))
    print(bonded_heavy.indices)
    neighbours = system.select_atoms('mass 2 to 999 and not index {} and '\
            'not bonded index {}'.format(bonded_heavy.indices[0], 
            bonded_heavy.indices[0]))
    RAD_heavy = getUALevelRAD(bonded_heavy.indices[0], 
            bonded_heavy.positions[0], neighbours, system, dimensions)
    print(RAD_heavy)
    RAD_all = []
    for A in RAD_heavy:
        RAD_all.append(A)
        bonded = system.select_atoms('bonded index {}'.format(A))
        for b in bonded:
            if b.mass < 1.1:
                RAD_all.append(b.index)
            else:
                continue
    print(RAD_all)



def getUALevelRAD(atom_index, atom_coords, neighbours, system, dimensions,
        RAD_all):
    '''
    '''
    sorted_indices, sorted_distances = getNeighbourList(atom_coords, 
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
            costheta_jik = angle(j_coords, i_coords, k_coords, 
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

    #RAD_all = []
    for A in RAD_heavy:
        if A not in RAD_all:
            RAD_all.append(A)
        bonded = system.select_atoms('bonded index {}'.format(A))
        for b in bonded:
            if b.mass < 1.1:
                if b.index not in RAD_all:
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




def pdbGenerate(all_data, fileName, dimensions):
    '''
    Output a pdb file to visualise different waters in system.
    Might have to also make an equivalent .csv file so that
    we have an easier time analysing the files.
    But append to one global file rather than each frame, 
    so need a column of frame number for analysis later.
    pdb file pure for producing images with vmd, 
    but could create these later anyway? 
    Then we can choose our beta column?
    '''

    data = open(fileName+'_RAD.pdb', "w") 
        #solute resid, int(dist) from solute resid

    file_list = [data, data3, data6]
    
    for x in range(0, len(all_data)):
        atom = all_data[x]


        try:
            a = int(atom.nearestAnyNonlike[0].resid)
        except TypeError:
            a = 0
        try:
            b = int(atom.nearestAnyNonlike[1])
        except TypeError:
            b = 0
        try:
            e = len(atom.RAD)
        except TypeError:
            e = 0
        try:
            As = len(atom.RAD_shell_ranked[1])
            Ds = len(atom.RAD_shell_ranked[2])
            DAhh_type = '%s%s' % (Ds, As)
            f = str(DAhh_type)
        except TypeError:
            f = '00'
        try:
            i = int(atom.hydrationShellRAD_solute)
        except (TypeError, ValueError):
            i = 0

        data.write("\n".join(["{:6s}{:5d} {:^4s}{:1s}{:3s}" \
                " {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}" \
                "{:6.0f}{:6.0f}          {:>2s}{:2s}".format(
                    "ATOM", int(str(atom.atom_num)[0:5]), 
                    atom.atom_name, " ", atom.resname[0:3], 
                    "A", int(str(atom.resid)[0:4]), " ", 
                    float(atom.coords[0]), 
                    float(atom.coords[1]), 
                    float(atom.coords[2]), 
                    a, b, " ", " ")]) + "\n")


    for d in file_list:
        d.close()


