#!/usr/bin/env python

'''
This module is for finding symmetric atoms and ordering them according
to the closest neighbouring atom which breaks symmetry.
'''

import numpy as np
from scipy.spatial import distance

class Permuter(object):
    '''
    Takes coords, finds distance matrix
    '''
 
    def __init__(self, molecule):
        self.atoms = molecule.atoms
        self.coords = molecule.coords
        self.get_distances()


    def get_distances(self):
        print(self.atoms)
        for coords in self.coords:
            print(coords)
            mat_dist = distance.cdist(coords, coords, 'euclidean')
            print(mat_dist)
            t = np.select([mat_dist < 0.5, mat_dist < 2], 
                    [0, 1])
            #t = t.astype(int)
            print(t)
            print()


