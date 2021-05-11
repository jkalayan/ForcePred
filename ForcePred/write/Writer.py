#!/usr/bin/env python

'''
This module is for writing info into a particular format. 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Writer(object):
    '''
    '''

    def write_xyz(coords, atoms, filename):
        xyz_file = open(filename, 'w')
        for molecule in range(len(coords)):
            xyz_file.write('{}\n{}\n'.format(len(atoms), molecule+1))
            for atom in range(len(atoms)):
                x = coords[molecule][atom][0]
                y = coords[molecule][atom][1]
                z = coords[molecule][atom][2]
                xyz_file.write('{:4} {:11.6f} {:11.6f} {:11.6f}\n'.format(
                        atoms[atom], x, y, z))
        xyz_file.close()



        
