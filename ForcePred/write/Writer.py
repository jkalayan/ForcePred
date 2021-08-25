#!/usr/bin/env python

'''
This module is for writing info into a particular format. 
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..calculate.Converter import Converter

class Writer(object):
    '''
    '''

    def write_xyz(coords, atoms, filename, open_type, i=False):
        xyz_file = open(filename, open_type)
        for molecule in range(len(coords)):
            count = molecule
            if i:
                count = i
            xyz_file.write('{}\n{}\n'.format(len(atoms), count+1))
            for atom in range(len(atoms)):
                x = coords[molecule][atom][0]
                y = coords[molecule][atom][1]
                z = coords[molecule][atom][2]
                xyz_file.write('{:4} {:11.6f} {:11.6f} {:11.6f}\n'.format(
                        atoms[atom], x, y, z))
        xyz_file.close()

    def write_gaus_cart(coords, atoms, method_type, filename):
        symbols = [Converter._ZSymbol[atom] for atom in atoms]
        gaus_cart_file = open('{}.com'.format(filename), 'w')
        for molecule in range(len(coords)):
            if molecule != 0:
                gaus_cart_file.write('--Link1--\n')
            chkpoint = '%Chk={}.chk'.format(filename)
            method = '# B3LYP/6-31+G(d) {}'.format(method_type)
            title = '{} calculate {} {}'.format(molecule, method_type, 
                    filename)
            #assumes uncharged and multiplicity 1
            gaus_cart_file.write('{}\n{}\n\n{}\n\n0 1\n'.format(
                chkpoint, method, title)) 
            for atom in range(len(atoms)):
                x = coords[molecule][atom][0]
                y = coords[molecule][atom][1]
                z = coords[molecule][atom][2]
                gaus_cart_file.write(
                        '{:3} {:14.5f} {:14.5f} {:14.5f}\n'.format(
                        symbols[atom], x, y, z))
            gaus_cart_file.write('\n') 
        gaus_cart_file.close()

    def write_list(list_, filename, open_type):
        outfile = open(filename, open_type)
        for i in list_:
            outfile.write('{}\n'.format(i))
        outfile.close()
    
    def write_csv(list_, filename, header):
        np.savetxt('{}.csv'.format(filename), np.column_stack(list_), 
                delimiter=',', header=header, fmt='%.3f')
