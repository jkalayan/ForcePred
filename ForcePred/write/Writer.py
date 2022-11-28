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
    
    def write_csv(list_, filename, header, delimiter=',', ext='csv'):
        np.savetxt('{}.{}'.format(filename, ext), np.column_stack(list_), 
                delimiter=delimiter, header=header, fmt='%.6f')

    def write_amber_inpcrd(coords, filename):
        natoms = len(coords)
        inpcrd_file = open('{}.inpcrd'.format(filename), 'w')
        inpcrd_file.write('{}\n'.format('default_name')) 
        inpcrd_file.write('{:5d}\n'.format(natoms)) 
        coords.reshape(-1,6)
        for i in coords:
            for j in i:
                inpcrd_file.write('{:12.7f}'.format(j))
            inpcrd_file.write('\n')
        inpcrd_file.close()

    def write_pdb(coords, resname, resid, atoms, filename, open_type):
        outfile = open(filename, open_type)
        for i in range(len(atoms)):
            atom_name = '{}{}'.format(Converter._ZSymbol[atoms[i]], i+1)
            outfile.write('\n'.join(['{:6s}{:5d} {:^4s}{:1s}{:3s}' \
                    ' {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}' \
                    '{:6.0f}{:6.0f}          {:>2s}{:2s}'.format(
                        'ATOM', i+1, atom_name, ' ', resname[0:3], 
                        'A', int(str(resid)[0:4]), ' ', 
                        coords[i][0], coords[i][1], coords[i][2], 
                        1, 1, ' ', ' ')]) + '\n')
        outfile.write('TER\n')
        outfile.close()

