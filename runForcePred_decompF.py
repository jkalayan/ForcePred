#!/usr/bin/env python

__author__ = 'Jas Kalayan'
__credits__ = ['Jas Kalayan', 'Ismaeel Ramzan', 
        'Neil Burton',  'Richard Bryce']
__license__ = 'GPL'
__maintainer__ = 'Jas Kalayan'
__email__ = 'jkalayan@gmail.com'
__status__ = 'Development'

from datetime import datetime
import argparse
import numpy as np

openmm = True
if openmm:
    from ForcePred.calculate.OpenMM import OpenMM
                               
from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, AMBLAMMPSParser, AMBERParser, XYZParser, Binner, \
        Writer, Plotter, Network, Conservation

from keras.models import Model, load_model    
import sys
#import numpy as np
#from itertools import islice

#import os
#os.environ['OMP_NUM_THREADS'] = '8'


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()
    #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt

    '''
    AMBERParser('molecules.prmtop', coord_files, force_files, 
            molecule)
    '''

    '''
    XYZParser(atom_file, coord_files, force_files, 
            energy_files, molecule)
    '''

    NPParser(atom_file, coord_files, force_files, energy_files, molecule)



    '''
    A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    indices, pairs_dict = Molecule.find_equivalent_atoms(molecule.atoms, A)
    print('\nZ', molecule.atoms)
    print('A', A) 
    print('indices', indices)
    print('pairs_dict', pairs_dict)

    sys.stdout.flush()
    #sys.exit()
    '''


    #A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    #print(molecule.atoms)
    #print(A)



    get_decompF = True
    if get_decompF:
        print(molecule)
        print('\ncheck and get force decomp')
        unconserved = Molecule.check_force_conservation(molecule) #
        Converter(molecule) #get pairwise forces
        print(datetime.now() - startTime)
        sys.stdout.flush()

    get_decompE = False
    if get_decompE:
        print('get mat_E')
        Converter.get_interatomic_energies(molecule, bias_type='1/r')
        print(datetime.now() - startTime)
        sys.stdout.flush()

    run_net = True
    split = 20 #2 4 5 20 52 260
    if run_net:
        train = round(len(molecule.coords) / split, 3)
        nodes = 1000
        input = molecule.mat_NRF
        output = molecule.mat_F
        print('\nget train and test sets, '\
                'training set is {} points.'\
                '\nNumber of nodes is {}'.format(train, nodes))
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        print('\nget ANN model')
        sys.stdout.flush()
        network = Network(molecule) #initiate network class
        Network.get_variable_depth_model(network, molecule, 
                nodes, input, output) #train NN

    sys.stdout.flush()


    print(datetime.now() - startTime)


def main():

    try:
        usage = 'runForcePred.py [-h]'
        parser = argparse.ArgumentParser(description='Program for reading '\
                'in molecule forces, coordinates and energies for '\
                'force prediction.', usage=usage, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group('Options')
        group = parser.add_argument('-i', '--input_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing forces '\
                'coordinates and energies.')
        group = parser.add_argument('-a', '--atom_file', 
                metavar='file', default=None,
                help='name of file/s containing atom nuclear charges.')
        group = parser.add_argument('-c', '--coord_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing coordinates.')
        group = parser.add_argument('-f', '--force_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing forces.')
        group = parser.add_argument('-e', '--energy_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing energies.')
        group = parser.add_argument('-q', '--charge_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing charges.')
        group = parser.add_argument('-l', '--list_files', action='store', 
                metavar='file', default=False,
                help='file containing list of file paths.')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    run_force_pred(input_files=op.input_files, atom_file=op.atom_file, 
            coord_files=op.coord_files, force_files=op.force_files, 
            energy_files=op.energy_files, charge_files=op.charge_files, 
            list_files=op.list_files)

if __name__ == '__main__':
    main()


