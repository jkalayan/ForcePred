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

from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, AMBLAMMPSParser, AMBERParser, XYZParser, Binner, \
        Writer, Plotter, Network

import sys
#import numpy as np

#import os
#os.environ['OMP_NUM_THREADS'] = '8'


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        list_files='list_files'):

    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()
    print('\nread in Gaussian .out file/s:')
    OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt


    bias_type = '1' #1, 1/r, 1/r2, NRF, r
    help(Converter.get_interatomic_charges)
    print('\nget decomposed pairwise charges with bias {}'.format(bias_type))
    Converter.get_interatomic_charges(molecule, bias_type=bias_type)



    '''
    XYZParser(atom_file, coord_files, force_files, 
            energy_files, molecule)
    '''

    '''
    Writer.write_xyz(molecule.coords, molecule.atoms, 'coords.xyz', 'w')
    Writer.write_xyz(molecule.forces, molecule.atoms, 'forces.xyz', 'w')
    np.savetxt('energies.txt', molecule.energies)
    np.savetxt('charges.txt', molecule.charges)
    '''

    sys.stdout.flush()

    dihedrals = Binner()
    list_dih = [[7, 1, 4, 9]]
    dihedrals.get_dih_pop(molecule.coords, list_dih)
    atom_names = [Converter._ZSymbol[i] for i in molecule.atoms]
    Writer.write_csv((dihedrals.phis.T[0], molecule.energies.flatten(), 
            molecule.charges), 
            'dih_E_Q', 'OCCO_dih,energy,' + ','.join(atom_names))

    #Plotter.xy_scatter([dihedrals.phis.T[0].flatten()], 
            #[molecule.energies.flatten()], 
            #['$\\tau$ vs E'], ['k'], '$\\tau$', 'E / kJ mol$^-1$', 
            #'dih_vs_E')


    run_net = True
    split = 30 #2 4 5 20 52 260
    if run_net:
        print('\nget train and test sets, '\
                'training set is 1/{} of points'.format(split))
        Molecule.make_train_test(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        print('\nget ANN model')
        input = dihedrals.phis.T[0].reshape(-1,1)
        #input = molecule.mat_NRF 
        output = molecule.energies 
        #output = molecule.mat_Q
        #print('input', input)
        #print('output', output)

        network = Network() #initiate network class
        Network.get_variable_depth_model(network, molecule, input, output)
        #'''
    sys.stdout.flush()

    print('\nAtom pairs:')
    print('i indexA indexB - atomA atomB')
    _N = -1
    for i in range(len(molecule.atoms)):
        for j in range(i):
            _N += 1
            print(_N+1, i+1, j+1, '-', Converter._ZSymbol[molecule.atoms[i]], 
                    Converter._ZSymbol[molecule.atoms[j]])


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
            energy_files=op.energy_files, list_files=op.list_files)

if __name__ == '__main__':
    main()


