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

mdanal = False
if mdanal:
    from ForcePred.read.AMBLAMMPSParser import AMBLAMMPSParser
    from ForcePred.read.AMBERParser import AMBERParser
                               
from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, XYZParser, Binner, \
        Writer, Plotter, Conservation
from ForcePred.nn.Network_perminv import Network

from keras.models import Model, load_model    
import sys
#import numpy as np
#from itertools import islice

#import os
#os.environ['OMP_NUM_THREADS'] = '8'

import os
# Get number of cores reserved by the batch system
# ($NSLOTS is set by the batch system, or use 1 otherwise)
NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files', 
        dihedrals='dihedrals'):


    startTime = datetime.now()

    print(startTime)
    #molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()
    #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt

    scan1_molecule = Molecule() #initiate molecule class
    OPTParser(input_files, scan1_molecule, opt=True) #read in FCEZ for opt
    print(scan1_molecule)

    scan2_molecule = Molecule() #initiate molecule class
    OPTParser(input_files, scan2_molecule, opt=True) #read in FCEZ for opt

    scan3_molecule = Molecule() #initiate molecule class
    OPTParser(input_files, scan3_molecule, opt=True) #read in FCEZ for opt

    print(datetime.now() - startTime)
    sys.stdout.flush()


    bias_type='1/r'
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type)
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type)
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type)

    print(datetime.now() - startTime)
    sys.stdout.flush()


    scan1_measures = Binner()
    scan1_measures.get_dih_pop(scan1_molecule.coords, [[5,1,2,3]])

    scan2_measures = Binner()
    scan2_measures.get_bond_pop(scan2_molecule.coords, [[5,1]])

    scan3_measures = Binner()
    scan3_measures.get_bond_pop(scan3_molecule.coords, [[1,2]])

    print('---', scan1_measures.phis.shape)
    print(scan1_measures.phis.T.flatten()) 
    print(scan1_molecule.mat_FE[:,8].flatten().shape)
    print(scan1_molecule.energies.flatten().shape)

    count = 0
    for i in range(len(scan1_molecule.atoms)):
        for j in range(i):
            print(count, '-', i, j)
            count += 1


    Plotter.twinx_plot(
            [scan1_measures.phis.T.flatten()[1:], 
                scan1_measures.phis.T.flatten()[1:]], 
            [scan1_molecule.mat_FE[:,8].flatten()[1:], 
                scan1_molecule.energies.flatten()[1:]], 
            ['q', 'E'], ['k', 'dodgerblue'], 
            '$\\tau$ / degrees', '$q_{ij}$', '$E$ / kcal mol$^-1$', 
            [10, 10], 'scan-5123.pdf')

    '''
    Plotter.twinx_plot(
            [scan2_measures.rs.T.flatten(), scan2_measures.rs.T.flatten(),
                scan3_measures.rs.T.flatten(), scan3_measures.rs.T.flatten()], 
            [scan2_molecule.mat_FE.flatten(), 
                scan2_molecule.energies.flatten(), 
                scan3_molecule.mat_FE.flatten(), 
                scan3_molecule.energies.flatten()], 
            ['q', 'E', 'q', 'E'], ['k', 'dodgerblue', 'k', 'dodgerblue'], 
            '$\\tau$ / degrees', '$q_{ij}$', '$E$ / kcal mol$^-1$', 
            [10, 10, 10, 10], 'scan-51-12.pdf')
    '''



    print('end')
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
        group = parser.add_argument('-d', '--dihedrals', nargs='+', 
                action='append', type=int, default=[],
                help='list of dihedrals')
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
            dihedrals=op.dihedrals, list_files=op.list_files)

if __name__ == '__main__':
    main()


