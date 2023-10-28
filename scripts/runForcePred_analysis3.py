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

mdanal = True
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

import os
#os.environ['OMP_NUM_THREADS'] = '8'

NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()

    print('Load data')
    dataset_molecule = Molecule()
    NPParser(atom_file, [coord_files[0]], [force_files[0]], [energy_files[0]], 
            dataset_molecule)
    print('dataset', dataset_molecule.coords.shape, 
            dataset_molecule.forces.shape, dataset_molecule.energies.shape)
    print(dataset_molecule)

    model_molecule = Molecule()
    NPParser(atom_file, [coord_files[1]], [force_files[1]], [energy_files[1]], 
            model_molecule)
    print('model', model_molecule.coords.shape, model_molecule.forces.shape, 
            model_molecule.energies.shape)
    print(model_molecule)


    gaff_molecule = Molecule()
    AMBERParser('../amber/molecules.prmtop', ['../amber/3md.mdcrd'], 
            ['../amber/ascii3.frc'], gaff_molecule)
    print('gaff', gaff_molecule.coords.shape, gaff_molecule.forces.shape)
    print(gaff_molecule)


    print(datetime.now() - startTime)
    sys.stdout.flush()

    pairs = []
    for i in range(len(model_molecule.atoms)):
        for j in range(i):
            pairs.append([i, j])


    dataset_interatomic_measures = Binner()
    dataset_interatomic_measures.get_bond_pop(dataset_molecule.coords[::10], 
            pairs)

    model_interatomic_measures = Binner()
    model_interatomic_measures.get_bond_pop(model_molecule.coords, pairs)

    gaff_interatomic_measures = Binner()
    gaff_interatomic_measures.get_bond_pop(gaff_molecule.coords, pairs)


    info = [[
                dataset_interatomic_measures.rs.T.flatten(),
                model_interatomic_measures.rs.T.flatten(), 
                gaff_interatomic_measures.rs.T.flatten(), 
                1000, 
                '$r / \AA$', 'all']
            ]

    for i in info:
        bin_edges, hist = Binner.get_hist(i[0], i[3])
        bin_edges2, hist2 = Binner.get_hist(i[1], i[3])
        bin_edges3, hist3 = Binner.get_hist(i[2], i[3])
        Plotter.xy_scatter(
                [bin_edges2,bin_edges, bin_edges3], 
                [hist2, hist, hist3], 
                ['ML', 'DFT', 'MM'], ['k', 'r', 'b'], i[4], 
                'P($r$)', [10, 10, 10],
                'hist-model-dataset-gaff-r-{}.png'.format(i[5]))




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


