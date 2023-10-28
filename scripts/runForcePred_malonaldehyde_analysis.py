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

    print('Load data')
    train_molecule = Molecule()
    NPParser(atom_file, [coord_files[0]], [force_files[0]], [], 
            train_molecule)
    test_molecule = Molecule()
    NPParser(atom_file, [coord_files[1]], [force_files[1]], [], test_molecule)

    mlff_molecule = Molecule()
    mlff2_molecule = Molecule()
    XYZParser(atom_file, [coord_files[2]], [], [], mlff2_molecule)
    NPParser(atom_file, [], [force_files[2]], [energy_files[0]], 
            mlff_molecule)
    mlff_molecule.coords = mlff2_molecule.coords
    #mlff_molecule.forces = mlff_molecule.forces
    mlff_molecule.energies = mlff_molecule.energies
    print(mlff_molecule.coords.shape, #mlff_molecule.forces.shape, 
            mlff_molecule.energies.shape)

    gaff_molecule = Molecule()
    gaff2_molecule = Molecule()
    XYZParser(atom_file, [coord_files[3]], [], [], gaff2_molecule)
    NPParser(atom_file, [], [force_files[3]], [energy_files[1]], 
            gaff_molecule)
    gaff_molecule.coords = gaff2_molecule.coords

    print(datetime.now() - startTime)
    sys.stdout.flush()


    time = np.array(list(range(0, 20000))) * 0.0005 #10ps

    '''
    time = np.array(list(range(0, 2609))) * 0.005 #13ns

    print(len(time), len(mlff_molecule.energies.flatten()))

    time2 = []
    dE = []
    for t, e in zip(time, mlff_molecule.energies.flatten()):
        if e > -3 and e < 100:
            time2.append(t)
            dE.append(e)
        else:
            continue



    info = [[time2, dE, 'MLFF', r'$\Delta E$ / kcal/mol', 10, 'mlff-dE-2'],
            ]

    for i in info:
        Plotter.xy_scatter([i[0]], [i[1]], [i[2]], ['k'], 'time / ns', 
                i[3], i[4], 'scatter-{}.png'.format(i[5]))
    '''


    #A = Molecule.find_bonded_atoms(mlff_molecule.atoms, 
            #mlff_molecule.coords[0])
    #print(A)

    #CCCOOHHHH
    bonded_list = [[0,1], [0,4], [0,5], 
            [1,2], [1,6], [1,7], 
            [2,3], [2,8]]
    angle_list = [[1,0,4], [1,0,5], [4,0,5], 
            [0,1,2], [0,1,7], [0,1,6], [2,1,6], [2,1,7],
            [1,2,3], [1,2,8], [3,2,8]]
    dihedral_list = [[0,1,2,3], [0,1,2,8],
            [2,1,0,4], [2,1,0,5], 
            [3,2,1,6],[3,2,1,7],
            [4,0,1,6], [4,0,1,7],
            [5,0,1,6], [5,0,1,7],
            [6,1,2,8], [7,1,2,8]]

    dihedrals = [[0,1,2,3], [2,1,0,4]]
    angles = [[0,1,2]]
    bonds = [[0,1]]

    mlff_measures = Binner()
    mlff_measures.get_dih_pop(mlff_molecule.coords, dihedrals)
    mlff_measures.get_angle_pop(mlff_molecule.coords, angles)
    mlff_measures.get_bond_pop(mlff_molecule.coords, bonds)

    gaff_measures = Binner()
    gaff_measures.get_dih_pop(gaff_molecule.coords, dihedrals)
    gaff_measures.get_angle_pop(gaff_molecule.coords, angles)
    gaff_measures.get_bond_pop(gaff_molecule.coords, bonds)


    Plotter.xy_scatter([mlff_measures.phis.T[0]], [mlff_measures.phis.T[1]], 
            ['mlff'], ['k'], 'CCCO3 dih', 'CCCO4 dih', 2, 'dihs-mlff.png')
    Plotter.xy_scatter([gaff_measures.phis.T[0]], [gaff_measures.phis.T[1]], 
            ['gaff'], ['k'], 'CCCO3 dih', 'CCCO4 dih', 2, 'dihs-gaff.png')

    Plotter.hist_2d(mlff_measures.phis.T[0], mlff_measures.phis.T[1], 
            'CCCO3', 'CCCO4', 'hist2d-mlff-dihs.png')
    Plotter.hist_2d(gaff_measures.phis.T[0], gaff_measures.phis.T[1], 
            'CCCO3', 'CCCO4', 'hist2d-gaff-dihs.png')

    print(datetime.now() - startTime)
    sys.stdout.flush()

    pairs = []
    for i in range(len(mlff_molecule.atoms)):
        for j in range(i):
            pairs.append([i, j])

    mlff_interatomic_measures = Binner()
    mlff_interatomic_measures.get_bond_pop(mlff_molecule.coords, pairs)

    gaff_interatomic_measures = Binner()
    gaff_interatomic_measures.get_bond_pop(gaff_molecule.coords, pairs)

    train_interatomic_measures = Binner()
    train_interatomic_measures.get_bond_pop(train_molecule.coords, pairs)

    info = [[mlff_molecule.energies, gaff_molecule.energies, 100, 'dE'],
            [mlff_measures.phis.T[0], gaff_measures.phis.T[0], 180, 'CCCO3'],
            [mlff_measures.phis.T[1], gaff_measures.phis.T[1], 180, 'CCCO4'],
            [mlff_measures.thetas.T[0], gaff_measures.thetas.T[0], 180, 
                'CCC'],
            [mlff_measures.rs.T[0], gaff_measures.rs.T[0], 100, 'CC'],
            [mlff_interatomic_measures.rs.T.flatten(), 
                gaff_interatomic_measures.rs.T.flatten(), 1000, 
                'r']
            ]

    for i in info:
        bin_edges, hist = Binner.get_hist(i[0], i[2])
        bin_edges2, hist2 = Binner.get_hist(i[1], i[2])
        Plotter.xy_scatter([bin_edges, bin_edges2], [hist, hist2], 
                ['MLFF', 'GAFF'], ['k', 'b'], i[3], 'probability', 40,
                'hist-mlff-gaff-{}.png'.format(i[3]))

    print(datetime.now() - startTime)
    sys.stdout.flush()


    info = [[time, mlff_molecule.energies, 'MLFF', r'$\Delta E$ / kcal/mol', 
                10, 'mlff-dE'],
            [time, gaff_molecule.energies, 'GAFF', r'$\Delta E$ / kcal/mol', 
                10, 'gaff-dE'],
            [time, mlff_measures.rs.T[0], 'MLFF', r'CC $r / \AA$', 10, 'CC'],
            [time, mlff_measures.phis.T[0], 'MLFF', r'CCCO3 $\Phi$', 10, 
                'CCCO3'],
            ]

    for i in info:
        Plotter.xy_scatter([i[0]], [i[1]], [i[2]], ['k'], 'time / ns', 
                i[3], i[4], 'scatter-{}.png'.format(i[5]))


    print(datetime.now() - startTime)
    sys.stdout.flush()


    train_measures = Binner()
    train_measures.get_dih_pop(train_molecule.coords, dihedrals)
    train_measures.get_bond_pop(train_molecule.coords, pairs)

    test_measures = Binner()
    test_measures.get_dih_pop(test_molecule.coords, dihedrals)
    test_measures.get_bond_pop(test_molecule.coords, pairs)

    info = [
            [train_molecule.forces.flatten(), test_molecule.forces.flatten(),
                100, 'F'],
            [train_measures.phis.T[0], test_measures.phis.T[0], 180, 'CCCO3'],
            [train_measures.phis.T[1], train_measures.phis.T[1], 180, 
                'CCCO4'],
            [train_measures.rs.T.flatten(), test_measures.rs.T.flatten(), 200, 
                'interatomic-rs'],
            [train_measures.rs.T.flatten(), 
                mlff_interatomic_measures.rs.T.flatten(),  200, 
                'interatomic-rs-train-mlff']
            ]

    for i in info:
        bin_edges, hist = Binner.get_hist(i[0], i[2])
        bin_edges2, hist2 = Binner.get_hist(i[1], i[2])
        Plotter.xy_scatter([bin_edges, bin_edges2], [hist, hist2], 
                ['train', 'test'], ['k', 'b'], i[3], 'probability', 40,
                'hist-train-test-{}.png'.format(i[3]))

    print(datetime.now() - startTime)
    sys.stdout.flush()

    info = [[mlff_interatomic_measures.rs.T.flatten(), 
                train_interatomic_measures.rs.T.flatten(), 1000, 
                'r']
            ]

    for i in info:
        bin_edges, hist = Binner.get_hist(i[0], i[2])
        bin_edges2, hist2 = Binner.get_hist(i[1], i[2])
        Plotter.xy_scatter([bin_edges, bin_edges2], [hist, hist2], 
                ['MLFF', 'MD17'], ['k', 'r'], i[3], 'probability', 10,
                'hist-mlff-md17-train-{}.png'.format(i[3]))








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


