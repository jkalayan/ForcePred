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
    mlff_molecule.forces = mlff_molecule.forces
    mlff_molecule.energies = mlff_molecule.energies
    print(mlff_molecule.coords.shape, mlff_molecule.forces.shape, 
            mlff_molecule.energies.shape)

    gaff_molecule = Molecule()
    gaff2_molecule = Molecule()
    XYZParser(atom_file, [coord_files[3]], [], [], gaff2_molecule)
    NPParser(atom_file, [], [force_files[3]], [energy_files[1]], 
            gaff_molecule)
    gaff_molecule.coords = gaff2_molecule.coords

    print(datetime.now() - startTime)
    sys.stdout.flush()


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
            

    #bin_edges, hist = Binner.get_hist(mlff_molecule.energies, 50)
    #Plotter.plot_2d([bin_edges], [hist], ['mlff'], 
            #'$\Delta E$/ kcal/mol', 'probability', 'hist-energy-mlff.png')
    #Plotter.xy_scatter([bin_edges], [hist], ['mlff'], ['k'], 
            #'$\Delta E$/ kcal/mol', 'probability', 2, 'hist-energy-mlff.png')
    #Plotter.hist_1d([mlff_molecule.energies], '$\Delta E$/ kcal/mol', 
            #'probability', 'hist-energy-mlff.png')
    #Plotter.hist_1d([gaff_molecule.energies], '$\Delta E$/ kcal/mol', 
            #'probability', 'hist-energy-gaff.png')

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

    info = [[mlff_molecule.energies, gaff_molecule.energies, 'dE'],
            [mlff_measures.phis.T[0], gaff_measures.phis.T[0], 'CCCO3'],
            [mlff_measures.phis.T[1], gaff_measures.phis.T[1], 'CCCO4'],
            [mlff_measures.thetas.T[0], gaff_measures.thetas.T[0], 'CCC'],
            [mlff_measures.rs.T[0], gaff_measures.rs.T[0], 'CC']
            ]

    for i in info:
        bin_edges, hist = Binner.get_hist(i[0], 60)
        bin_edges2, hist2 = Binner.get_hist(i[1], 60)
        Plotter.xy_scatter([bin_edges, bin_edges2], [hist, hist2], 
                ['mlff', 'gaff'], ['k', 'b'], i[2], 'probability', 40,
                'hist-mlff-gaff-{}.png'.format(i[2]))

    '''
    #Plotter.plot_bar(bin_edges, hist, 'CCCO3 dihedral', 
            #'probability', 'hist-dih-CCCO3-mlff.png')
    bin_edges, hist = Binner.get_hist(mlff_measures.phis.T[1], 60)
    bin_edges2, hist2 = Binner.get_hist(gaff_measures.phis.T[1], 60)
    Plotter.xy_scatter([bin_edges, bin_edges2], [hist, hist2], 
            ['mlff', 'gaff'], ['k', 'b'], 'CCCO4', 'probability', 40,
            'hist-dihCCCO4-mlff-gaff.png')
    #Plotter.plot_bar(bin_edges, hist, 'CCCO4 dihedral', 
            #'probability', 'hist-dih-CCCO4-mlff.png')
    '''



    ''' #slow!
    Plotter.hist_1d(mlff_measures.phis.T[0], 'CCCO3 dihedral', 
            'probability', 'hist-dih-CCCO3-mlff.png')
    Plotter.hist_1d(mlff_measures.phis.T[1], 'CCCO4 dihedral', 
            'probability', 'hist-dih-CCCO4-mlff.png')
    '''

    '''
    Plotter.xy_scatter([mlff_measures.phis.T[0]], [gaff_measures.phis.T[0]], 
            ['CCCO3'], ['k'], 'mlff', 'gaff', 2, 'dihCCCO3-mlff-gaff.png')
    Plotter.xy_scatter([mlff_measures.phis.T[1]], [gaff_measures.phis.T[1]], 
            ['CCCO4'], ['k'], 'mlff', 'gaff', 2, 'dihCCCO4-mlff-gaff.png')
    '''

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


