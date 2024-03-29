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
    dataset_molecule = Molecule()
    NPParser(atom_file, [coord_files[0]], [force_files[0]], [energy_files[0]], 
            dataset_molecule)
    print('dataset', dataset_molecule.coords.shape, 
            dataset_molecule.forces.shape, dataset_molecule.energies.shape)

    model_molecule = Molecule()
    NPParser(atom_file, [coord_files[1]], [force_files[1]], [energy_files[1]], 
            model_molecule)
    print('model', model_molecule.coords.shape, model_molecule.forces.shape, 
            model_molecule.energies.shape)

    #'''
    trucate_model = 1576 # 7 1644 #5 2667 #3 2757 #2 3138 #4 
    model_molecule.coords = model_molecule.coords[:trucate_model]
    model_molecule.forces = model_molecule.forces[:trucate_model]
    model_molecule.energies = model_molecule.energies[:trucate_model]
    #'''

    print(datetime.now() - startTime)
    sys.stdout.flush()

    pairs = []
    for i in range(len(model_molecule.atoms)):
        for j in range(i):
            pairs.append([i, j])
    model_interatomic_measures = Binner()
    model_interatomic_measures.get_bond_pop(model_molecule.coords, pairs)
    ###get pairwise forces
    mm_molecule = Molecule()
    NPParser(atom_file, [coord_files[1]], [force_files[2]], [], mm_molecule)
    mm_molecule.coords = mm_molecule.coords[:trucate_model]
    mm_molecule.forces = mm_molecule.forces[:trucate_model]
    Converter(model_molecule)
    Converter(mm_molecule)
    print(model_molecule.mat_F.shape)
    print(mm_molecule.mat_F.shape)
    n_model = range(0, len(model_molecule.coords))
    model_OO = model_molecule.mat_F[:,9]
    mm_OO = mm_molecule.mat_F[:,9]
    Plotter.xy_scatter(
            #[n_model, n_model], 
            [model_interatomic_measures.rs.T[9].flatten(), 
                model_interatomic_measures.rs.T[9].flatten()],
            [model_OO, mm_OO],
            ['model_OO', 'mm_OO'],
            ['k', 'r'], 'rOO', 'qOO', [1, 1],
            'scatter-OO.png')

    sys.exit()

    '''
    #time = np.array(list(range(0, 20000))) * 0.0005 #10ps
    time = np.array(list(range(0, 60000000, 10000))) * 0.0000005 + 2 #30ns
    dE = model_molecule.energies.flatten()[480:]
    print(len(time), len(dE))
    info = [[time, dE, 'MLFF', r'$\Delta E$ / kcal/mol', 1, 'dE'],
            ]
    for i in info:
        Plotter.xy_scatter([i[0]], [i[1]], [i[2]], ['k'], 'time / ns', 
                i[3], [i[4]], 'scatter-{}.png'.format(i[5]))
    '''

    #ismaeel's malonaldehye
    #dihedrals = [[4,0,1,2], [0,1,2,3]] #both OCCC dihs
    #revised MD17 model
    dihedrals = [[4,0,1,2], [0,1,2,3]] #both OCCC dihs
    model_measures = Binner()
    model_measures.get_dih_pop(model_molecule.coords, dihedrals)
    dataset_measures = Binner()
    dataset_measures.get_dih_pop(dataset_molecule.coords[::10], dihedrals)

    n_dataset = range(0, len(dataset_molecule.coords[::10]))
    n_model = range(0, len(model_molecule.coords))
    Plotter.xy_scatter([n_dataset, n_model, n_dataset, n_model], 
            [dataset_measures.phis.T[0], model_measures.phis.T[0], 
            dataset_measures.phis.T[1], model_measures.phis.T[1]],
            ['dataset $\zeta_1$', 'model $\zeta_1$', 'dataset $\zeta_2$', 
            'model $\zeta_2$'],
            ['k', 'grey', 'r', 'orange'], 'step', '$\zeta_x$ (deg)', [1]*4,
            'scatter-step-dihs.png')


    Plotter.xy_scatter([dataset_measures.phis.T[0], 
            model_measures.phis.T[0]], 
            [dataset_measures.phis.T[1], model_measures.phis.T[1]], 
            ['MD17', 'MLFF'], ['k', 'r'], '$\zeta_1$ (deg)', 
            '$\zeta_2$ (deg)', [1, 5],
            'dihs-dataset-model.png')

    Plotter.xy_scatter([dataset_measures.phis.T[0]], 
            [dataset_measures.phis.T[1]], 
            ['MD17'], ['k'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', [1], 
            'dihs-dataset.png')

    Plotter.xy_scatter([model_measures.phis.T[0]], 
            [model_measures.phis.T[1]], 
            ['MLFF'], ['r'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', [5], 
            'dihs-model.png')

    Plotter.hist_2d([dataset_measures.phis.T[0], model_measures.phis.T[0]], 
            [dataset_measures.phis.T[1], model_measures.phis.T[1]], 
            ['Greys', 'Reds'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', 
            'hist2d-dihs-dataset-model.png')

    Plotter.hist_2d([dataset_measures.phis.T[0]], 
            [dataset_measures.phis.T[1]], 
            ['Greys'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', 
            'hist2d-dihs-dataset.png')

    Plotter.hist_2d([model_measures.phis.T[0]], 
            [model_measures.phis.T[1]], 
            ['Reds'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', 
            'hist2d-dihs-model.png')






    '''

    print('CCCCCCCCCOOOOHHHHHHHH')
    A = Molecule.find_bonded_atoms(model_molecule.atoms, 
            model_molecule.coords[0])
    print(A)

    print(datetime.now() - startTime)
    sys.stdout.flush()

    #CCCCCCCCCOOOOHHHHHHHH


    dihedrals = [[4,8,12,6], [6,5,7,9], [12,6,5,7]] #CCOC, CCCO, OCCC
    angles = [[5,6,12], [6,5,7]] #CCO, CCC
    bonds = [[12,6]] #CC

    model_measures = Binner()
    model_measures.get_dih_pop(model_molecule.coords, dihedrals)
    model_measures.get_angle_pop(model_molecule.coords, angles)
    model_measures.get_bond_pop(model_molecule.coords, bonds)

    gaff_measures = Binner()
    gaff_measures.get_dih_pop(gaff_molecule.coords, dihedrals)
    gaff_measures.get_angle_pop(gaff_molecule.coords, angles)
    gaff_measures.get_bond_pop(gaff_molecule.coords, bonds)

    Plotter.xy_scatter([model_measures.phis.T[0]], [model_measures.phis.T[1]], 
            ['model'], ['k'], 'CCOC dih', 'CCCO dih', 2, 'dihs-model.png')
    Plotter.xy_scatter([gaff_measures.phis.T[0]], [gaff_measures.phis.T[1]], 
            ['gaff'], ['k'], 'CCOC dih', 'CCCO dih', 2, 'dihs-gaff.png')
    '''

    '''
    Plotter.hist_2d(model_measures.phis.T[0], model_measures.phis.T[1], 
            'CCOC', 'CCCO', 'hist2d-model-dihs.png')
    Plotter.hist_2d(gaff_measures.phis.T[0], gaff_measures.phis.T[1], 
            'CCOC', 'CCCO', 'hist2d-gaff-dihs.png')
    '''


    print(datetime.now() - startTime)
    sys.stdout.flush()

    pairs = []
    for i in range(len(model_molecule.atoms)):
        for j in range(i):
            pairs.append([i, j])

    model_interatomic_measures = Binner()
    model_interatomic_measures.get_bond_pop(model_molecule.coords, pairs)


    dataset_interatomic_measures = Binner()
    dataset_interatomic_measures.get_bond_pop(dataset_molecule.coords[::10], 
            pairs)



    info = [[model_interatomic_measures.rs.T.flatten(), 
                dataset_interatomic_measures.rs.T.flatten(), 1000, 
                '$r_{ij}/ \AA$', 'all'],
            [model_interatomic_measures.rs.T[9].flatten(), 
                dataset_interatomic_measures.rs.T[9].flatten(), 200, 
                '$r_{ij}/ \AA$', '4-3']
            ]

    for i in info:
        bin_edges, hist = Binner.get_hist(i[0], i[2])
        bin_edges2, hist2 = Binner.get_hist(i[1], i[2])
        Plotter.xy_scatter([bin_edges2, bin_edges], [hist2, hist], 
                ['MD17', 'MLFF'], ['k', 'r'], i[3], 'P($r_{ij}$)', [10, 10],
                'hist-model-dataset-r-{}.png'.format(i[4]))


    Plotter.xy_scatter([n_model], 
            [model_interatomic_measures.rs.T[9].flatten()],
            ['model $r_{OO}$'],
            ['k'], 'step', '$r_{OO}$', [1],
            'scatter-step-rOO.png')




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


