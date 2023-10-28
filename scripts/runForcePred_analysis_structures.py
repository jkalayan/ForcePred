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
#from ForcePred.nn.Network_perminv import Network
from ForcePred.nn.Network_double_decomp import Network

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
    molecule = Molecule()
    NPParser(atom_file, [coord_files[0]], [force_files[0]], [energy_files[0]], 
            molecule)
    #'''
    pred_molecule = Molecule()
    NPParser(atom_file, [coord_files[0]], [force_files[1]], [energy_files[1]], 
            pred_molecule)
    print('dataset', molecule.coords.shape, 
            molecule.forces.shape, molecule.energies.shape)
    print(molecule)
    #'''


    n_atoms = len(molecule.atoms)
    _NC2 = int(n_atoms * (n_atoms-1)/2)


    molecule.mat_FE = np.reshape(np.loadtxt('data/q.dat'), (-1,_NC2))
    pred_molecule.mat_FE = np.reshape(np.loadtxt('data/pred_q.dat'), (-1,_NC2))


    ##truncate dataset
    #molecule.coords = molecule.coords[::100]
    #molecule.forces = molecule.forces[::100]
    #molecule.energies = molecule.energies[::100]

    molecule.orig_energies = np.copy(molecule.energies)


    split = 2
    train = round(len(molecule.coords) / split, 3)
    print('\nget train and test sets, '\
            'training set is {} points'.format(train))
    print('!!!use regularly spaced training')
    molecule.train = np.arange(2, len(molecule.coords), split).tolist() 
    molecule.test = [x for x in range(0, len(molecule.coords)) 
            if x not in molecule.train]
    print(len(molecule.train))
    print(len(molecule.test))

    #'''
    prescale = [0, 1, 0, 1]
    train = True
    if train:
        ###!!!!!!!!
        #print('!!!TRAIN OVER-RIDDEN FOR SCALING')
        #train_forces = molecule.forces
        #train_energies = molecule.energies

        train_forces = np.take(molecule.forces, molecule.train, axis=0)
        train_energies = np.take(molecule.energies, molecule.train, axis=0)

        forces_min = np.min(molecule.forces)
        forces_max = np.max(molecule.forces)
        forces_diff = forces_max - forces_min
        #print(forces_diff)
        forces_rms = np.sqrt(np.mean(molecule.forces.flatten()**2))
        energies_rms = np.sqrt(np.mean(molecule.energies.flatten()**2))
        forces_mean = np.mean(molecule.forces.flatten())
        energies_mean = np.mean(molecule.energies.flatten())
        energies_max = np.max(np.absolute(molecule.energies))
        #molecule.energies = (molecule.energies - energies_mean) / energies_rms
        #molecule.forces = (molecule.forces * 0) # forces_rms)


        print('E ORIG min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('train E ORIG min: {} max: {}'.format(np.min(train_energies), 
                np.max(train_energies)))
        print('F ORIG min: {} max: {}'.format(np.min(molecule.forces), 
                np.max(molecule.forces)))
        print('train F ORIG min: {} max: {}'.format(np.min(train_forces), 
                np.max(train_forces)))
        prescale_energies = True
        #prescale = [0, 1, 0, energies_mean, forces_rms, energies_max, 
                #forces_max]
        prescale = [0, 1, 0, 1]
        if prescale_energies:
            print('\nprescale energies so that magnitude is '\
                    'comparable to forces')
            min_e = np.min(train_energies)
            max_e = np.max(train_energies)
            min_f = np.min(train_forces)
            max_f = np.max(train_forces)

            molecule.energies = ((max_f - min_f) * 
                    (molecule.energies - min_e) / 
                    (max_e - min_e) + min_f)

            prescale[0] = min_e
            prescale[1] = max_e
            prescale[2] = min_f
            prescale[3] = max_f

            print('E SCALED min: {} max: {}'.format(
                    np.min(molecule.energies), np.max(molecule.energies)))


        print('prescale value:', prescale)
        sys.stdout.flush()
    #'''

    '''
    bias_type = '1/r' #'r'#
    print('bias_type:', bias_type)
    Converter.get_simultaneous_interatomic_energies_forces(molecule, 
            bias_type)

    np.savetxt('q.dat', (molecule.mat_FE[:]).reshape(-1,_NC2))
    np.savetxt('C.dat', (molecule.coords[:]).reshape(-1,3))
    np.savetxt('F.dat', (molecule.forces[:]).reshape(-1,3))
    np.savetxt('E.dat', (molecule.orig_energies[:]).reshape(-1,1))


    print(datetime.now() - startTime)
    sys.stdout.flush()

    network = Network(molecule)
    model = Network.get_coord_FE_model(network, molecule, prescale, 
            training_model=False, model_file='../model80/best_ever_model_0')

    print('Checking Model Predictions')
    prediction = model.predict(molecule.coords[0].reshape(1,-1,3))

    print(prediction)
    forces = prediction[0]#[2]
    energies = prediction[2]

    print('C')
    print(molecule.coords[0])
    print('F')
    print(molecule.forces[0])
    print(forces)
    print('E')
    print(molecule.energies[0])
    print(energies)

    print(datetime.now() - startTime)
    sys.stdout.flush()

    prediction = model.predict(molecule.coords)

    np.savetxt('pred_q.dat', (prediction[1]).reshape(-1,_NC2))
    np.savetxt('pred_F.dat', (prediction[0]).reshape(-1,3))
    np.savetxt('pred_E.dat', (prediction[2]).reshape(-1,1))

    print(datetime.now() - startTime)
    sys.stdout.flush()

    sys.exit()
    '''

    print(datetime.now() - startTime)
    sys.stdout.flush()


    atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
            zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
    pairs = []
    chosen_inx = []
    _N = 0
    for i in range(len(molecule.atoms)):
        for j in range(i):
            pairs.append([i, j])
            print(_N+1, atom_names[i], atom_names[j])
            _N += 1
    interatomic_measures = Binner()
    interatomic_measures.get_bond_pop(molecule.coords, pairs)
    print(interatomic_measures.rs.shape, molecule.mat_FE.shape)
    dists = interatomic_measures.rs.flatten()

    if len(molecule.mat_FE[0]) == _NC2+1:
        dists = np.ones((len(molecule.coords), _NC2+1))
        dists[:,-1] = interatomic_measures.rs
        dists = dists.flatten()

    pred_molecule.energies = ((max_f - min_f) * 
            (pred_molecule.energies - min_e) / 
            (max_e - min_e) + min_f)

    s = np.arange(len(molecule.coords))
    s_tile_q = np.tile(s, (_NC2,1)).T.flatten()
    E = molecule.energies.flatten()
    E_mae = (pred_molecule.energies.flatten() - E) ** 2
    E_large_mae_idx = np.where(E_mae > 200)
    print(E_large_mae_idx, len(E_large_mae_idx[0]))
    print(np.take(molecule.mat_FE, E_large_mae_idx[0], axis=0))
    q = molecule.mat_FE.flatten()
    q_mean = np.mean(np.abs(molecule.mat_FE), dtype=np.float32, axis=1).flatten()
    q_max = np.amax(np.abs(molecule.mat_FE), axis=1).flatten()
    q_range = (np.amax(molecule.mat_FE, axis=1) - np.amin(molecule.mat_FE, 
            axis=1)).flatten()
    F_max = np.amax(np.abs(molecule.forces).reshape(
            -1,3*n_atoms), axis=1).flatten()
    F_mae = (np.sum((pred_molecule.forces.reshape(-1,3*n_atoms) - 
            molecule.forces.reshape(-1,3*n_atoms)) ** 2, 
            axis=1) / (3*n_atoms)).flatten()
    F_large_mae = np.take(F_mae, np.where(F_mae > 1))
    F_low_mae = np.take(F_mae, np.where(F_mae <= 1))
    print(F_large_mae.shape, F_low_mae.shape)

    print('abs max_q: {} abs min_q: {}'.format(
        max(abs(q)), min(abs(q))))

    Plotter.xy_scatter([E], [q_mean], [''], ['k'],
            'scaled E / kcal/mol/$\AA$', 'q (mean) / kcal/mol/$\AA$',  [10], 
            'scatter-q_mean-E.png')
    Plotter.xy_scatter([E_mae], [q_mean], [''], ['k'],
            'scaled E (MAE)/ kcal/mol/$\AA$', 
            'q (mean) / kcal/mol/$\AA$',  [10], 
            'scatter-q_mean-E_mae.png')
    Plotter.xy_scatter([E_mae], [q_max], [''], ['k'],
            'scaled E (MAE)/ kcal/mol/$\AA$', 
            'q (max) / kcal/mol/$\AA$',  [10], 
            'scatter-q_max-E_mae.png')
    Plotter.xy_scatter([E], [q_max], [''], ['k'],
            'scaled E / kcal/mol/$\AA$', 'q (max) / kcal/mol/$\AA$',  [10], 
            'scatter-q_max-E.png')
    Plotter.xy_scatter([E], [q_range], [''], ['k'],
            'scaled E / kcal/mol/$\AA$', 'q (range) / kcal/mol/$\AA$',  [10], 
            'scatter-q_range-E.png')
    Plotter.xy_scatter([F_max], [q_mean], [''], ['k'],
            'F (max) / kcal/mol/$\AA$', 'q (mean) / kcal/mol/$\AA$',  [10], 
            'scatter-q_mean-F_max.png')
    Plotter.xy_scatter([F_mae], [q_mean], [''], ['k'],
            'F (MAE) / kcal/mol/$\AA$', 'q (mean) / kcal/mol/$\AA$',  [10], 
            'scatter-q_mean-F_mae.png', log=True)

    Plotter.xy_scatter([F_mae], [F_max], [''], ['k'],
            'F (MAE) / kcal/mol/$\AA$', 'F (max) / kcal/mol/$\AA$',  [10], 
            'scatter-F_max-F_mae.png', log=True)
    Plotter.xy_scatter([s], [q_max], [''], ['k'],
            'Structure #', 'q (max) / kcal/mol/$\AA$',  [10], 
            'scatter-s-q_max.png')


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


