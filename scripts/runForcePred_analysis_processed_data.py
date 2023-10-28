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
        Permuter, XYZParser, Binner, Writer, Plotter, Conservation
        #Network
from ForcePred.nn.Network_shared_ws import Network
from keras.models import Model, load_model    
from keras import backend as K                                              
import sys
import tensorflow as tf
import os
import math

NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


tf.config.threading.set_inter_op_parallelism_threads(NUMCORES) 
tf.config.threading.set_intra_op_parallelism_threads(NUMCORES)
tf.config.set_soft_device_placement(1)


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files', 
        n_nodes='n_nodes', n_layers='n_layers', n_training='n_training', 
        n_val='n_val', n_test='n_test', grad_loss_w='grad_loss_w', 
        qFE_loss_w='qFE_loss_w', E_loss_w='E_loss_w', bias='bias',
        filtered='filtered', load_model='load_model'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class
    prescale = [0, 1, 0, 1, 0, 0]
    #read in input data files
    if list_files:
        input_files = open(list_files).read().split()
    mol=False
    if atom_file and mol:
        #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
        NPParser(atom_file, coord_files, force_files, energy_files, molecule)
        print(datetime.now() - startTime)
        sys.stdout.flush()

        print(molecule)
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        sys.stdout.flush()

        # write pdb file for first frame
        Writer.write_pdb(molecule.coords[0], 'MOL', 1, molecule.atoms, 
                'molecule.pdb', 'w')

        #n_training = 10000
        #n_val = 50
        if n_test == -1 or n_test > len(molecule.coords):
            n_test = len(molecule.coords) - n_training
        print('\nn_nodes', n_nodes, '\nn_layers:', n_layers,
                '\nn_training', n_training, '\nn_val', n_val,
                '\nn_test', n_test, '\ngrad_loss_w', grad_loss_w,
                '\nqFE_loss_w', qFE_loss_w, '\nE_loss_w', E_loss_w,
                '\nbias', bias)


        ### Scale energies using min/max forces in training set

        def prescale_energies(molecule, train_idx):
            train_forces = np.take(molecule.forces, train_idx, axis=0)
            train_energies = np.take(molecule.orig_energies, train_idx, axis=0)
            min_e = np.min(train_energies)
            max_e = np.max(train_energies)
            #min_f = np.min(train_forces)
            min_f = np.min(np.abs(train_forces))
            #max_f = np.max(train_forces)
            max_f = np.max(np.abs(train_forces))
            molecule.energies = ((max_f - min_f) * 
                    (molecule.orig_energies - min_e) / 
                    (max_e - min_e) + min_f)
            prescale[0] = min_e
            prescale[1] = max_e
            prescale[2] = min_f
            prescale[3] = max_f
            return prescale

        # save original energies first, before scaling by forces
        molecule.orig_energies = np.copy(molecule.energies)

        #bias = '1/r' #default for now
        extra_cols = 0 #n_atoms
        pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                pairs.append([i, j])
        if filtered:
            print('\nprescale energies so that magnitude is comparable to forces')

            prescale = prescale_energies(molecule, list(range(len(molecule.coords))))
            print('E SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                    np.max(molecule.energies)))
            print('prescale value:', prescale)
            sys.stdout.flush()

            print('\nget decomposed forces and energies simultaneously '\
                    'with energy bias: {}'.format(bias))

            Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                    bias_type=bias, extra_cols=extra_cols)



            '''
            atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                    zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
            atom_pairs = []
            for i in range(len(molecule.atoms)):
                for j in range(i):
                    atom_pairs.append('{}_{}'.format(atom_names[i], 
                        atom_names[j]))
            qq_root = []
            for i in range(n_atoms):
                for j in range(i):
                    v = (molecule.atoms[i] * molecule.atoms[j]) ** 0.5
                    qq_root.append(v)
            print('\npair ave_r pair_freq q_F')
            ave_rs_all = np.average(molecule.mat_r, axis=0)
            qq_root = np.array(qq_root)
            q_F = (np.sum(molecule.mat_F, axis=0) ** 2) / len(molecule.coords)
            pair_freq = q_F / qq_root
            for ij in range(_NC2):
                print('{} {:.3f} {:.3f} {:.3f}'.format(
                        atom_pairs[ij], ave_rs_all[ij], pair_freq[ij], q_F[ij]))
            Plotter.xy_scatter([ave_rs_all], [pair_freq], [''], ['k'], 
                    '$<r_{ij}> / \AA$', 'pair_freq', [10], 
                    'scatter-ave_r-pair_freq.png')
            Plotter.xy_scatter([ave_rs_all], [q_F], [''], ['k'], 
                    '$<r_{ij}> / \AA$', 'q / kcal/mol/$\AA$', [10], 
                    'scatter-ave_r-q_F.png')
            sys.exit()
            '''

            print('\nPlot decompFE vs r and histogram of decompFE')
            Plotter.hist_1d([molecule.mat_FE], 'q / kcal/mol', 'P(q)', 
                    'hist_q.png')
            interatomic_measures = Binner()
            interatomic_measures.get_bond_pop(molecule.coords, pairs)
            dists = interatomic_measures.rs.flatten()
            decompFE = molecule.mat_FE[:,:_NC2,].flatten()
            Plotter.xy_scatter([dists], [decompFE], [''], ['k'], '$r_{ij} / \AA$',
                    'q / kcal/mol', [10], 'scatter-r-decompFE.png')
            sys.stdout.flush()

            print('\nRemove high magnitude decompFE structures')
            print('n_structures', len(molecule.coords))
            q_max = np.amax(np.abs(molecule.mat_FE), axis=1).flatten()
            mean_q_max = np.mean(q_max)
            print('q_max', np.amax(q_max), 'mean_q_max', mean_q_max)
            cap_q_max = np.percentile(q_max, 98)
            q_close_idx = np.where(q_max <= cap_q_max)
            print('n_filtered_structures', len(q_close_idx[0]))
            molecule.coords = np.take(molecule.coords, q_close_idx, axis=0)[0]
            molecule.forces = np.take(molecule.forces, q_close_idx, axis=0)[0]
            molecule.energies = np.take(molecule.energies, q_close_idx, 
                    axis=0)[0]
            molecule.orig_energies = np.take(molecule.orig_energies, 
                    q_close_idx, axis=0)[0]
            sys.stdout.flush()



        print('\nDefine training and test sets')
        train_split = math.ceil(len(molecule.coords) / n_training)
        print('train_split', train_split)
        molecule.train = np.arange(0, len(molecule.coords), train_split).tolist() 
        val_split = math.ceil(len(molecule.train) / n_val)
        print('val_split', val_split)
        molecule.val = molecule.train[::val_split]
        molecule.train2 = [x for x in molecule.train if x not in molecule.val]
        molecule.test = [x for x in range(0, len(molecule.coords)) 
                if x not in molecule.train]
        if n_test > len(molecule.test):
            n_test = len(molecule.test)
        if n_test == 0:
            test_split = 0
            molecule.test = [0]
        if n_test !=0:
            test_split = math.ceil(len(molecule.test) / n_test)
            molecule.test = molecule.test[::test_split]
        print('test_split', test_split)
        print('Check! \nN training2 {} \nN val {} \nN test {}'.format(
            len(molecule.train2), len(molecule.val), len(molecule.test)))
        sys.stdout.flush()

        
        print('\nScale orig energies again, according to training set forces')
        prescale = prescale_energies(molecule, molecule.train)
        print('E SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('prescale value:', prescale)
        sys.stdout.flush()
        Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                bias_type=bias, extra_cols=extra_cols)
        print('N train: {} \nN test: {}'.format(len(molecule.train), 
            len(molecule.test)))
        print('train E min/max', 
                np.min(np.take(molecule.energies, molecule.train)), 
                np.max(np.take(molecule.energies, molecule.train)))
        train_forces = np.take(molecule.forces, molecule.train, axis=0)
        train_energies = np.take(molecule.energies, molecule.train, axis=0)
        print('E ORIG min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('training E ORIG min: {} max: {}'.format(np.min(train_energies), 
                np.max(train_energies)))
        print('F ORIG min: {} max: {}'.format(np.min(molecule.forces), 
                np.max(molecule.forces)))
        print('training F ORIG min: {} max: {}'.format(np.min(train_forces), 
                np.max(train_forces)))
        sys.stdout.flush()


        print('\nPlot decompFE vs r and histogram of decompFE '\
                '(if filtered=True, then for refined) data')
        interatomic_measures = Binner()
        interatomic_measures.get_bond_pop(molecule.coords, pairs)
        dists = interatomic_measures.rs.flatten()
        decompFE = molecule.mat_FE[:,:_NC2,].flatten()
        Plotter.xy_scatter([dists], [decompFE], [''], ['k'], '$r_{ij} / \AA$',
                'q / kcal/mol', [10], 'scatter-r-decompFE2.png')
        Plotter.hist_1d([molecule.mat_FE], 'q / kcal/mol', 'P(q)', 'hist_q2.png')
        sys.stdout.flush()

    '''
    network = Network(molecule)
    if load_model == None: 
        print('\nTrain ANN')
        #bias = 'sum'
        model = Network.get_coord_FE_model(network, molecule, prescale, 
                n_nodes=n_nodes, n_layers=n_layers, grad_loss_w=grad_loss_w, 
                qFE_loss_w=qFE_loss_w, E_loss_w=E_loss_w, bias=bias)
        sys.stdout.flush()

    if load_model != None:
        #model = load_model('best_ever_model_6')
        model = Network.get_coord_FE_model(network, molecule, prescale,
                training_model=False, load_model=load_model)
        sys.stdout.flush()
        print(datetime.now() - startTime)
        #################################################################
    '''


    file_path = input_files[0]
    E = np.genfromtxt(file_path+'/E_m100_m121.csv', delimiter=',', 
            dtype=None)
    F = np.genfromtxt(file_path+'/F_m100_m121.csv', delimiter=',', 
            dtype=None)
    L1_E = np.genfromtxt(file_path+'/L1_E_m100_m121.csv', delimiter=',', 
            dtype=None)
    L1_F = np.genfromtxt(file_path+'/L1_F_m100_m121.csv', delimiter=',', 
            dtype=None)
    stability = False
    if stability:
        stability_t = np.genfromtxt(file_path+'/stability_long_sims_m100.csv', 
                delimiter=',', dtype=None)


    headers = list(E[1:,0].astype(str))
    E = E[1:,1:].astype('float64')
    F = F[1:,1:].astype('float64')
    L1_E = L1_E[1:,1:].astype('float64')
    L1_F = L1_F[1:,1:].astype('float64')
    if stability:
        stability = stability_t[1:,1:].astype('float64')
    print(headers)
    print(E)

    val_idx = 2 #0, 2
    stdev_idx = 3 #1, 3
    max_val = 4 #2, 4
    Plotter.twinx_error_bars_plot([list(range(1,len(headers)+1))]*2, 
            [E.T[val_idx], L1_E.T[val_idx]], 
            [E.T[stdev_idx], L1_E.T[stdev_idx]], ['']*2, 
            ['b', 'r'], [':']*2, ['o']*2, 
            '', 'Energy MAE / kcal/mol', 'L1 / %', [100]*2, [1,2], 
            None, 'E.pdf', x_ticks_labels=headers, max_val=max_val)

    Plotter.twinx_error_bars_plot([list(range(1,len(headers)+1))]*2, 
            [F.T[val_idx], L1_F.T[val_idx]], 
            [F.T[stdev_idx], L1_F.T[stdev_idx]], ['']*2, 
            ['b', 'r'], [':']*2, ['o']*2, 
            '', 'Force MAE / kcal/mol/$\mathrm{\AA}$', 'L1 / %', [100]*2, [1,2], 
            None, 'F.pdf', x_ticks_labels=headers, max_val=max_val)

    if stability:
        Plotter.error_bars_plot([list(range(1,len(headers)+1))], 
            [stability.T[val_idx]], [stability.T[stdev_idx]], [''], 
            ['b'], [':'], ['o'], '', 'Time / ns', [100], 
            None, 'stability.pdf', x_ticks_labels=headers, max_val=52)


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
        group = parser.add_argument('-n_nodes', '--n_nodes', 
                action='store', default=1000, type=int, 
                help='number of nodes in neural network hidden layer/s')
        group = parser.add_argument('-n_layers', '--n_layers', 
                action='store', default=1, type=int, 
                help='number of dense layers in neural network')
        group = parser.add_argument('-n_training', '--n_training', 
                action='store', default=1000, type=int,
                help='number of data points for training neural network')
        group = parser.add_argument('-n_val', '--n_val', 
                action='store', default=50, type=int,
                help='number of data points for validating neural network')
        group = parser.add_argument('-n_test', '--n_test', 
                action='store', default=-1, type=int,
                help='number of data points for testing neural network')
        group = parser.add_argument('-grad_loss_w', '--grad_loss_w', 
                action='store', default=1000, type=int,
                help='loss weighting for gradients')
        group = parser.add_argument('-qFE_loss_w', '--qFE_loss_w', 
                action='store', default=1, type=int,
                help='loss weighting for pairwise decomposed forces '\
                        'and energies')
        group = parser.add_argument('-E_loss_w', '--E_loss_w', 
                action='store', default=1, type=int,
                help='loss weighting for energies')
        group = parser.add_argument('-bias', '--bias', 
                action='store', default='1/r', type=str,
                help='choose the bias used to describe decomposed/pairwise '\
                        'terms, options are - \n1:   No bias (bias=1)'\
                        '\n1/r:  1/r bias \n1/r2: 1/r^2 bias'\
                        '\nNRF:  bias using nuclear repulsive forces '\
                        '(zA zB/r^2) \nr:    r bias')
        group = parser.add_argument('-filtered', '--filtered', 
                action='store_true', 
                help='filter structures by removing high '\
                        'magnitude q structures')
        group = parser.add_argument('-load_model', '--load_model', 
                action='store', default=None,
                help='load an existing network to perform MD, provide '\
                        'the path+folder to model here')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    run_force_pred(input_files=op.input_files, atom_file=op.atom_file, 
            coord_files=op.coord_files, force_files=op.force_files, 
            energy_files=op.energy_files, charge_files=op.charge_files, 
            list_files=op.list_files, n_nodes=op.n_nodes, 
            n_layers=op.n_layers, n_training=op.n_training, n_val=op.n_val, 
            n_test=op.n_test, grad_loss_w=op.grad_loss_w, 
            qFE_loss_w=op.qFE_loss_w, E_loss_w=op.E_loss_w, bias=op.bias,
            filtered=op.filtered, load_model=op.load_model)

if __name__ == '__main__':
    main()


