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
from keras import backend as K                                              
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
    ##shorten num molecules here:
    trunc = 100
    molecule.coords = molecule.coords[0:trunc]
    molecule.forces = molecule.forces[0:trunc]
    molecule.energies = molecule.energies[0:trunc]
    '''


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

    print(molecule)
    n_atoms = len(molecule.atoms)
    _NC2 = int(n_atoms * (n_atoms-1)/2)

    A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    print(molecule.atoms)
    print(A)
    sys.stdout.flush()


    prescale_energies = True
    prescale = 0
    if prescale_energies:
        print('\nprescale energies so that magnitude is comparable to forces')
        prescale = np.max(np.absolute(molecule.energies))
        max_f = np.max(np.absolute(molecule.forces))
        print('ORIG min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        prescale = [prescale, max_f]
        molecule.energies = np.add(molecule.energies, prescale[0]) * \
                prescale[1]
        print('SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('prescale value:', prescale)
        #sys.exit()
        sys.stdout.flush()

    #Converter(molecule) #get pairwise forces
    #sys.exit()

    #get atomwise decomposed forces and NRF inputs
    get_atomF = False
    if get_atomF:
        Converter.get_atomwise_decompF(molecule, bias_type='NRF')
        recompF = Converter.get_atomwise_recompF([molecule.coords[0]], 
                [molecule.atom_F[0]], molecule.atoms, n_atoms, _NC2)
        print('atom_NRF', molecule.atom_NRF[0])
        print('atom_F', molecule.atom_F[0])
        print('recompF', recompF)
        print()
        print('F', molecule.forces[0])
        print(molecule.atom_NRF.shape, molecule.atom_F.shape)
        #molecule.atom_NRF = molecule.atom_NRF[:,0]
        #molecule.atom_F = molecule.atom_F[:,0]
        #sys.exit()
        sys.stdout.flush()

    get_decompF = False
    if get_decompF:
        print(molecule)
        print('\ncheck and get force decomp')
        unconserved = Molecule.check_force_conservation(molecule) #
        Converter(molecule) #get pairwise forces
        print('mat_r', molecule.mat_r[0])
        print('mat_NRF', molecule.mat_NRF[0])
        print('mat_F', molecule.mat_F[0])
        recompF = Conservation.get_recomposed_forces([molecule.coords[0]], 
                [molecule.mat_F[0]], n_atoms, _NC2)
        print('recompF', recompF)
        print('forces', molecule.forces[0])
        print(datetime.now() - startTime)
        sys.stdout.flush()
        #sys.exit()

    get_decompE = False
    bias_type = '1/r'  #'1/r' # #'NRF'
    print('\nenergy bias type: {}'.format(bias_type))
    if get_decompE:
        print('\nget mat_E')
        Converter.get_interatomic_energies(molecule, bias_type)
        print('molecule energy', molecule.energies[0])
        print('sum mat_E', np.sum(molecule.mat_E[0]))
        print(datetime.now() - startTime)
        sys.stdout.flush()

    get_decompFE = True
    if get_decompFE:
        print('\nget decomposed forces and energies simultaneously')
        #for some reason, using r as the bias does not give back recomp 
        #values, no idea why!
        Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                bias_type)
        for i in range(1):
            print('\ni', i)
            print('\nmat_FE', molecule.mat_FE[i])
            print('\nget recomposed FE')
            print('actual')
            print(molecule.forces[i])
            print(molecule.energies[i])
            n_atoms = len(molecule.atoms)
            _NC2 = int(n_atoms*(n_atoms-1)/2)
            recompF, recompE = Converter.get_recomposed_FE(
                    [molecule.coords[i]], [molecule.mat_FE[i]], 
                    molecule.atoms, n_atoms, _NC2, bias_type)
            print('\nrecomp from FE')
            print(recompF)
            print(recompE)
            '''
            print('\nrecomp from F only')
            recompF2 = Conservation.get_recomposed_forces(
                [molecule.coords[i]], [molecule.mat_F[i]], n_atoms, _NC2)
            print(recompF2)
            '''
            print(datetime.now() - startTime)
            sys.stdout.flush()
    #sys.exit()

    #'''
    print('\ninternal FE decomposition')
    network = Network(molecule)
    Network.get_coord_FE_model(network, molecule)
    print('exit')
    print(datetime.now() - startTime)
    sys.stdout.flush()
    sys.exit()
    #'''

    run_net = False
    split = 20 #2 4 5 20 52 260
    if run_net:
        train = round(len(molecule.coords) / split, 3)
        nodes = 1000
        input = molecule.mat_NRF #atom_NRF #
        output = molecule.mat_FE #atom_F #
        #output = molecule.energies
        print('\nget train and test sets, '\
                'training set is {} points.'\
                '\nNumber of nodes is {}'.format(train, nodes))
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        print('\nget ANN model')
        sys.stdout.flush()
        network = Network(molecule) #initiate network class
        train_prediction, test_prediction = Network.get_variable_depth_model(
                network, molecule, nodes, input, output) #train std NN
        #Network.get_decompE_sum_model(network, molecule, 
                #nodes, input, output) #train NN with sum energies

    sys.stdout.flush()


    train_input = np.take(molecule.coords, molecule.train, axis=0)
    train_e = np.take(molecule.energies, molecule.train, axis=0)
    train_f = np.take(molecule.forces, molecule.train, axis=0)

    test_input = np.take(molecule.coords, molecule.test, axis=0)
    test_e = np.take(molecule.energies, molecule.test, axis=0)
    test_f = np.take(molecule.forces, molecule.test, axis=0)


    if get_atomF:
        #n_atoms = 1
        #train_f = train_f[:,0]
        #test_f = test_f[:,0]
        train_pred_f = Converter.get_atomwise_recompF(
                train_input, train_prediction.reshape(-1,n_atoms,_NC2), 
                molecule.atoms, n_atoms, _NC2)
        test_pred_f = Converter.get_atomwise_recompF(
                test_input, test_prediction.reshape(-1,n_atoms,_NC2), 
                molecule.atoms, n_atoms, _NC2)

        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                    train_pred_f.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                    test_pred_f.flatten())
        print('\nForces:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

    if get_decompE:
        train_pred_e = np.sum(train_prediction, axis=1)
        test_pred_e = np.sum(test_prediction, axis=1)
        train_mae, train_rms = Binner.get_error(train_e.flatten(), 
                    train_pred_e.flatten())
        test_mae, test_rms = Binner.get_error(test_e.flatten(), 
                    test_pred_e.flatten())
        print('\nEnergies:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))
        print(train_e, train_pred_e)

    if get_decompF:
        train_pred_f = Conservation.get_recomposed_forces(train_input, 
                train_prediction, n_atoms, _NC2)
        test_pred_f = Conservation.get_recomposed_forces(test_input, 
                test_prediction, n_atoms, _NC2)
        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                    train_pred_f.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                    test_pred_f.flatten())
        print('\nForces:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

    if get_decompFE:
        train_pred_f, train_pred_e = Converter.get_recomposed_FE(train_input, 
                train_prediction, molecule.atoms, n_atoms, _NC2, bias_type)
        test_pred_f, test_pred_e = Converter.get_recomposed_FE(test_input, 
                test_prediction, molecule.atoms, n_atoms, _NC2, bias_type)

        print(train_e, train_pred_e)
        train_mae, train_rms = Binner.get_error(train_e.flatten(), 
                    train_pred_e.flatten())
        test_mae, test_rms = Binner.get_error(test_e.flatten(), 
                    test_pred_e.flatten())
        print('\nEnergies:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

        train_e = np.subtract(train_e / prescale[1], prescale[0])
        train_pred_e = np.subtract(train_pred_e / prescale[1], prescale[0])
        test_e = np.subtract(test_e / prescale[1], prescale[0])
        test_pred_e = np.subtract(test_pred_e / prescale[1], prescale[0])
        print(train_e, train_pred_e)
        print(train_f[0], train_pred_f[0])
        train_mae, train_rms = Binner.get_error(train_e.flatten(), 
                    train_pred_e.flatten())
        test_mae, test_rms = Binner.get_error(test_e.flatten(), 
                    test_pred_e.flatten())
        print('\nEnergies kcal/mol Ang:\nTrain MAE: {} \nTrain RMS: {} '\
                '\nTest MAE: {} \nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))
        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                    train_pred_f.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                    test_pred_f.flatten())
        print('\nForces from mat_FE:\nTrain MAE: {} \nTrain RMS: {} '\
                '\nTest MAE: {} \nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))


    scurves = True
    if scurves:
        train = None
        test = None
        train_pred = None
        test_pred = None
        if get_atomF or get_decompF or get_decompFE:
            train = train_f
            test = test_f
            train_pred = train_pred_f
            test_pred = test_pred_f
        if get_decompE: #get_decompFE or 
            train = train_e
            test = test_e
            train_pred = train_pred_e
            test_pred = test_pred_e

        print(train.shape, train_pred.shape, test.shape, test_pred.shape)

        train_bin_edges, train_hist = Binner.get_scurve(train.flatten(), 
                train_pred.flatten(), 'train-hist.txt')

        test_bin_edges, test_hist = Binner.get_scurve(test.flatten(), #actual
                test_pred.flatten(), #prediction
                'test-hist.txt')
       
        Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                [train_hist, test_hist], ['train', 'test'], 
                'Error', '% of points below error', 's-curves-recomp.png')

    #'''
    print('check decompE')
    ###malonaldehyde 
    #scale_NRF = 13088.721638547617
    #scale_NRF_min = 14.486865558162252
    ###decompE, 1/r 
    #scale_E = 18838.72967827923
    #scale_E_min = 679.9222840668109
    ###decompE, 1/qqr2
    #scale_E = 81838.51564994424 
    #scale_E_min = 1.8099282995681707

    #ethanediol r_6layers, syn scan
    #scale_NRF = 7990.182422267673 
    #scale_NRF_min = 16.93017848874296
    #bias r
    #scale_E = 10391.176054613952 
    #scale_E_min = 496.02300749772684 
    #bias 1/r
    #scale_E = 11947.537628797449 
    #scale_E_min = 570.8028197171167

    scale_NRF = network.scale_input_max 
    scale_NRF_min = network.scale_input_min
    scale_E = network.scale_output_max
    scale_E_min = network.scale_output_min
    dr = 0.001

    print('get network')
    network = Network.get_network(molecule, scale_NRF, scale_NRF_min, 
            scale_E, scale_E_min)

    #model = load_model('best_ever_model')

    c_loss = False
    if c_loss:
        def custom_loss1(weights):
            def custom_loss(y_true, y_pred):
                return K.mean(K.abs(y_true - y_pred) * weights)
            return custom_loss

        weights = np.zeros((_NC2+1)) #np.ones((_NC2+1)) #
        weights[-1] = 1 #sumE_weight
        cl = custom_loss1(weights)
        model = load_model('best_ever_model', 
                custom_objects={'custom_loss': custom_loss1(weights)})
    else:
        model = load_model('best_ever_model')
    #model = None
    sys.stdout.flush()

    if get_decompFE:
        print('get forces from finite difference')
        molecule.energies = np.subtract(molecule.energies / prescale[1], 
                prescale[0])
        #for i in range(len(molecule.coords)):
        for i in range(1):
            print('\ni', i)
            print('actual energy', molecule.energies[i])
            print('actual forces', molecule.forces[i])

            forces, curl = Conservation.get_forces_from_energy(
                    molecule.coords[i], 
                    molecule.atoms, scale_NRF, scale_NRF_min, scale_E, 
                    model, dr, bias_type, molecule, prescale)
            print('pred forces', forces)
            sys.stdout.flush()

        #get all Fs from Es with finite diff and check errors
        train_pred_f2 = np.zeros((len(train_input), n_atoms, 3))
        test_pred_f2 = np.zeros((len(test_input), n_atoms, 3))
        for s in range(len(train_input)):
            forces, curl = Conservation.get_forces_from_energy(
                    train_input[s], 
                    molecule.atoms, scale_NRF, scale_NRF_min, scale_E, 
                    model, dr, bias_type, molecule, prescale)
            train_pred_f2[s] = forces[0]
        for s in range(len(test_input)):
            forces, curl = Conservation.get_forces_from_energy(
                    test_input[s], 
                    molecule.atoms, scale_NRF, scale_NRF_min, scale_E, 
                    model, dr, bias_type, molecule, prescale)
            test_pred_f2[s] = forces[0]

        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                train_pred_f2.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                test_pred_f2.flatten())

        print(train_pred_f2.shape, test_pred_f2.shape)

        print('\nForces from Es:\nTrain MAE: {} \nTrain RMS: {}'\
                '\nTest MAE: {} \nTest RMS: {}'.format(
                train_mae, train_rms, test_mae, test_rms))

        train_bin_edges, train_hist = Binner.get_scurve(train_f.flatten(), 
                train_pred_f2.flatten(), 'train-hist2.txt')

        test_bin_edges, test_hist = Binner.get_scurve(test_f.flatten(),
                test_pred_f2.flatten(),
                'test-hist2.txt')
       
        Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                [train_hist, test_hist], ['train', 'test'], 
                'Error', '% of points below error', 's-curves-recomp2.png')

    #'''


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


