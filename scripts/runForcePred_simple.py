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

openmm = False
if openmm:
    from ForcePred.calculate.OpenMM import OpenMM

mdanal = False
if mdanal:
    from ForcePred.read.AMBLAMMPSParser import AMBLAMMPSParser
    from ForcePred.read.AMBERParser import AMBERParser
                               
from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, XYZParser, Binner, Writer, Plotter, Conservation
        #Network
from ForcePred.nn.Network_simple import Network
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
        qFE_loss_w='qFE_loss_w', E_loss_w='E_loss_w', bias='bias'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class

    #read in input data files
    if list_files:
        input_files = open(list_files).read().split()
    OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    print(datetime.now() - startTime)
    sys.stdout.flush()

    print(molecule)
    n_atoms = len(molecule.atoms)
    _NC2 = int(n_atoms * (n_atoms-1)/2)
    sys.stdout.flush()


    #split data into training, validation and test sets
    molecule.train = list(range(n_training))
    '''
    print('!!!use regularly spaced training data')
    train_split = math.ceil(len(molecule.coords) / n_training)
    print('train_split', train_split)
    molecule.train = np.arange(0, len(molecule.coords), split).tolist() 
    '''
    val_split = math.ceil(len(molecule.train) / n_val)
    molecule.val = molecule.train[::val_split]
    molecule.train2 = [x for x in molecule.train if x not in molecule.val]
    molecule.test = [x for x in range(0, len(molecule.coords)) 
            if x not in molecule.train]

    np.savetxt('idx_training.dat', (molecule.train), fmt='%s')
    np.savetxt('idx_val.dat', (molecule.val), fmt='%s')
    np.savetxt('idx_testset.dat', (molecule.test), fmt='%s')
    #print('train', molecule.train)
    #print('val', molecule.val)
    #print('train2', molecule.train2)
    #print('test', molecule.test)

    np.savetxt('molecule_z.dat', (molecule.atoms), fmt='%s')
    np.savetxt('molecule_c.dat', (molecule.coords).reshape(-1,3))
    np.savetxt('molecule_f.dat', (molecule.forces).reshape(-1,3))
    np.savetxt('moleculet_e.dat', (molecule.energies).reshape(-1,1))


    ### Scale energies using min/max forces in training set
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
    prescale_energies = True
    prescale = [0, 1, 0, 1]
    if prescale_energies:
        print('\nprescale energies so that magnitude is comparable to forces')
        min_e = np.min(train_energies)
        max_e = np.max(train_energies)
        min_f = np.min(train_forces)
        max_f = np.max(train_forces)
        molecule.energies = ((max_f - min_f) * (molecule.energies - min_e) / 
                (max_e - min_e) + min_f)
        prescale[0] = min_e
        prescale[1] = max_e
        prescale[2] = min_f
        prescale[3] = max_f
        print('E SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))

    print('prescale value:', prescale)
    sys.stdout.flush()


    bias = '1/r' #default for now
    print('\nget decomposed forces and energies simultaneously '\
            'with energy bias: {}'.format(bias))

    Converter.get_simultaneous_interatomic_energies_forces(molecule, bias, 
            remove_ebias=False)

    print('\ninternal FE decomposition')
    network = Network(molecule)
    model = Network.get_coord_FE_model(network, molecule, prescale, 
            n_nodes=n_nodes, n_layers=n_layers, grad_loss_w=grad_loss_w, 
            qFE_loss_w=qFE_loss_w, E_loss_w=E_loss_w)
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
        group = parser.add_argument('-n_nodes', '--n_nodes', 
                action='store', default=1000, type=int, 
                help='number of nodes in neural network hidden layer/s')
        group = parser.add_argument('-n_layers', '--n_layers', 
                action='store', default=1, type=int, 
                help='number of dense layers in neural network')
        group = parser.add_argument('-n_training', '--n_training', 
                action='store', default=1, type=int,
                help='number of data points for training neural network')
        group = parser.add_argument('-n_val', '--n_val', 
                action='store', default=1, type=int,
                help='number of data points for validating neural network')
        group = parser.add_argument('-n_test', '--n_test', 
                action='store', default=1, type=int,
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
            qFE_loss_w=op.qFE_loss_w, E_loss_w=op.E_loss_w, bias=op.bias)

if __name__ == '__main__':
    main()


