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
from itertools import islice

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
from ForcePred.nn.Network_shared_ws import Network
from keras.models import Model, load_model    
from keras import backend as K                                              
import sys
import tensorflow as tf
import os
import math
import glob

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

    molecule_md17 = Molecule() #initiate molecule class
    md17_path = input_files[0]
    NPParser(atom_file, [md17_path+'/coords.txt'], [], [],
            #[md17_path+'/F.txt'], 
            #[md17_path+'/E.txt'], 
            molecule_md17)


    molecule_cp2k = Molecule() #initiate molecule class
    cp2k_path = input_files[1]
    NPParser(atom_file, [cp2k_path+'/C.txt'], 
            [], 
            [], 
            molecule_cp2k)

    n_atoms = len(molecule_md17.atoms)                                 
    masses = np.array([Converter._ZM[a] for a in molecule_md17.atoms]) 
    for i in range(len(molecule_md17.coords)):
        translate = False
        if i == 0:
            translate = True
        molecule_md17.coords[i] = Converter.superimpose_coords(
                molecule_md17.coords[0], molecule_md17.coords[i], 
                masses, n_atoms, translate)

    for i in range(len(molecule_cp2k.coords)):
        translate = False
        if i == 0:
            translate = True
        molecule_cp2k.coords[i] = Converter.superimpose_coords(
                molecule_md17.coords[0], molecule_cp2k.coords[i], 
                masses, n_atoms, translate)

    Writer.write_xyz(molecule_md17.coords, molecule_md17.atoms, 
            'C_md17_imaged.xyz', 'w')
    Writer.write_xyz(molecule_cp2k.coords, molecule_cp2k.atoms, 
            'C_cp2k_imaged.xyz', 'w')

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


