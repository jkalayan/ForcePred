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

    molecule = Molecule() #initiate molecule class
    molecule_name = input_files[0]
    base_path = '/mnt/iusers01/rb01/mbdxwjk2/scratch/'\
            'cp2k_qmmm/{}'.format(molecule_name)
    base_path2 = '/mnt/iusers01/rb01/mbdxwjk2/'\
            'rMD17_benchmark/{}'.format(molecule_name)

    #for mol_path in sorted(glob.glob(base_path)):

    #print(mol_path)
    all_mol_paths = []
    for i in range(1,11):
        all_mol_paths.append(base_path+'/cp2k_500K_100ps/'+str(i))
    print(all_mol_paths)

    #molecule = Molecule() #initiate molecule class
    # read in cp2k files
    #base_path = '/mnt/iusers01/rb01/mbdxwjk2/scratch/cp2k_qmmm/'\
            #'malonaldehyde'

    #cp2k_path = base_path+'/cp2k_500K_100ps/1'
    atom_order_file = base_path+'/revised_data/atom_order.txt'
    atoms = base_path2+'/revised_data/nuclear_charges.txt'
    sys.stdout.flush()


    atom_order = np.loadtxt(atom_order_file).astype(int)-1
    atom_order = list(atom_order)
    print(atom_order)
    molecule.atom_order = np.array(atom_order)
    molecule.atoms = np.array(np.loadtxt(atoms).astype(int))
    molecule.natoms = len(molecule.atoms) + 400*3
    print(len(molecule.atoms), molecule.natoms)
    all_atom_order = list(range(molecule.natoms))
    all_atom_order[:len(molecule.atoms)] = atom_order
    all_atom_order = np.array(all_atom_order)


    coord_paths = [cp2k_path+'/cp2k-qmmm-example-pos-1.xyz' 
            for cp2k_path in all_mol_paths]
    force_paths = [cp2k_path+'/cp2k-qmmm-example-frc-1.xyz' 
            for cp2k_path in all_mol_paths]
    dft_force_paths = [cp2k_path+'/cp2k-qmmm-example-dft-derivatives.frc' 
            for cp2k_path in all_mol_paths]
    dft_energy_paths = [cp2k_path+'/cp2k*-{}.o*'.format(molecule_name) 
            for cp2k_path in all_mol_paths]

    #for f in [cp2k_path+'/cp2k-qmmm-example-pos-1.xyz']:
    energies = []
    for f in coord_paths:
        file_energies = []
        input_ = open(f, 'r')
        for line in input_:
            if 'E = ' in line:
                cleaned = line.strip('\n').split()[-1:]
                file_energies.append(cleaned)
            else:
                continue
        energies += file_energies[-10000:]
    molecule.energies = np.array(energies).reshape(-1,1)[::5]
    print('** E', molecule.energies[:10])
    molecule.energies = np.asarray(molecule.energies, 
            dtype='float64') * Converter.Eh2kcalmol
    print('energies', molecule.energies[0], molecule.energies.shape)
    np.savetxt('whole_sys_E.txt', molecule.energies)
    sys.stdout.flush()


    def iterate_xyz_files(molecule, filenames, var):
        for f in filenames:
            file_var = []
            input_ = open(f, 'r')
            for line in input_:
                xyz = clean(molecule, extract(molecule, 1, input_))
                file_var.append(xyz)
            #print(file_var[0], len(file_var))
            var += file_var[-10000:]
        return var

    def extract(molecule, padding, input_):
        return (list(islice(input_, padding + 
                molecule.natoms))[-molecule.natoms:])

    def clean(molecule, raw):
        cleaned = np.empty(shape=[molecule.natoms, 3])
        for i, atom in enumerate(raw):
            cleaned[i] = atom.strip('\n').split()[-3:]
        #print(cleaned.shape)
        return np.array(cleaned)


    molecule.coords=[]
    molecule.coords = iterate_xyz_files(molecule, 
            #[cp2k_path+'/cp2k-qmmm-example-pos-1.xyz'], 
            coord_paths, 
            molecule.coords)
    molecule.coords = np.array(molecule.coords).reshape(
            -1,molecule.natoms, 3)[::5]
    #molecule.coords = np.take(molecule.coords, all_atom_order, axis=1)
    print('coords', molecule.coords[0][:len(molecule.atoms)], 
            molecule.coords.shape)
    np.savetxt('whole_sys_C.txt', molecule.coords.reshape(-1,3))
    #molecule.dft_coords = np.take(molecule.coords, molecule.atom_order, axis=1)
    molecule.dft_coords = molecule.coords[:,:len(molecule.atoms),:]
    sys.stdout.flush()

    print('dft_coords', molecule.dft_coords[0], molecule.dft_coords.shape)
    np.savetxt('C.txt', molecule.dft_coords.reshape(-1,3))
    Writer.write_xyz(molecule.dft_coords, molecule.atoms, 'C.xyz', 'w')

    Writer.write_gaus_cart(molecule.dft_coords,#[0].reshape(1,-1,3), 
            molecule.atoms, '', 'cp2k_all')

    molecule.forces=[]
    iterate_xyz_files(molecule, 
            #[cp2k_path+'/cp2k-qmmm-example-frc-1.xyz'], 
            force_paths,
            molecule.forces)
    molecule.forces = np.array(molecule.forces).reshape(
            -1,molecule.natoms, 3)[::5]
    #molecule.forces = np.take(molecule.forces, all_atom_order, axis=1)
    print('** forces', molecule.forces[0][:len(molecule.atoms)], 
            molecule.forces.shape)
    molecule.forces =  molecule.forces * \
            Converter.Eh2kcalmol / Converter.au2Ang
    print('forces', molecule.forces[0][:len(molecule.atoms)], 
            molecule.forces.shape)

    np.savetxt('whole_sys_F.txt', molecule.forces.reshape(-1,3))
    sys.stdout.flush()

    dft_forces = []
    #for f in [cp2k_path+'/cp2k-qmmm-example-dft-derivatives.frc']:
    for f in dft_force_paths:
        input_ = open(f, 'r')
        file_dft_forces = []
        for line in input_:
            if 'total' in line and 'Sum of total' not in line:
                cleaned = line.strip('\n').split()[-3:]
                file_dft_forces.append(cleaned)
            else:
                continue
        #print(file_dft_forces[0], len(file_dft_forces))
        dft_forces += file_dft_forces[-10000*len(molecule.atoms):]
    molecule.dft_forces = np.array(dft_forces).reshape(
            -1,len(molecule.atoms), 3)[::5]
    molecule.dft_forces = np.take(molecule.dft_forces, 
            molecule.atom_order, axis=1) 
    print('** dft_forces', molecule.dft_forces[0], molecule.dft_forces.shape)
    molecule.dft_forces = np.asarray(molecule.dft_forces, 
            dtype='float64') * Converter.Eh2kcalmol / Converter.au2Ang
    print('dft_forces', molecule.dft_forces[0], molecule.dft_forces.shape)
    np.savetxt('F.txt', molecule.dft_forces.reshape(-1,3)) 
    sys.stdout.flush()

    dft_energies = []
    #for f in [cp2k_path+'/cp2k1-malonaldehyde.o4057076']:
    for f in dft_energy_paths:
        for f2 in glob.glob(f):
            print(f2)
        input_ = open(f2, 'r')
        file_dft_energies = []
        for line in input_:
            if 'Total energy:' in line:
                cleaned = line.strip('\n').split()[-1:]
                file_dft_energies.append(cleaned)
            else:
                continue
        #print(file_dft_energies[0], len(file_dft_energies))
        dft_energies += file_dft_energies[-20000:][1::2] #to match other files
    molecule.dft_energies = np.array(dft_energies).reshape(-1,1)[::5]
    print('** dft_E', molecule.dft_energies[:10])
    molecule.dft_energies = np.asarray(molecule.dft_energies, 
            dtype='float64') * Converter.Eh2kcalmol
    print('dft_energies', molecule.dft_energies[0], molecule.dft_energies.shape)
    np.savetxt('E.txt', molecule.dft_energies) 
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


