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
        Writer, Plotter, Conservation
from ForcePred.nn.Network_perminv import Network

from keras.models import Model, load_model    
import sys
#import numpy as np
#from itertools import islice

#import os
#os.environ['OMP_NUM_THREADS'] = '8'

import os
# Get number of cores reserved by the batch system
# ($NSLOTS is set by the batch system, or use 1 otherwise)
NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files', 
        dihedrals='dihedrals'):


    startTime = datetime.now()

    print(startTime)
    #molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()
    #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt

    scan1_molecule = Molecule() #initiate molecule class
    OPTParser(
            #[input_files[0], input_files[1]], 
            [input_files[0]], 
            scan1_molecule, opt=True) #read in FCEZ for opt
    scan1_molecule.orig_energies = np.copy(scan1_molecule.energies)
    #scan1_molecule.coords = scan1_molecule.coords[:-1]
    #scan1_molecule.forces = scan1_molecule.forces[:-1]
    #scan1_molecule.energies = scan1_molecule.energies[:-1]
    scan1_molecule.energies = np.abs(scan1_molecule.energies)
    print(scan1_molecule)

    scan2_molecule = Molecule() #initiate molecule class
    OPTParser(
            #[input_files[2], input_files[3]], 
            [input_files[1], input_files[2]], 
            scan2_molecule, opt=True) #read in FCEZ for opt
    scan2_molecule.orig_energies = np.copy(scan2_molecule.energies)
    print(scan2_molecule)

    scan3_molecule = Molecule() #initiate molecule class
    OPTParser(
            #[input_files[4], input_files[5]], 
            [input_files[3], input_files[4], input_files[5]], 
            scan3_molecule, opt=True) #read in FCEZ for opt
    scan3_molecule.orig_energies = np.copy(scan3_molecule.energies)
    print(scan3_molecule)

    print(datetime.now() - startTime)
    sys.stdout.flush()



    def scale_e(molecule, forces):
        min_e = np.min(molecule.energies[5:10])
        max_e = np.max(molecule.energies[5:10])
        min_f = np.min(np.abs(forces[5:10]))
        max_f = np.max(np.abs(forces[5:10]))
        molecule.energies = ((max_f - min_f) * (molecule.energies - min_e) / 
                (max_e - min_e) + min_f)

    bias_type='qq/r2'
    Converter(scan1_molecule) #to get mat_F
    #scale_e(scan1_molecule, scan1_molecule.forces)
    scale_e(scan1_molecule, 
            np.concatenate((scan1_molecule.forces, scan2_molecule.forces, 
            scan2_molecule.forces)))
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='qq/r2')
    #'''
    scan1_molecule.mat_FE_qqr2 = np.copy(scan1_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='1/r')
    scan1_molecule.mat_FE_recipr = np.copy(scan1_molecule.mat_FE)
    print('\nrecomp from FE with dot 1/r')
    recompF = np.dot(scan1_molecule.mat_eij[0][:-1], 
            scan1_molecule.mat_FE_recipr[0])
    recompE = np.dot(scan1_molecule.mat_bias[0], 
            scan1_molecule.mat_FE_recipr[0])
    print('recompF', recompF.reshape(-1,3))
    print('recompE', recompE)
    print()
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='1/r2')
    scan1_molecule.mat_FE_recipr2 = np.copy(scan1_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='r2')
    scan1_molecule.mat_FE_r2 = np.copy(scan1_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='1/qqr2')
    scan1_molecule.mat_FE_recipqqr2 = np.copy(scan1_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='r3')
    scan1_molecule.mat_FE_r3 = np.copy(scan1_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='1/r3')
    scan1_molecule.mat_FE_recipr3 = np.copy(scan1_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='r')
    scan1_molecule.mat_FE_r = np.copy(scan1_molecule.mat_FE)
    print('\nrecomp from FE with dot r')
    recompF = np.dot(scan1_molecule.mat_eij[0][:-1], 
            scan1_molecule.mat_FE_r[0])
    recompE = np.dot(scan1_molecule.mat_bias[0], 
            scan1_molecule.mat_FE_r[0])
    print('recompF', recompF.reshape(-1,3))
    print('recompE', recompE)
    print()
    print('F', scan1_molecule.forces[0])
    print('E', scan1_molecule.energies[0])
    print()

    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='1')
    scan1_molecule.mat_FE_1 = np.copy(scan1_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan1_molecule, 
            bias_type='0')
    scan1_molecule.mat_FE_0 = np.copy(scan1_molecule.mat_FE)
    print('mat_F', scan1_molecule.mat_F[:,36])
    print('bias 0', scan1_molecule.mat_FE_0[:,36])
    print('bias 1', scan1_molecule.mat_FE_1[:,36])
    print('bias r', scan1_molecule.mat_FE_r[:,36])
    print()
    #print(scan1_molecule.mat_FE_qqr2)
    #'''


    

    scale_e(scan2_molecule, 
            np.concatenate((scan1_molecule.forces, scan2_molecule.forces, 
            scan2_molecule.forces)))
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type='qq/r2')
    scan2_molecule.mat_FE_qqr2 = np.copy(scan2_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type='0')
    scan2_molecule.mat_FE_0 = np.copy(scan2_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type='1')
    scan2_molecule.mat_FE_1 = np.copy(scan2_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type='r')
    scan2_molecule.mat_FE_r = np.copy(scan2_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type='1/r')
    scan2_molecule.mat_FE_recipr = np.copy(scan2_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type='1/r2')
    scan2_molecule.mat_FE_recipr2 = np.copy(scan2_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan2_molecule, 
            bias_type='1/r3')
    scan2_molecule.mat_FE_recipr3 = np.copy(scan2_molecule.mat_FE)

    scale_e(scan3_molecule, 
            np.concatenate((scan1_molecule.forces, scan2_molecule.forces, 
            scan2_molecule.forces)))
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type='qq/r2')
    scan3_molecule.mat_FE_qqr2 = np.copy(scan3_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type='0')
    scan3_molecule.mat_FE_0 = np.copy(scan3_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type='1')
    scan3_molecule.mat_FE_1 = np.copy(scan3_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type='r')
    scan3_molecule.mat_FE_r = np.copy(scan3_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type='1/r')
    scan3_molecule.mat_FE_recipr = np.copy(scan3_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type='1/r2')
    scan3_molecule.mat_FE_recipr2 = np.copy(scan3_molecule.mat_FE)
    Converter.get_simultaneous_interatomic_energies_forces(scan3_molecule, 
            bias_type='1/r3')
    scan3_molecule.mat_FE_recipr3 = np.copy(scan3_molecule.mat_FE)


    print(
            'E', scan1_molecule.energies.T, '\n',
            #scan2_molecule.energies, '\n',
            #scan3_molecule.energies, '\n',
            'mat_F', np.sum(scan1_molecule.mat_F, axis=1), '\n',
            'bias 0', np.sum(scan1_molecule.mat_FE_0, axis=1), '\n',
            'bias 1', np.sum(scan1_molecule.mat_FE_1, axis=1), '\n',
            'bias r', np.sum(scan1_molecule.mat_FE_r, axis=1), '\n',
            'bias 1/r', np.sum(scan1_molecule.mat_FE_recipr, axis=1), '\n',
            #np.sum(scan1_molecule.mat_FE_recipqqr2, axis=1), '\n',
            )


    print(datetime.now() - startTime)
    sys.stdout.flush()

    n_atoms = len(scan1_molecule.atoms)
    _NC2 = int(n_atoms * (n_atoms-1)/2)
    count = 0
    for i in range(len(scan1_molecule.atoms)):
        for j in range(i):
            print(count, '-', i, j)
            count += 1


    scan1_measures = Binner()
    #scan1_measures.get_dih_pop(scan1_molecule.coords, [[4,0,1,2]]) #malon
    scan1_measures.get_dih_pop(scan1_molecule.coords, [[0,5,7,9]]) #asp
    inds = np.argwhere(scan1_measures.phis.flatten() < 190) #& 
            #(scan1_measures.phis.flatten() > -170))
    scan1_measures.phis = scan1_measures.phis.flatten()[inds].reshape(-1,1)
    scan1_molecule.energies = scan1_molecule.energies[inds]
    scan1_molecule.orig_energies = scan1_molecule.orig_energies[inds]
    scan1_molecule.mat_FE_qqr2 = \
            scan1_molecule.mat_FE_qqr2[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_recipr = \
            scan1_molecule.mat_FE_recipr[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_recipr2 = \
            scan1_molecule.mat_FE_recipr2[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_r2 = \
            scan1_molecule.mat_FE_r2[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_recipqqr2 = \
            scan1_molecule.mat_FE_recipqqr2[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_r = \
            scan1_molecule.mat_FE_r[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_1 = \
            scan1_molecule.mat_FE_1[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_0 = \
            scan1_molecule.mat_FE_0[inds].reshape(-1,_NC2)
    scan1_molecule.mat_FE_recipr3 = \
            scan1_molecule.mat_FE_recipr3[inds].reshape(-1,_NC2)

    scan2_measures = Binner()
    #scan2_measures.get_bond_pop(scan2_molecule.coords, [[4,0]]) #malon
    scan2_measures.get_bond_pop(scan2_molecule.coords, [[7,10]]) #asp
    inds = np.argwhere(scan2_measures.rs.flatten() <= 2)
    scan2_measures.rs = scan2_measures.rs.flatten()[inds].reshape(-1,1)
    scan2_molecule.energies = scan2_molecule.energies[inds]
    scan2_molecule.orig_energies = scan2_molecule.orig_energies[inds]
    scan2_molecule.mat_FE_qqr2 = \
            scan2_molecule.mat_FE_qqr2[inds].reshape(-1,_NC2)
    scan2_molecule.mat_FE_0 = \
            scan2_molecule.mat_FE_0[inds].reshape(-1,_NC2)
    scan2_molecule.mat_FE_1 = \
            scan2_molecule.mat_FE_1[inds].reshape(-1,_NC2)
    scan2_molecule.mat_FE_r = \
            scan2_molecule.mat_FE_r[inds].reshape(-1,_NC2)
    scan2_molecule.mat_FE_recipr = \
            scan2_molecule.mat_FE_recipr[inds].reshape(-1,_NC2)
    scan2_molecule.mat_FE_recipr2 = \
            scan2_molecule.mat_FE_recipr2[inds].reshape(-1,_NC2)
    scan2_molecule.mat_FE_recipr3 = \
            scan2_molecule.mat_FE_recipr3[inds].reshape(-1,_NC2)

    scan3_measures = Binner()
    #scan3_measures.get_bond_pop(scan3_molecule.coords, [[0,1]]) #malon
    scan3_measures.get_bond_pop(scan3_molecule.coords, [[6,12]]) #asp
    inds = np.argwhere((scan3_measures.rs.flatten() <= 2) & 
            (scan3_measures.rs.flatten() >= 1.2))
    #inds = np.argwhere(scan3_measures.rs.flatten() <= 2)
    scan3_measures.rs = scan3_measures.rs.flatten()[inds].reshape(-1,1)
    scan3_molecule.energies = scan3_molecule.energies[inds]
    scan3_molecule.orig_energies = scan3_molecule.orig_energies[inds]
    scan3_molecule.mat_FE_qqr2 = \
            scan3_molecule.mat_FE_qqr2[inds].reshape(-1,_NC2)
    scan3_molecule.mat_FE_0 = \
            scan3_molecule.mat_FE_0[inds].reshape(-1,_NC2)
    scan3_molecule.mat_FE_1 = \
            scan3_molecule.mat_FE_1[inds].reshape(-1,_NC2)
    scan3_molecule.mat_FE_r = \
            scan3_molecule.mat_FE_r[inds].reshape(-1,_NC2)
    scan3_molecule.mat_FE_recipr = \
            scan3_molecule.mat_FE_recipr[inds].reshape(-1,_NC2)
    scan3_molecule.mat_FE_recipr2 = \
            scan3_molecule.mat_FE_recipr2[inds].reshape(-1,_NC2)
    scan3_molecule.mat_FE_recipr3 = \
            scan3_molecule.mat_FE_recipr3[inds].reshape(-1,_NC2)

    num_repeats = 7
    #dih = 8 #malon
    dih = 36 #asp
    Plotter.twinx_plot(
            [scan1_measures.phis.T.flatten()]*num_repeats, 
            [
                scan1_molecule.energies.flatten(),
                #scan1_molecule.mat_FE[:,dih].flatten(),
                scan1_molecule.mat_FE_1[:,dih].flatten(),
                #scan1_molecule.mat_FE_qqr2[:,dih].flatten(), ##
                #scan1_molecule.mat_FE_recipqqr2[:,dih].flatten(), ##
                scan1_molecule.mat_FE_recipr[:,dih].flatten(),
                #scan1_molecule.mat_FE_r2[:,dih].flatten(), ##
                scan1_molecule.mat_FE_recipr2[:,dih].flatten(),
                scan1_molecule.mat_FE_recipr3[:,dih].flatten(),
                scan1_molecule.mat_FE_r[:,dih].flatten(), ##
                scan1_molecule.mat_FE_0[:,dih].flatten(),

            ], 
            ['$E\'_\mathrm{scan}$', 
                '$X_{ij} = 1$',
                '$X_{ij} = || \\vec{r_{ij}}||^{-1}$',
                '$X_{ij} = || \\vec{r_{ij}}||^{-2}$',
                '$X_{ij} = || \\vec{r_{ij}}||^{-3}$',
                '$X_{ij} = || \\vec{r_{ij}}||$',
                '$X_{ij} = 0$',

            ],
            #['']*num_repeats, 
            ['dodgerblue', 'darkorange', 'grey', 'purple', 'darkgreen', 
                'navy', 'k'], 
            ['-', '--', '-.', '--', '-.', ':', ':'],
            ['o', 's', 'D', 'o', 'd', 'v', '^'],
            '$\\tau$ / degrees', '$q_{ij}$', 
            '$E\'_\mathrm{scan}$ / kcal mol$^-1$ $\mathrm{\AA}^{-1}$', 
            [100]*num_repeats, 
            [2] + [1]*(num_repeats-1), 'f',
            'scan-dih.pdf')

    #b1 = 6 #malon
    #b2 = 0 #malon
    b1 = 52 #asp
    b2 = 72 #asp
    num_repeats = 7
    Plotter.twinx_plot(
            [scan2_measures.rs.T.flatten()]*num_repeats +
            [scan3_measures.rs.T.flatten()]*num_repeats
            , 
            [
                scan2_molecule.energies.flatten(), 
                #scan2_molecule.mat_FE_qqr2[:,b1].flatten(),
                scan2_molecule.mat_FE_1[:,b1].flatten(),
                scan2_molecule.mat_FE_recipr[:,b1].flatten(),
                scan2_molecule.mat_FE_recipr2[:,b1].flatten(),
                scan2_molecule.mat_FE_recipr3[:,b1].flatten(),
                scan2_molecule.mat_FE_r[:,b1].flatten(),
                scan2_molecule.mat_FE_0[:,b1].flatten(),

                scan3_molecule.energies.flatten(),
                #scan3_molecule.mat_FE_qqr2[:,b2].flatten(),
                scan3_molecule.mat_FE_1[:,b2].flatten(),
                scan3_molecule.mat_FE_recipr[:,b2].flatten(),
                scan3_molecule.mat_FE_recipr2[:,b2].flatten(),
                scan3_molecule.mat_FE_recipr3[:,b2].flatten(),
                scan3_molecule.mat_FE_r[:,b2].flatten(),
                scan3_molecule.mat_FE_0[:,b2].flatten(),
            ], 
            ['']*num_repeats*2, 
            ['dodgerblue', 'darkorange', 'grey', 'purple', 'darkgreen', 
                'navy', 'k', 
            'dodgerblue', 'darkorange', 'grey', 'purple', 'darkgreen', 
                'navy', 'k'], 
            ['-', '--', '-.', '--', '-.', ':', ':',
                '-', '--', '-.', '--', '-.', ':', ':'], 
            ['o', 's', 'D', 'o', 'd', 'v', '^', 
                'o', 's', 'D', 'o', 'd', 'v', '^',],
            '$|| \\vec{r_{ij}} ||$ / $\mathrm{\AA}$', 
            '$q_{ij}$', 
            '$E\'_\mathrm{scan}$ / kcal mol$^-1$ $\mathrm{\AA}^{-1}$', 
            [100]*num_repeats*2, 
            [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], 'p',
            'scan-rs.pdf')



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
        group = parser.add_argument('-d', '--dihedrals', nargs='+', 
                action='append', type=int, default=[],
                help='list of dihedrals')
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
            dihedrals=op.dihedrals, list_files=op.list_files)

if __name__ == '__main__':
    main()


