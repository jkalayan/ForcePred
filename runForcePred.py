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

from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, AMBLAMMPSParser, AMBERParser, Binner, Writer, Plotter, \
        Network

import sys
#import numpy as np

#import os
#os.environ['OMP_NUM_THREADS'] = '8'


def run_force_pred(input_files='input_files', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class
    #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt
    '''
    AMBLAMMPSParser('molecules.prmtop', '1md.mdcrd',
        coord_files, force_files, energy_files, molecule)
    '''

    '''
    NPParser('types_z', 
            ['train_1frame_aspirin_coords'], 
            ['train_1frame_aspirin_forces'], molecule)
    '''
    '''
    NPParser('types_z', 
            ['train50000_aspirin_coordinates'], 
            ['train50000_aspirin_forces'], molecule)
    '''
    '''
    AMBERParser('molecules.prmtop', '1md.mdcrd', 'ascii.frc', 
            molecule)
    '''
    Molecule.check_force_conservation(molecule) #
    Converter(molecule) #get pairwise forces
    print(molecule)

    Writer.write_xyz(molecule.coords, molecule.atoms, 'coords.xyz', 'w')
    Writer.write_xyz(molecule.forces, molecule.atoms, 'forces.xyz', 'w')

    #'''
    Converter.get_rotated_forces(molecule)
    molecule.forces = molecule.rotated_forces
    molecule.coords = molecule.rotated_coords
    Molecule.check_force_conservation(molecule) #
    Converter(molecule) # get pairwise forces

    print(np.amax(molecule.mat_NRF), np.amin(molecule.mat_NRF))
    #sys.exit()

    Writer.write_xyz(molecule.rotated_coords, 
            molecule.atoms, 'rot_coords.xyz', 'w')
    Writer.write_xyz(molecule.rotated_forces, 
            molecule.atoms, 'rot_forces.xyz', 'w')
    #'''

    #print(molecule.mat_NRF[0])
    #Converter.get_coords_from_NRF(molecule.mat_NRF[0]/33, molecule.atoms,
            #molecule.coords[0], 33)
    #sys.exit()

    #Writer.write_gaus_cart(molecule.coords[0:3], 
            #molecule.atoms, 'SP Force', 'ethanediolSP')

    '''
    for i in range(5):
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        recomp = Network.get_recomposed_forces(molecule.coords[i], 
                molecule.mat_F[i].reshape(1,-1), n_atoms, _NC2)
        #print(molecule.mat_F[0].shape)
        print(molecule.forces[i], '\n')
        print(recomp,'\n')
        print()
    '''

    sys.stdout.flush()

    #'''
    run_net = True
    if run_net:
        #Molecule.make_train_test(molecule, molecule.energies) 
            #get train and test sets
        network = Network() #initiate network class
        Network.get_variable_depth_model(network, molecule) #train NN
        nsteps=1000
        Network.run_NVE(network, molecule, timestep=0.1, nsteps=nsteps)

    #'''


    '''
    NPParser('types_z', 
            ['train_asp_coords'], 
            ['train_asp_forces'], molecule)
    '''

    #OPTParser(input_files, molecule)

    '''
    AMBLAMMPSParser('molecules.prmtop', '1md.mdcrd', 
            ['Trajectory_npt_1.data'], 
            ['Forces_npt_1.data'], 
            ['Energy_npt_1.data'], molecule)
    '''

    #AMBERParser('molecules.prmtop', '1md.mdcrd', 'ascii.frc', 
            #molecule)
    #Molecule.check_force_conservation(molecule)
    #Converter(molecule)
    #print(molecule.mat_F)
    #Writer.write_xyz(molecule.coords, molecule.atoms, 
            #'test.xyz')




    '''
    bonds = Binner()
    list_bonds = [[1,4]]
    bonds.get_bond_pop(molecule, list_bonds)
    '''

    '''
    angles = Binner()
    list_angles = [[2, 1, 3]]
    angles.get_angle_pop(molecule, list_angles)
    '''

    '''
    dihedrals = Binner()
    list_dih = [[1, 2, 3, 6], [3, 2, 1, 7],
            [2, 1, 7, 10]]
    dihedrals.get_dih_pop(molecule, list_dih)

    phis_1 = [i[0] for i in molecule.phis]
    phis_2 = [i[1] for i in molecule.phis]
    phis_3 = [i[2] for i in molecule.phis]
    #_Es = molecule.energies
    #train, test = Molecule.make_train_test(molecule, _Es)
    #train = np.array(range(len(molecule.cooords)))

    #Plotter.xyz_scatter(np.take(phis_1, train), np.take(phis_2, train), 
            #np.take(_Es, train), 
            #'$\phi_1$', '$\phi_2$', '$U$', 'train-12c')
    #Plotter.xyz_scatter(np.take(phis_1, test), np.take(phis_2, test), 
            #np.take(_Es, test), 
            #'$\phi_1$', '$\phi_2$', '$U$', 'test-12')

    Plotter.hist_2d(phis_1, phis_2, '$\phi_1$', '$\phi_2$', 
            'dih_binned12')
    Plotter.hist_2d(phis_1, phis_3, '$\phi_1$', '$\phi_3$', 
            'dih_binned13')
    Plotter.hist_2d(phis_2, phis_3, '$\phi_2$', '$\phi_3$', 
            'dih_binned23')
    '''



    #Molecule.make_train_test(molecule, molecule.energies)
    #network = Network()
    #Network.get_variable_depth_model(network, molecule)

    #Permuter(molecule) #not used yet

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
        group = parser.add_argument('-c', '--coord_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing coordinates.')
        group = parser.add_argument('-f', '--force_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing forces.')
        group = parser.add_argument('-e', '--energy_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing energies.')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    run_force_pred(input_files=op.input_files, coord_files=op.coord_files, 
            force_files=op.force_files, energy_files=op.energy_files)

if __name__ == '__main__':
    main()


