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

def run_force_pred(input_files='input_files'):


    startTime = datetime.now()

    print(startTime)
    
    molecule = Molecule() #initiate molecule class
    OPTParser(input_files, molecule) #read in FCEZ
    Molecule.check_force_conservation(molecule) #
    Converter(molecule) #check forces are invariant and get pairwise forces
    Molecule.make_train_test(molecule, molecule.energies) 
        #get train and test sets
    network = Network() #initiate network class
    Network.get_variable_depth_model(network, molecule) #train NN
    nsteps=5000
    mm = Network.run_NVE(network, molecule, timestep=0.5, nsteps=nsteps)
    coords, forces = [], []
    for i in range(len(mm.forces)):
        if i%(nsteps/100) == 0:
            coords.append(mm.coords[i])
            forces.append(mm.forces[i])
    Writer.write_xyz(coords, molecule.atoms, 
        'nn-coords.xyz')
    Writer.write_xyz(forces, molecule.atoms, 
        'nn-forces.xyz')

    Writer.write_xyz([molecule.coords[0]], molecule.atoms, 
        'coords.xyz')
    Writer.write_xyz([molecule.forces[0]], molecule.atoms, 
        'forces.xyz')

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
    _Es = molecule.energies
    train, test = Molecule.make_train_test(molecule, _Es)

    Plotter.xyz_scatter(np.take(phis_1, train), np.take(phis_2, train), 
            np.take(_Es, train), 
            '$\phi_1$', '$\phi_2$', '$U$', 'train-12c')
    Plotter.xyz_scatter(np.take(phis_1, test), np.take(phis_2, test), 
            np.take(_Es, test), 
            '$\phi_1$', '$\phi_2$', '$U$', 'test-12')
    #'''

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
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    run_force_pred(input_files=op.input_files)

if __name__ == '__main__':
    main()


