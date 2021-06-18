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
        Permuter, AMBLAMMPSParser, AMBERParser, XYZParser, Binner, \
        Writer, Plotter, Network

import sys
#import numpy as np

#import os
#os.environ['OMP_NUM_THREADS'] = '8'


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        list_files='list_files'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()
    #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt
    '''
    AMBLAMMPSParser('molecules.prmtop', '1md.mdcrd',
        coord_files, force_files, energy_files, molecule)
    '''

    '''
    NPParser('types_z', 
            ['train1_aspirin_coordinates'], 
            ['train1_aspirin_forces'], molecule)
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

    #'''
    XYZParser(atom_file, coord_files, force_files, 
            energy_files, molecule)
    #'''


    '''
    Writer.write_xyz(molecule.coords, molecule.atoms, 'qm-coords.xyz', 'w')
    Writer.write_xyz(molecule.forces, molecule.atoms, 'qm-forces.xyz', 'w')
    np.savetxt('qm-energies.txt', molecule.energies)
    '''

    get_decomp = True
    if get_decomp:
        print(molecule)
        print('check and get force decomp')
        Molecule.check_force_conservation(molecule) #
        Converter(molecule) #get pairwise forces




    get_rotate = False
    if get_rotate:
        print('rotate coords/forces')
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

    get_train_from_dih = False
    if get_train_from_dih:
        dihedrals = Binner()
        list_dih = [[1, 2, 3, 6], [3, 2, 1, 7],
                [2, 1, 7, 10]]
        '''
        list_dih = [[1,2,3,6], [3,2,1,7], [2,1,7,10], [4,2,3,6], [5,2,3,6],
                [4,2,1,7], [5,2,1,7], [4,2,1,8], [5,2,1,8], [4,2,1,9], 
                [5,2,1,9], [8,1,2,3], [9,1,2,3], [8,1,7,10], [9,1,7,10]]
        '''
        dihedrals.get_dih_pop(molecule.coords, list_dih)

        phis_1 = [i[0] for i in dihedrals.phis]
        phis_2 = [i[1] for i in dihedrals.phis]
        phis_3 = [i[2] for i in dihedrals.phis]

        p1_list = []
        p2_list = []
        p3_list = []
        train_phis = []
        for i, p1, p2, p3 in zip(range(1,len(phis_1)+1), phis_1, 
                phis_2, phis_3):
            if int(p1) not in p1_list or int(p2) not in p2_list or \
                    int(p3) not in p3_list:
                train_phis.append(i)
                p1_list.append(int(p1))
                p2_list.append(int(p2))
                p3_list.append(int(p3))
            else:
                continue

        test_phis = []
        for i in range(1, len(phis_1)+1):
            if i not in train_phis:
                test_phis.append(i)
            else:
                continue

        train_phis = np.array(train_phis)
        test_phis = np.array(test_phis)
        print('train_phis', train_phis.shape, 'test_phis', test_phis.shape)

                


    run_net = True
    split = 2 #2 4 5 20 52 260
    if run_net:
        print('get NN')
        Molecule.make_train_test(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        #override train test
        #molecule.train = train_phis
        #molecule.test = test_phis
        network = Network() #initiate network class
        ###Network.get_variable_depth_model(network, molecule) #train NN
        nsteps=1000
        mm = Network.run_NVE(network, molecule, timestep=0.5, nsteps=nsteps)


    print('checking mm forces')
    mm.coords = mm.get_3D_array([mm.coords])
    mm.forces = mm.get_3D_array([mm.forces]) 
    mm.energies = np.array(mm.energies)
    mm.check_force_conservation()
    #sys.exit()

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
    bonds.get_bond_pop(molecule.coords, list_bonds)
    '''

    '''
    angles = Binner()
    list_angles = [[2, 1, 3]]
    angles.get_angle_pop(molecule.coords, list_angles)
    '''

    get_dihs = False
    if get_dihs:
        dihedrals = Binner()
        list_dih = [[1, 2, 3, 6], [3, 2, 1, 7],
                [2, 1, 7, 10]]
        dihedrals.get_dih_pop(molecule.coords, list_dih)

        phis_1 = [i[0] for i in dihedrals.phis]
        phis_2 = [i[1] for i in dihedrals.phis]
        phis_3 = [i[2] for i in dihedrals.phis]

        _Es = molecule.energies.flatten()
        Molecule.make_train_test(molecule, _Es, split)
        train = molecule.train
        test = molecule.test
        #train = np.array(range(len(molecule.coords)))

        Plotter.xyz_scatter(np.take(phis_1, train), np.take(phis_2, train), 
                np.take(_Es, train), 
                '$\phi_1$', '$\phi_2$', '$U$', 'train-12.png')
        Plotter.xyz_scatter(np.take(phis_1, test), np.take(phis_2, test), 
                np.take(_Es, test), 
                '$\phi_1$', '$\phi_2$', '$U$', 'test-12.png')

        mm_dihedrals = Binner()
        mm_dihedrals.get_dih_pop(mm.coords, list_dih)
        mm_phis_1 = [i[0] for i in mm_dihedrals.phis]
        mm_phis_2 = [i[1] for i in mm_dihedrals.phis]
        mm_phis_3 = [i[2] for i in mm_dihedrals.phis]


        Plotter.xy_scatter([np.take(phis_1, train), mm_phis_1], 
                [np.take(phis_2, train), mm_phis_2], 
                ['train', 'mm'], ['k', 'r'], 
                '$\phi_1$', '$\phi_2$', 'train-xy-12.png')
        Plotter.xy_scatter([np.take(phis_1, test)], [np.take(phis_2, test)], 
                [''], ['k'], '$\phi_1$', '$\phi_2$', 'test-xy-12.png')
        Plotter.xy_scatter([np.take(phis_1, train), mm_phis_1], 
                [np.take(phis_3, train), mm_phis_3], 
                ['train', 'mm'], ['k', 'r'], 
                '$\phi_1$', '$\phi_3$', 'train-xy-13.png')
        Plotter.xy_scatter([np.take(phis_1, test)], [np.take(phis_3, test)], 
                [''], ['k'], '$\phi_1$', '$\phi_3$', 'test-xy-13.png')

        Plotter.hist_2d(phis_1, phis_2, '$\phi_1$', '$\phi_2$', 
                'dih_hist12')
        Plotter.hist_2d(phis_1, phis_3, '$\phi_1$', '$\phi_3$', 
                'dih_hist13')
        Plotter.hist_2d(phis_2, phis_3, '$\phi_2$', '$\phi_3$', 
                'dih_hist23')

        Plotter.xy_scatter([phis_1], [phis_2], [''], ['k'], 
                '$\phi_1$', '$\phi_2$', 'dih_xy12')
        Plotter.xy_scatter([phis_1], [phis_3], [''], ['k'], 
                '$\phi_1$', '$\phi_3$', 'dih_xy13')
        Plotter.xy_scatter([phis_2], [phis_3], [''], ['k'], 
                '$\phi_2$', '$\phi_3$', 'dih_xy23')


    scurves = False
    if scurves:
        train_bin_edges, train_hist = Binner.get_scurve(
                np.loadtxt('train-actual-decomp-force.txt').flatten(), 
                np.loadtxt('train-decomp-force.txt').flatten(), 
                'train-hist.txt')

        test_bin_edges, test_hist = Binner.get_scurve(
                np.loadtxt('test-actual-decomp-force.txt').flatten(), 
                np.loadtxt('test-decomp-force.txt').flatten(), 
                'test-hist.txt')
       
        Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                [train_hist, test_hist], ['train', 'test'], 
                'Error', '% of points below error', 's-curves.png')

    '''
    coords = np.take(molecule.coords, molecule.train, axis=0)
    forces = np.take(molecule.forces, molecule.train, axis=0)
    mat_NRF = np.loadtxt('train-NRF.txt')
    decomp = np.loadtxt('train-decomp-force.txt')
    print('\nmat_NRF', mat_NRF.shape)
    print('\ndecomp', decomp.shape)

    measures = Binner()
    list_bonds = [[1,2]]
    measures.get_bond_pop(coords, list_bonds)
    _CC = decomp[:,0]
    print(measures.rs.shape, _CC.shape)
    Plotter.xy_scatter([measures.rs], [_CC], ['CC bond'], ['k'], 
            '$r$', 'decomp F', 'train-bond-CC.png')

    list_angles = [[2, 1, 3]]
    measures.get_angle_pop(coords, list_angles)
    _OCC = decomp[:,2]
    Plotter.xy_scatter([measures.thetas], [_OCC], ['OCC angle'], ['k'],
            '$\\theta$', 'decomp F', 'train-angle-OCC.png')

    list_dih = [[3, 2, 1, 7]]
    measures.get_dih_pop(coords, list_dih)
    _OCCO = decomp[:,17]
    Plotter.xy_scatter([measures.phis], [_OCCO], ['OCCO dih'], ['k'],
            '$\Phi$', 'decomp F', 'train-dih-OCCO.png')

    coords = np.take(molecule.coords, molecule.test, axis=0)
    forces = np.take(molecule.forces, molecule.test, axis=0)
    mat_NRF = np.loadtxt('test-NRF.txt')
    decomp = np.loadtxt('test-decomp-force.txt')
    print('\nmat_NRF', mat_NRF.shape)
    print('\ndecomp', decomp.shape)

    measures = Binner()
    list_bonds = [[1,2]]
    measures.get_bond_pop(coords, list_bonds)
    _CC = decomp[:,0]
    print(measures.rs.shape, _CC.shape)
    Plotter.xy_scatter([measures.rs], [_CC], ['CC bond'], ['k'], 
            '$r$', 'decomp F', 'test-bond-CC.png')

    list_angles = [[2, 1, 3]]
    measures.get_angle_pop(coords, list_angles)
    _OCC = decomp[:,2]
    Plotter.xy_scatter([measures.thetas], [_OCC], ['OCC angle'], ['k'], 
            '$\\theta$', 'decomp F', 'test-angle-OCC.png')

    list_dih = [[3, 2, 1, 7]]
    measures.get_dih_pop(coords, list_dih)
    _OCCO = decomp[:,17]
    Plotter.xy_scatter([measures.phis], [_OCCO], ['OCCO dih'], ['k'], 
            '$\Phi$', 'decomp F', 'test-dih-OCCO.png')
    '''








    '''
    #for h in range(len(decomp)):
    for h in range(1):
        n = -1
        for i in range(len(molecule.atoms)):
            zA = molecule.atoms[i]
            for j in range(i):
                n += 1
                zB = molecule.atoms[j]
                _NRF = mat_NRF[h][n]
                #r = molecule.mat_r[h][n]
                #r_recomp = (zA * zB / _NRF) ** 0.5
                #print(n+1, '-', i+1, j+1, '-', zA, zB, '-', decomp[h][n])
                #print('\t', _NRF, r, r_recomp)
    '''


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
            energy_files=op.energy_files, list_files=op.list_files)

if __name__ == '__main__':
    main()


