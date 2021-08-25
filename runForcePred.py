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
        Writer, Plotter, Network, Conservation

import sys
#import numpy as np

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
    OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #print(molecule.charges)
    #print(molecule.charges.shape)
    #Converter.get_interatomic_charges(molecule)
    #sys.exit()
    #OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt
    '''
    AMBLAMMPSParser('molecules.prmtop', '1md.mdcrd',
        coord_files, force_files, energy_files, molecule)
    '''

    #NPParser(atom_file, coord_files, force_files, energy_files, molecule)

    '''
    NPParser('../types_z', 
            ['../train1_aspirin_coordinates'], 
            ['../train1_aspirin_forces'], [], molecule)
    '''
    '''
    NPParser('types_z', 
            ['train50000_aspirin_coordinates'], 
            ['train50000_aspirin_forces'], [], molecule)
    '''
    '''
    AMBERParser('molecules.prmtop', '1md.mdcrd', 'ascii.frc', 
            molecule)
    '''

    '''
    XYZParser(atom_file, coord_files, force_files, 
            energy_files, molecule)
    '''


    '''
    Writer.write_xyz(molecule.coords, molecule.atoms, 'ensemble-coords.xyz', 'w')
    Writer.write_xyz(molecule.forces, molecule.atoms, 'ensemble-forces.xyz', 'w')
    np.savetxt('ensemble-energies.txt', molecule.energies)
    np.savetxt('ensemble-charges.txt', molecule.charges)
    sys.exit()
    '''

    #'''
    Writer.write_gaus_cart(molecule.coords, 
            molecule.atoms, 'FORCE POP=MK INTEGRAL=(GRID=ULTRAFINE)', 'mm')

    sys.stdout.flush()
    sys.exit()
    #'''

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


    get_decomp = True
    if get_decomp:
        print(molecule)
        print('\ncheck and get force decomp')
        unconserved = Molecule.check_force_conservation(molecule) #
        Converter(molecule) #get pairwise forces

    '''
    print('\nget equivalent atoms')
    sorted_N_list, resorted_N_list = Molecule.get_sorted_pairs(
            molecule.mat_NRF, pairs_dict)
    #print('sorted_N_list', sorted_N_list)
    #print('resorted_N_list', resorted_N_list)

    #a = np.array([[0,1,2],[3,4,5]])
    #ind = np.array([[0,2,1],[1,2,0]])
    #print(np.take_along_axis(a, ind, axis=1))
    #sys.exit()
    '''

    sys.stdout.flush()

    get_rotate = False
    if get_rotate:
        print('rotate coords/forces')
        Converter.get_rotated_forces(molecule)
        molecule.forces = molecule.rotated_forces
        molecule.coords = molecule.rotated_coords
        unconserved = Molecule.check_force_conservation(molecule) #
        Converter(molecule) # get pairwise forces

        print(np.amax(molecule.mat_NRF), np.amin(molecule.mat_NRF))
        #sys.exit()

        Writer.write_xyz(molecule.rotated_coords, 
                molecule.atoms, 'rot_coords.xyz', 'w')
        Writer.write_xyz(molecule.rotated_forces, 
                molecule.atoms, 'rot_forces.xyz', 'w')

    sys.stdout.flush()
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
        #list_dih = [[1, 2, 3, 6], [3, 2, 1, 7],
                #[2, 1, 7, 10]]
        '''
        list_dih = [[1,2,3,6], [3,2,1,7], [2,1,7,10], [4,2,3,6], [5,2,3,6],
                [4,2,1,7], [5,2,1,7], [4,2,1,8], [5,2,1,8], [4,2,1,9], 
                [5,2,1,9], [8,1,2,3], [9,1,2,3], [8,1,7,10], [9,1,7,10]]
        '''
        list_dih = []
        n_atoms = len(molecule.atoms)
        for i in range(1, n_atoms+1):
            for j in range(i):
                for k in range(j):
                    for l in range(k):
                        list_dih.append([i,j,k,l])
        #print(list_dih)
        dihedrals.get_dih_pop(molecule.coords, list_dih)
        #print(np.int_(dihedrals.phis.T)) 
        #print(dihedrals.phis.shape, dihedrals.phis.T.shape)

        u, indices = np.unique(np.int_(dihedrals.phis), axis=0, 
                return_index=True)
        #print(u)
        #print(indices)
        train_phis = np.unique(indices)
        print(train_phis)

        test_phis = []
        for i in range(dihedrals.phis.shape[0]):
            if i not in train_phis:
                test_phis.append(i)
            else:
                continue

        print(test_phis)
        train_phis = np.array(train_phis)
        test_phis = np.array(test_phis)
        print('train_phis', train_phis.shape, 'test_phis', test_phis.shape)

        train_NRF = np.take(molecule.mat_NRF, train_phis, axis=0)
        #_NRF = np.take(molecule.mat_NRF, test_phis, axis=0)
        print(train_NRF.shape)

        #sys.exit()


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
        '''

    sys.stdout.flush()

    run_net = False
    split = 5 #2 4 5 20 52 260
    train = round(len(molecule.coords) / split, 3)
    nodes = 1000
    input = molecule.mat_NRF
    output = molecule.mat_F
    if run_net:
        print('\nget train and test sets, '\
                'training set is {} points.'\
                '\nNumber of nodes is {}'.format(train, nodes))
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        #override train test
        #molecule.train = train_phis
        #molecule.test = test_phis
        print('\nget ANN model')
        network = Network() #initiate network class
        Network.get_variable_depth_model(network, molecule, 
                nodes, input, output) #train NN

        run_mm = True
        if run_mm:
            print('\nrun MM with ANN potential')
            nsteps=500 #10000
            mm = Network.run_NVE(network, molecule, timestep=0.5, 
                    nsteps=nsteps)

            print('checking mm forces')
            mm.coords = mm.get_3D_array([mm.coords])
            mm.forces = mm.get_3D_array([mm.forces]) 
            mm.energies = np.array(mm.energies)
            unconserved = mm.check_force_conservation()
    #sys.exit()
    sys.stdout.flush()


    check_E_conservation = True
    if check_E_conservation:

        #mat_NRF = Network.get_NRF_input([molecule.coords[0]], molecule.atoms, 
               #len(molecule.atoms), len(molecule.mat_NRF[0]))
        #print(mat_NRF.shape)
        #print(molecule.mat_F[0])

        '''
        scale_NRF = 9650.66147293977
        scale_F = 131.25482358398773
        q_scaled = Conservation.get_conservation(molecule.coords[0], 
                molecule.forces[0], molecule.atoms, scale_NRF, scale_F, 
                'best_ever_model', molecule, 0.001)
        #print(q_scaled)
        '''

        run_mm = True
        if run_mm:
            print('\nrun MM with ANN potential')
            nsteps=500
            network = None
            mm = Network.run_NVE(network, molecule, timestep=0.5, 
                    nsteps=nsteps)

            print('checking mm forces')
            mm.coords = mm.get_3D_array([mm.coords])
            mm.forces = mm.get_3D_array([mm.forces]) 
            mm.energies = np.array(mm.energies)
            unconserved = mm.check_force_conservation()


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
    #unconserved = Molecule.check_force_conservation(molecule)
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
                ['train', 'nn-mm'], ['k', 'r'], 
                '$\phi_1$', '$\phi_2$', 'train-xy-12.png')
        Plotter.xy_scatter([np.take(phis_1, test)], [np.take(phis_2, test)], 
                [''], ['k'], '$\phi_1$', '$\phi_2$', 'test-xy-12.png')
        Plotter.xy_scatter([np.take(phis_1, train), mm_phis_1], 
                [np.take(phis_3, train), mm_phis_3], 
                ['train', 'nn-mm'], ['k', 'r'], 
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

    mm_qm_comparison = False
    if mm_qm_comparison:
        print('read and check mm forces')
        mm_molecule = Molecule()
        XYZParser(atom_file, [coord_files[0]], [force_files[0]], 
                [energy_files[0]], mm_molecule)
        print(datetime.now() - startTime)
        Converter.get_rotated_forces(mm_molecule)
        mm_molecule.forces = mm_molecule.rotated_forces
        mm_molecule.coords = mm_molecule.rotated_coords
        mm_unconserved = Molecule.check_force_conservation(mm_molecule) #
        print(datetime.now() - startTime)
        sys.stdout.flush()

        print('read and check qm forces')
        qm_molecule = Molecule()
        XYZParser(atom_file, [coord_files[1]], [force_files[1]], 
                [energy_files[1]], qm_molecule)
        print(datetime.now() - startTime)
        Converter.get_rotated_forces(qm_molecule)
        qm_molecule.forces = qm_molecule.rotated_forces
        qm_molecule.coords = qm_molecule.rotated_coords
        qm_unconserved = Molecule.check_force_conservation(qm_molecule) #
        print(datetime.now() - startTime)
        sys.stdout.flush()

        print('remove unconserved')
        unconserved = np.concatenate([mm_unconserved, qm_unconserved])
        if len(unconserved) > 0:
            unconserved = np.unique(unconserved)
            mm_molecule.remove_variants(unconserved)
            qm_molecule.remove_variants(unconserved)
        print('unconserved', unconserved)
        print('get mm decomp forces')
        Converter(mm_molecule) #get pairwise forces
        print(datetime.now() - startTime)
        sys.stdout.flush()
        print('get qm decomp forces')
        Converter(qm_molecule) #get pairwise forces
        print(datetime.now() - startTime)
        sys.stdout.flush()

        print('compare mm and qm decomp forces')
        print(mm_molecule.coords.shape, qm_molecule.coords.shape)
        print(range(len(mm_molecule.atoms)))
        print(mm_molecule.atoms)
        _N = -1
        for i in range(len(mm_molecule.atoms)):
            for j in range(i):
                _N += 1
                print(_N, i, j, '-', mm_molecule.atoms[i], 
                        mm_molecule.atoms[j])
                Plotter.xy_scatter([mm_molecule.mat_F.T[_N]], 
                        [qm_molecule.mat_F.T[_N]], 
                        ['mm vs qm'], ['k'], 
                        'mm decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'qm decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'mm_vs_qm_decomp_forces_{}_{}{}'.format(_N, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
        print(datetime.now() - startTime)
        sys.stdout.flush()

    '''
    print('compare mm and qm energies')
    Plotter.xy_scatter([mm_molecule.energies], [qm_molecule.energies], 
            ['mm vs qm'], ['k'], 
            'mm Etot / kcal mol$^{-1}$', 'qm Etot / kcal mol$^{-1}$', 
            'mm_vs_qm_energies')

    print('check mm and qm dihedrals are the same')
    mm_dihedrals = Binner()
    qm_dihedrals = Binner()
    list_dih = [[1, 2, 3, 6], [3, 2, 1, 7],
            [2, 1, 7, 10]]
    mm_dihedrals.get_dih_pop(mm_molecule.coords, list_dih)
    qm_dihedrals.get_dih_pop(mm_molecule.coords, list_dih)

    Plotter.xy_scatter([mm_dihedrals.phis.T[1]], [mm_dihedrals.phis.T[1]], 
            ['mm vs qm'], ['k'], 
            'mm OCCO dih', 'qm OCCO dih', 
            'mm_vs_qm_dihs')
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


