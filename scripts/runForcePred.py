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
from keras.models import Model, load_model                                   

from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, AMBLAMMPSParser, AMBERParser, XYZParser, Binner, \
        Writer, Plotter, Network, Conservation

openmm = False
if openmm:
    from ForcePred.calculate.OpenMM import OpenMM

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
    AMBERParser('molecules.prmtop', coord_files, force_files, 
            molecule)
    '''

    '''
    XYZParser(atom_file, coord_files, force_files, 
            energy_files, molecule)
    '''


    NPParser(atom_file, coord_files, force_files, [], molecule)

    '''
    Writer.write_xyz(molecule.coords, molecule.atoms, 'ensemble-coords.xyz', 'w')
    Writer.write_xyz(molecule.forces, molecule.atoms, 'ensemble-forces.xyz', 'w')
    np.savetxt('ensemble-energies.txt', molecule.energies)
    np.savetxt('ensemble-charges.txt', molecule.charges)
    sys.exit()
    '''

    '''
    Writer.write_gaus_cart(molecule.coords[0:11550:10], molecule.atoms, 
            'FORCE POP=MK INTEGRAL=(GRID=ULTRAFINE)', 'mm-SPoints')

    sys.stdout.flush()
    sys.exit()
    '''


    '''
    Writer.write_amber_inpcrd(molecule.coords[0], 'molecules1.inpcrd')

    sys.stdout.flush()
    sys.exit()
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

    '''

    def iterate_xyz_files(filename, var):
        for filename in filename:
            input_ = open(filename, 'r')
            for line in input_:
                xyz = clean(extract(1, input_))
                var.append(xyz)
        var = get_3D_array(var, 10)
        return var

    def extract(padding, input_):
        return (list(islice(input_, padding + 
                10))[-10:])

    def clean(raw):
        cleaned = np.empty(shape=[10, 3])
        for i, atom in enumerate(raw):
            cleaned[i] = atom.strip('\n').split()[-3:]
        #print(cleaned.shape)
        return np.array(cleaned)

    def get_3D_array(np_list, n_atoms):
        return np.reshape(np.vstack(np_list), (-1,n_atoms,3))
    velocities = []
    velocities = iterate_xyz_files(['nn-velocities.xyz'], velocities)
    np.savetxt('nn-velocities.txt', velocities.reshape(-1,30))
    sys.exit()
    '''

    #A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    #print(molecule.atoms)
    #print(A)

    '''
    open('decomp_E.txt', 'w')             
    open('molecular_E.txt', 'w')             
    f1 = open('decomp_E.txt', 'ab') 
    f2 = open('molecular_E.txt', 'ab') 
    n_atoms = len(molecule.atoms)
    _NC2 = int(n_atoms * (n_atoms-1)/2)
    for i in range(len(molecule.coords)):
        decomp_E = Converter.get_energy_decomposition(molecule.atoms, 
                molecule.coords[i], molecule.energies[i])
        _E = molecule.energies[i]
        print(_E)
        print(decomp_E)
        np.savetxt(f1, decomp_E.reshape(1,-1))
        np.savetxt(f2, _E)
        sum_decomp = np.sum(decomp_E)
        print(sum_decomp, sum_decomp/_E)
        recomp = Converter.get_recomposed_energy(molecule.coords[i], decomp_E, 
                n_atoms, _NC2)
        print(recomp)
        print()
    '''

    get_decomp = True
    if get_decomp:
        print(molecule)
        print('\ncheck and get force decomp')
        unconserved = Molecule.check_force_conservation(molecule) #
        Converter(molecule) #get pairwise forces
        print(datetime.now() - startTime)

    '''
    #aspirin
    scale_NRF = 13036.56150157717
    scale_NRF_min = 4.296393449372179 
    scale_F = 114.0951589972158
    scale_F_min = 1.1633573357150429e-06
    '''

    #'''
    #malonaldehyde 
    scale_NRF = 12364.344210692281
    scale_NRF_min = 15.377554794644805
    scale_F = 102.58035961189837
    scale_F_min = 2.4191760310543486e-06
    #'''


    '''
    #ethanediol    
    scale_NRF = 19911.051275945494
    scale_NRF_min = 11.648288117257646
    scale_F = 105.72339728320982 
    scale_F_min = 0
    '''

    check_model = False
    if check_model:
        print('check model error')
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        Converter(molecule) #get pairwise forces
        scaled_input, in_max, in_min = Network.get_scaled_values(
                molecule.mat_NRF, 
                scale_NRF, scale_NRF_min, method='A')
        scaled_output, _max, _min = Network.get_scaled_values(molecule.mat_F, 
                scale_F, scale_F_min, method='B')
        print('\nNRF_min: {} NRF_max: {}\nF_min: {} F_max: {}\n'.format(
            np.amin(molecule.mat_NRF), np.amax(molecule.mat_NRF),
            np.amin(np.absolute(molecule.mat_F)), 
            np.amax(np.absolute(molecule.mat_F))))
        model = load_model('best_ever_model')
        train_prediction_scaled = model.predict(scaled_input)
        train_prediction = Network.get_unscaled_values(
                train_prediction_scaled, 
                scale_F, scale_F_min, method='B')
        train_recomp_forces = Network.get_recomposed_forces(
                molecule.coords, train_prediction, n_atoms, _NC2)
        print(molecule.mat_NRF.shape, train_prediction.shape, 
                train_recomp_forces.shape)
        train_mae, train_rms = Binner.get_error(molecule.mat_F.flatten(), 
                    train_prediction.flatten())
        print('\nTrain MAE: {} \nTrain RMS: {}'.format(train_mae, train_rms))
        sys.stdout.flush()

    decomp_F = True
    if decomp_F:
        n_atoms = len(molecule.atoms)
        NRF_scale_method = 'A'
        model = load_model('best_ever_model')
        dr = 0.01
        open('scaled-forces.txt', 'w')
        open('predicted-forces.txt', 'w')
        open('scaled-qs.txt', 'w')
        open('predicted-qs.txt', 'w')
        open('separate-qs.txt', 'w')
        open('separate-scale-factors.txt', 'w')
        f2 = open('scaled-forces.txt', 'a')
        f3 = open('predicted-forces.txt', 'a')
        f4 = open('scaled-qs.txt', 'a')
        f5 = open('predicted-qs.txt', 'a')
        f6 = open('separate-qs.txt', 'a')
        f7 = open('separate-scale-factors.txt', 'a')
        n_structures = 1 #len(molecule.coords)
        for i in range(n_structures):
            print(i)
            pred_F, scaled_F, pred_q, scaled_q, scaling_factors = \
                    Conservation.get_conservation(i,
                    molecule.coords[i], molecule.forces[i], 
                    molecule.atoms, scale_NRF, 0, scale_F, 
                    model, molecule, dr, NRF_scale_method)
            scaled_F = scaled_F.reshape(n_atoms,-1)
            pred_F = pred_F.reshape(n_atoms,-1)
            #print(scaled_F)
            #print(pred_F)
            #print()
            np.savetxt(f2, scaled_F)
            np.savetxt(f3, pred_F)
            np.savetxt(f4, scaled_q)
            np.savetxt(f5, pred_q)
            sys.stdout.flush()

            displaced_of_displaced = True
            if displaced_of_displaced:
                displaced_coords = np.loadtxt(
                        '1_coords.txt').reshape(-1,n_atoms,3)
                print(displaced_coords.shape)
                for j in range(len(displaced_coords)):
                    print('j', j)
                    pred_F, scaled_F, pred_q, scaled_q, scaling_factors = \
                            Conservation.get_conservation(j,
                            displaced_coords[j], None, 
                            molecule.atoms, scale_NRF, 0, scale_F, 
                            model, molecule, dr, NRF_scale_method)
                    pred_q = pred_q.reshape(1,-1)
                    scaling_factors = scaling_factors.reshape(1,-1)
                    np.savetxt(f6, pred_q)
                    np.savetxt(f7, scaling_factors)
                    print('check q', pred_q)
                    print('check SF', scaling_factors)
                    print()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()
        f7.close()
        pred_F = np.loadtxt('predicted-forces.txt')
        scaled_F = np.loadtxt('scaled-forces.txt')
        pred_q = np.loadtxt('predicted-qs.txt')
        scaled_q = np.loadtxt('scaled-qs.txt')
        print(pred_F.shape)
        pred_mae, pred_rms = Binner.get_error(
                molecule.forces[:n_structures].flatten(), pred_F.flatten())
        scaled_mae, scaled_rms = Binner.get_error(
                molecule.forces[:n_structures].flatten(), scaled_F.flatten())
        print('F prediction MAE: {} RMS: {}'.format(pred_mae, pred_rms))
        print('F scaled MAE: {} RMS: {}'.format(scaled_mae, scaled_rms))

        pred_mae, pred_rms = Binner.get_error(
                molecule.mat_F[:n_structures].flatten(), pred_q.flatten())
        scaled_mae, scaled_rms = Binner.get_error(
                molecule.mat_F[:n_structures].flatten(), scaled_q.flatten())
        print('q prediction MAE: {} RMS: {}'.format(pred_mae, pred_rms))
        print('q scaled MAE: {} RMS: {}'.format(scaled_mae, scaled_rms))

        each_pred_mae, each_pred_rms = Binner.get_each_error(
                molecule.mat_F[:n_structures].flatten(), pred_q.flatten())

        each_pred_mae_F, each_pred_rms_F = Binner.get_each_error(
                molecule.forces[:n_structures].flatten(), pred_F.flatten())

        pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                pairs.append([i, j])

        interatomic_measures = Binner()
        interatomic_measures.get_bond_pop(molecule.coords, pairs)

        Plotter.xy_scatter(
                [interatomic_measures.rs[:n_structures].T.flatten()], 
                [each_pred_rms], [''], ['k'], 'r', 'rms', 2, 'rms_r-q.png')
        Plotter.xy_scatter(
                [interatomic_measures.rs[:n_structures].T.flatten()], 
                [pred_q.flatten()], [''], ['k'], 'r', 'qAB', 2, 'q_r.png')
        n_list = list(range(n_atoms))
        n_list = np.array([n_list * n_structures * 3])
        Plotter.xy_scatter([n_list], 
                [each_pred_rms_F], [''], ['k'], 'atom index', 
                'rms', 2, 'rms_N-F.png')

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
    if run_net:
        train = round(len(molecule.coords) / split, 3)
        nodes = 1000
        input = molecule.mat_NRF
        output = molecule.mat_F
        print('\nget train and test sets, '\
                'training set is {} points.'\
                '\nNumber of nodes is {}'.format(train, nodes))
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        #override train test
        #molecule.train = train_phis
        #molecule.test = test_phis
        print('\nget ANN model')
        network = Network(molecule) #initiate network class
        Network.get_variable_depth_model(network, molecule, 
                nodes, input, output) #train NN

        run_mm = False
        if run_mm:
            nsteps=15000 #10000
            print('\nrun {} step MD with ANN potential'.format(nsteps))
            mm = Network.run_NVE(network, molecule, timestep=0.5, 
                    nsteps=nsteps)

            print('checking mm forces')
            mm.coords = mm.get_3D_array([mm.coords])
            mm.forces = mm.get_3D_array([mm.forces]) 
            mm.energies = np.array(mm.energies)
            unconserved = mm.check_force_conservation()
    #sys.exit()
    sys.stdout.flush()


    check_E_conservation = False
    if check_E_conservation:

        #mat_NRF = Network.get_NRF_input([molecule.coords[0]], molecule.atoms, 
               #len(molecule.atoms), len(molecule.mat_NRF[0]))
        #print(mat_NRF.shape)
        #print(molecule.mat_F[0])

        '''
        scale_NRF = 9650.66147293977
        scale_F = 131.25482358398773
        q_scaled = Conservation.get_conservation(molecule.coords[0], 
                molecule.forces[0], molecule.atoms, scale_NRF, 0, scale_F, 
                'best_ever_model', molecule, 0.001)
        print(q_scaled)
        '''




    run_mm = False
    if run_mm:
        open('vary_qAB.txt', 'w').close()
        open('ave_delta_F.txt', 'w').close()
        qAB_factor = 1 #0.95
        #print(molecule.forces[0])
        for x in range(1):
            nsteps = 15000 #30000
            print('\nrun {} step MD with ANN potential'.format(nsteps))
            network = None
            mm = Network.run_NVE(network, molecule, timestep=0.5, 
                    nsteps=nsteps, qAB_factor=qAB_factor)

            print('checking mm forces')
            mm.coords = mm.get_3D_array([mm.coords])
            mm.forces = mm.get_3D_array([mm.forces]) 
            mm.energies = np.array(mm.energies)
            unconserved = mm.check_force_conservation()
            qAB_factor += 2



    correct_training = False
    if correct_training:
        #'''
        dr = 0.001
        NRF_scale_method = 'A'
        print('read and check training forces')
        molecule = Molecule()
        XYZParser(atom_file, coord_files, force_files, 
                energy_files, molecule)

        '''
        ###pick only some structures, hard-coded
        molecule.coords = molecule.coords[::50]
        molecule.forces = molecule.forces[::50]
        molecule.energies = molecule.energies[::50]
        '''


        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)


        print(datetime.now() - startTime)
        unconserved = Molecule.check_force_conservation(molecule) #
        if len(unconserved) > 0:
            molecule.remove_variants(unconserved)
        print(datetime.now() - startTime)
        sys.stdout.flush()
        split = 5 #2 4 5 20 52 260
        train = round(len(molecule.coords) / split, 3)
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        print('get training decomp forces')
        Converter(molecule) #get pairwise forces
        #scale_NRF = 17495.630534482527
        #scale_F = 214.93048169383425
        train_mat_NRF = np.take(molecule.mat_NRF, molecule.train, axis=0)
        train_mat_F = np.take(molecule.mat_F, molecule.train, axis=0)
        train_F = np.take(molecule.forces, molecule.train, axis=0)
        train_coords = np.take(molecule.coords, molecule.train, axis=0)

        scale_input_max = 19911.051275945494 #np.amax(train_mat_NRF)
        scale_input_min = 11.648288117257646 #np.amin(train_mat_NRF)
        scale_output_max = 105.72339728320982 #np.amax(np.absolute(train_mat_F))
        scale_output_min = 0 #np.amin(np.absolute(train_mat_F))

        scaled_input, _max, _min = \
                Network.get_scaled_values(train_mat_NRF, 
                scale_input_max, scale_input_min, method='A')
        scaled_output, _max, _min = \
                Network.get_scaled_values(train_mat_F, 
                scale_output_max, scale_output_min, method='B')

        model = load_model('../best_ever_model')
        train_prediction_scaled = model.predict(scaled_input)
        #train_prediction = (train_prediction_scaled - 0.5) * \
                #scale_output_max
        train_prediction = Network.get_unscaled_values(
                train_prediction_scaled, 
                scale_output_max, scale_output_min, method='B')
        train_recomp_forces = Network.get_recomposed_forces(
                train_coords, train_prediction, n_atoms, _NC2)
        print(train_mat_NRF.shape, train_prediction.shape, 
                train_recomp_forces.shape)

        train_mae, train_rms = Binner.get_error(train_mat_F.flatten(), 
                    train_prediction.flatten())
        print('\nTrain MAE: {} \nTrain RMS: {}'.format(train_mae, train_rms))
        print(datetime.now() - startTime)

        '''
        corrected_decomp_list = []
        corrected_recomp_list = []
        for i in range(len(train_coords)):
            corrected_prediction = Conservation.get_conservation(
                    train_coords[i], [train_prediction[i]], 
                    molecule.atoms, scale_input_max, scale_input_min, 
                    scale_output_max, '../best_ever_model', 
                    molecule, dr, NRF_scale_method)
            corrected_decomp_list.append(corrected_prediction)
            corr_recomp_forces = Network.get_recomposed_forces(
                    [train_coords[i]], [corrected_prediction], n_atoms, _NC2)
            corrected_recomp_list.append(corr_recomp_forces)
        corrected_decomp_list = np.array(corrected_decomp_list)
        corrected_recomp_list = np.array(corrected_recomp_list)

        corrected_mae, corrected_rms = Binner.get_error(train_mat_F.flatten(),
                corrected_decomp_list.flatten())
        print('\nCorrected MAE: {} \nCorrected RMS: {}'.format(
            corrected_mae, corrected_rms))
        '''


        _L1s = []
        corrected_L1s = []
        for i in range(_NC2):
            bin_edges, hist = Binner.get_scurve(train_mat_F.T[i].flatten(), 
                    train_prediction.T[i].flatten(), 
                    'scurve.txt')

            #bin_edges2, hist2 = Binner.get_scurve(train_mat_F.T[i].flatten(),
                    #corrected_decomp_list.T[i].flatten(), 
                    #'scurve.txt')
            print(i, bin_edges[19], hist[19])
            #print(i, bin_edges2[19], hist2[19])
            #print()
            _L1s.append(hist[19])
            #corrected_L1s.append(hist2[19])

        #Plotter.plot_2d([range(_NC2), range(_NC2)], 
                #[_L1s, corrected_L1s], ['L1', 'L1 with correction'], 
                #'Interaction number', 'L1', 'L1.png')

        Plotter.plot_2d([range(_NC2)], 
                [_L1s], ['L1'], 
                'Interaction number', 'L1', 'L1.png')


        #print()



        '''
        for i in range(len(train_coords)):
            print(train_mat_F[i])
            print()
            print(corrected_decomp_list[i])
            print()
            print(train_F[i])
            print()
            print(train_recomp_forces[i])
            print()
            print(corrected_recomp_list[i])
            print('\n\n')
        '''
        sys.stdout.flush()
        #'''



    #'''
    print('\nAtom pairs:')
    print('i indexA indexB - atomA atomB')
    _N = -1
    for i in range(len(molecule.atoms)):
        for j in range(i):
            _N += 1
            print(_N, i, j, '-', Converter._ZSymbol[molecule.atoms[i]], 
                    Converter._ZSymbol[molecule.atoms[j]])
            #print(_N+1, '-', atom_names[i], atom_names[j])
    #sys.exit()
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
        n_atoms = len(mm_molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        print(datetime.now() - startTime)
        #Converter.get_rotated_forces(mm_molecule)
        #mm_molecule.forces = mm_molecule.rotated_forces
        #mm_molecule.coords = mm_molecule.rotated_coords
        mm_unconserved = Molecule.check_force_conservation(mm_molecule) #
        print(datetime.now() - startTime)
        sys.stdout.flush()

        print('read and check qm forces')
        qm_molecule = Molecule()
        #XYZParser(atom_file, [coord_files[1]], [force_files[1]], 
                #[energy_files[1]], qm_molecule)
        OPTParser(input_files, qm_molecule, opt=False) #read in FCEZ for SP
        print(datetime.now() - startTime)
        qm_unconserved = Molecule.check_force_conservation(qm_molecule) #
        print(datetime.now() - startTime)
        sys.stdout.flush()

        #'''
        print('read and check training forces')
        molecule = Molecule()
        XYZParser(atom_file, [coord_files[1]], [force_files[1]], 
                [energy_files[1]], molecule)

        print(datetime.now() - startTime)
        unconserved = Molecule.check_force_conservation(molecule) #
        if len(unconserved) > 0:
            molecule.remove_variants(unconserved)
        print(datetime.now() - startTime)
        sys.stdout.flush()
        split = 5 #2 4 5 20 52 260
        train = round(len(molecule.coords) / split, 3)
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        print('get training decomp forces')
        Converter(molecule) #get pairwise forces
        #scale_NRF = 17495.630534482527
        #scale_F = 214.93048169383425
        train_mat_NRF = np.take(molecule.mat_NRF, molecule.train, axis=0)
        train_mat_F = np.take(molecule.mat_F, molecule.train, axis=0)
        train_F = np.take(molecule.forces, molecule.train, axis=0)
        train_coords = np.take(molecule.coords, molecule.train, axis=0)
        scaled_input, scale_input_max, scale_input_min = \
                Network.get_scaled_values(train_mat_NRF)
        scaled_output, scale_output_max, scale_output_min = \
                Network.get_scaled_values(train_mat_F)
        model = load_model('../best_ever_model')
        train_prediction_scaled = model.predict(scaled_input)
        train_prediction = (train_prediction_scaled - 0.5) * \
                scale_output_max
        train_recomp_forces = Network.get_recomposed_forces(
                train_coords, train_prediction, n_atoms, _NC2)
        print(train_mat_NRF.shape, train_prediction.shape, 
                train_recomp_forces.shape)
        print(datetime.now() - startTime)
        sys.stdout.flush()
        #'''

        print('read and check adjusted mm forces')
        fix_molecule = Molecule()
        XYZParser(atom_file, [coord_files[2]], [force_files[2]], 
                [energy_files[2]], fix_molecule)
        print(datetime.now() - startTime)
        fix_unconserved = Molecule.check_force_conservation(fix_molecule) #
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

        #train_errors = np.reshape(np.loadtxt('train_errors.txt'), (-1,2))
        ts = range(len(mm_molecule.coords))

        #for pairwise analysis
        _N = -1
        for i in range(len(mm_molecule.atoms)):
            for j in range(i):
                _N += 1
                #print(_N, i, j, '-', mm_molecule.atoms[i], 
                        #mm_molecule.atoms[j])

                '''
                #train_decompF = train_prediction.T[_N]
                train_mae, train_rms = Binner.get_error(
                        [train_prediction.T[_N]], 
                        [train_mat_F.T[_N]])
                '''
                #train_mae, train_rms = train_errors[_N][0], train_errors[_N][1]
                #print('\t', train_mae, train_rms)

                '''
                #train_min, train_max = np.amin(train_mat_NRF.T[_N]), \
                        #np.amax(train_mat_NRF.T[_N])
                train_min, train_max = np.amin(train_mat_F.T[_N]), \
                        np.amax(train_mat_F.T[_N])
                print(train_min, train_max)
                '''

                mm_mae, mm_rms = Binner.get_error(
                        [mm_molecule.mat_F.T[_N]], 
                        [qm_molecule.mat_F.T[_N]])
                #print('\t', mm_mae, mm_rms)

                mae_list, rms_list = [], []
                for t in ts:
                    mae, rms = Binner.get_error(
                            [mm_molecule.mat_F.T[_N][t]], 
                            [qm_molecule.mat_F.T[_N][t]])
                    mae_list.append(mae)
                    rms_list.append(rms)

                '''
                Plotter.xy_scatter([mm_molecule.mat_F.T[_N]], 
                        [qm_molecule.mat_F.T[_N]], 
                        ['mm vs qm'], ['k'], 
                        'mm decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'qm decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'mm_vs_qm_decomp_forces_{}_{}{}'.format(_N, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                '''

                '''
                Plotter.xy_scatter([ts, ts],
                        [mm_molecule.mat_F.T[_N], qm_molecule.mat_F.T[_N]], 
                        ['mm', 'qm'], ['k', 'b'], 
                        'timestep', 
                        'decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'decomp_forces_vs_ts_{}_{}{}_{}{}'.format(_N, i, j, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                '''

                '''
                Plotter.xy_scatter([ts, ts, ts, ts],
                        [mae_list, rms_list, [train_mae] * len(ts), 
                        [train_rms] * len(ts)], 
                        ['mae', 'rms', 'train_mae', 'train_rms'], 
                        ['k', 'b', 'r', 'm'], 
                        'timestep', 
                        'error', 
                        'errors_vs_ts_{}_{}{}_{}{}'.format(_N, i, j,
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                '''

                '''
                Plotter.xy_scatter([mm_molecule.mat_NRF.T[_N], 
                        qm_molecule.mat_NRF.T[_N]],
                        [mm_molecule.mat_F.T[_N], qm_molecule.mat_F.T[_N]], 
                        ['ml-md', 'qm'], 
                        ['k', 'b'], 
                        'NRF', 
                        'decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'decomp_forces_vs_NRF_{}_{}{}_{}{}'.format(_N, i, j, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                '''

                '''
                Plotter.xy_scatter([mm_molecule.mat_NRF.T[_N], 
                        qm_molecule.mat_NRF.T[_N], train_mat_NRF.T[_N], 
                        train_mat_NRF.T[_N]],
                        [mm_molecule.mat_F.T[_N], qm_molecule.mat_F.T[_N], 
                        train_prediction.T[_N], train_mat_F.T[_N]], 
                        ['mm', 'qm', 'nn-train', 'qm-train'], 
                        ['k', 'b', 'm', 'r'], 
                        'NRF', 
                        'decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'decomp_forces_vs_NRF_{}_{}{}_{}{}'.format(_N, i, j, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                '''

                #'''
                Plotter.xy_scatter([train_mat_NRF.T[_N],
                        mm_molecule.mat_NRF.T[_N][:1000], 
                        mm_molecule.mat_NRF.T[_N][1000:]],
                        [train_prediction.T[_N], 
                        mm_molecule.mat_F.T[_N][:1000], 
                        mm_molecule.mat_F.T[_N][1000:]], 
                        ['train', 'first 1000 frames', 'last 250 frames'], 
                        ['grey', 'k', 'b'], 
                        'NRF', 
                        'decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'frames_decomp_forces_vs_NRF_{}_{}{}_{}{}'.format(
                        _N, i, j, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                #'''


                '''
                Plotter.xy_scatter([ts, ts, ts], [mm_molecule.mat_NRF.T[_N], 
                        [train_min] * len(ts), [train_max] * len(ts)], 
                        ['mm', 'train_min', 'train_max'], 
                        ['k', 'r', 'm'], 
                        'timestep', 
                        'NRF', 
                        'NRF_vs_ts_{}_{}{}_{}{}'.format(_N, i, j, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                '''


                '''
                Plotter.xy_scatter([ts, ts, ts], [mm_molecule.mat_F.T[_N], 
                        [train_min] * len(ts), [train_max] * len(ts)], 
                        ['mm', 'train_min', 'train_max'], 
                        ['k', 'r', 'm'], 
                        'timestep', 
                        'decompF / kcal mol$^{-1}\ \AA^{-1}$', 
                        'decomp_forces_vs_ts_{}_{}{}_{}{}'.format(_N, i, j, 
                        mm_molecule.atoms[i], mm_molecule.atoms[j]))
                '''

        #for cartesian analysis
        _N = -1
        for i in range(n_atoms):
            for x in range(3):
                _N += 1

                '''
                train_mae, train_rms = Binner.get_error(
                        [train_recomp_forces.reshape(
                            len(train_mat_NRF),-1).T[_N]], 
                        [train_F.reshape(
                            len(train_mat_NRF),-1).T[_N]])
                #train_mae, train_rms = train_errors[_N][0], train_errors[_N][1]
                #print('\t', train_mae, train_rms)

                #train_min, train_max = np.amin(train_mat_NRF.T[_N]), \
                        #np.amax(train_mat_NRF.T[_N])
                train_min, train_max = np.amin(train_recomp_forces.reshape(
                            len(train_mat_NRF),-1).T[_N]), \
                        np.amax(train_recomp_forces.reshape(
                            len(train_mat_NRF),-1).T[_N])
                #print(train_min, train_max)
                '''

                mae_list, rms_list = [], []
                for t in ts:
                    mae, rms = Binner.get_error(
                            [mm_molecule.forces.reshape(len(ts),-1).T[_N][t]], 
                            [qm_molecule.forces.reshape(len(ts),-1).T[_N][t]])
                    mae_list.append(mae)
                    rms_list.append(rms)


                '''
                Plotter.xy_scatter([ts, ts, ts, ts], 
                        [mae_list, rms_list, [train_mae] * len(ts), 
                        [train_rms] * len(ts)], 
                        ['mae', 'rms', 'train_mae', 'train_rms'], 
                        ['k', 'b', 'r', 'm'], 
                        'timestep', 
                        'F / kcal mol$^{-1}\ \AA^{-1}$', 
                        'Ferrors_vs_ts_{}_{}_{}'.format(_N, i, 
                        mm_molecule.atoms[i]))
                '''

                '''
                Plotter.xy_scatter([ts, ts, ts, ts], 
                        [mm_molecule.forces.reshape(len(ts),-1).T[_N], 
                        qm_molecule.forces.reshape(len(ts),-1).T[_N],
                        [train_min] * len(ts), [train_max] * len(ts)], 
                        ['mm', 'qm', 'train_min', 'train_max'], 
                        ['k', 'b', 'r', 'm'], 
                        'timestep', 
                        'F / kcal mol$^{-1}\ \AA^{-1}$', 
                        'forces_vs_ts_{}_{}_{}'.format(_N, i, 
                        mm_molecule.atoms[i]))
                '''


                '''
                Plotter.xy_scatter([ts, ts], 
                        [mm_molecule.forces.reshape(len(ts),-1).T[_N], 
                        qm_molecule.forces.reshape(len(ts),-1).T[_N]], 
                        ['mm', 'qm'], 
                        ['k', 'b'], 
                        'timestep', 
                        'F / kcal mol$^{-1}\ \AA^{-1}$', 
                        'forces_vs_ts_{}_{}_{}'.format(_N, i, 
                        mm_molecule.atoms[i]))
                '''


        '''
        force_diff = (fix_molecule.forces.T[0][0] / 
                mm_molecule.forces[:len(fix_molecule.forces)].T[0][0])
        Plotter.xy_scatter([range(len(force_diff.flatten()))], 
                [force_diff.flatten()], 
                ['scaling factor'], 
                ['k'], 
                'timestep', 
                'scale factor / kcal mol$^{-1}\ \AA^{-1}$', 
                'SF_vs_ts_{}_{}_{}'.format(_N, i, 
                mm_molecule.atoms[i]))
        '''



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


