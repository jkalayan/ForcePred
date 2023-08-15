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
        Permuter, XYZParser, Binner, Writer, Plotter, Conservation
        #Network
from ForcePred.nn.Network_shared_ws import Network
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
        qFE_loss_w='qFE_loss_w', E_loss_w='E_loss_w', bias='bias',
        filtered='filtered', load_model='load_model', dihedrals='dihedrals'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class
    prescale = [0, 1, 0, 1, 0, 0]
    #read in input data files
    if list_files:
        input_files = open(list_files).read().split()
    mol=False
    if atom_file and mol:
        #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
        NPParser(atom_file, coord_files, force_files, energy_files, molecule)
        print(datetime.now() - startTime)
        sys.stdout.flush()

        print(molecule)
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        sys.stdout.flush()

        # write pdb file for first frame
        Writer.write_pdb(molecule.coords[0], 'MOL', 1, molecule.atoms, 
                'molecule.pdb', 'w')

        #n_training = 10000
        #n_val = 50
        if n_test == -1 or n_test > len(molecule.coords):
            n_test = len(molecule.coords) - n_training
        print('\nn_nodes', n_nodes, '\nn_layers:', n_layers,
                '\nn_training', n_training, '\nn_val', n_val,
                '\nn_test', n_test, '\ngrad_loss_w', grad_loss_w,
                '\nqFE_loss_w', qFE_loss_w, '\nE_loss_w', E_loss_w,
                '\nbias', bias)


        ### Scale energies using min/max forces in training set

        def prescale_energies(molecule, train_idx):
            train_forces = np.take(molecule.forces, train_idx, axis=0)
            train_energies = np.take(molecule.orig_energies, train_idx, axis=0)
            min_e = np.min(train_energies)
            max_e = np.max(train_energies)
            #min_f = np.min(train_forces)
            min_f = np.min(np.abs(train_forces))
            #max_f = np.max(train_forces)
            max_f = np.max(np.abs(train_forces))
            molecule.energies = ((max_f - min_f) * 
                    (molecule.orig_energies - min_e) / 
                    (max_e - min_e) + min_f)
            prescale[0] = min_e
            prescale[1] = max_e
            prescale[2] = min_f
            prescale[3] = max_f
            return prescale

        # save original energies first, before scaling by forces
        molecule.orig_energies = np.copy(molecule.energies)

        #bias = '1/r' #default for now
        extra_cols = 0 #n_atoms
        pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                pairs.append([i, j])
        if filtered:
            print('\nprescale energies so that magnitude is comparable to forces')

            prescale = prescale_energies(molecule, list(range(len(molecule.coords))))
            print('E SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                    np.max(molecule.energies)))
            print('prescale value:', prescale)
            sys.stdout.flush()

            print('\nget decomposed forces and energies simultaneously '\
                    'with energy bias: {}'.format(bias))

            Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                    bias_type=bias, extra_cols=extra_cols)



            '''
            atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                    zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
            atom_pairs = []
            for i in range(len(molecule.atoms)):
                for j in range(i):
                    atom_pairs.append('{}_{}'.format(atom_names[i], 
                        atom_names[j]))
            qq_root = []
            for i in range(n_atoms):
                for j in range(i):
                    v = (molecule.atoms[i] * molecule.atoms[j]) ** 0.5
                    qq_root.append(v)
            print('\npair ave_r pair_freq q_F')
            ave_rs_all = np.average(molecule.mat_r, axis=0)
            qq_root = np.array(qq_root)
            q_F = (np.sum(molecule.mat_F, axis=0) ** 2) / len(molecule.coords)
            pair_freq = q_F / qq_root
            for ij in range(_NC2):
                print('{} {:.3f} {:.3f} {:.3f}'.format(
                        atom_pairs[ij], ave_rs_all[ij], pair_freq[ij], q_F[ij]))
            Plotter.xy_scatter([ave_rs_all], [pair_freq], [''], ['k'], 
                    '$<r_{ij}> / \AA$', 'pair_freq', [10], 
                    'scatter-ave_r-pair_freq.pdf')
            Plotter.xy_scatter([ave_rs_all], [q_F], [''], ['k'], 
                    '$<r_{ij}> / \AA$', 'q / kcal/mol/$\AA$', [10], 
                    'scatter-ave_r-q_F.pdf')
            sys.exit()
            '''

            print('\nPlot decompFE vs r and histogram of decompFE')
            Plotter.hist_1d([molecule.mat_FE], 'q / kcal/mol', 'P(q)', 
                    'hist_q.pdf')
            interatomic_measures = Binner()
            interatomic_measures.get_bond_pop(molecule.coords, pairs)
            dists = interatomic_measures.rs.flatten()
            decompFE = molecule.mat_FE[:,:_NC2,].flatten()
            Plotter.xy_scatter([dists], [decompFE], [''], ['k'], '$r_{ij} / \AA$',
                    'q / kcal/mol', [10], 'scatter-r-decompFE.pdf')
            sys.stdout.flush()

            print('\nRemove high magnitude decompFE structures')
            print('n_structures', len(molecule.coords))
            q_max = np.amax(np.abs(molecule.mat_FE), axis=1).flatten()
            mean_q_max = np.mean(q_max)
            print('q_max', np.amax(q_max), 'mean_q_max', mean_q_max)
            cap_q_max = np.percentile(q_max, 98)
            q_close_idx = np.where(q_max <= cap_q_max)
            print('n_filtered_structures', len(q_close_idx[0]))
            molecule.coords = np.take(molecule.coords, q_close_idx, axis=0)[0]
            molecule.forces = np.take(molecule.forces, q_close_idx, axis=0)[0]
            molecule.energies = np.take(molecule.energies, q_close_idx, 
                    axis=0)[0]
            molecule.orig_energies = np.take(molecule.orig_energies, 
                    q_close_idx, axis=0)[0]
            sys.stdout.flush()



        print('\nDefine training and test sets')
        train_split = math.ceil(len(molecule.coords) / n_training)
        print('train_split', train_split)
        molecule.train = np.arange(0, len(molecule.coords), train_split).tolist() 
        val_split = math.ceil(len(molecule.train) / n_val)
        print('val_split', val_split)
        molecule.val = molecule.train[::val_split]
        molecule.train2 = [x for x in molecule.train if x not in molecule.val]
        molecule.test = [x for x in range(0, len(molecule.coords)) 
                if x not in molecule.train]
        if n_test > len(molecule.test):
            n_test = len(molecule.test)
        if n_test == 0:
            test_split = 0
            molecule.test = [0]
        if n_test !=0:
            test_split = math.ceil(len(molecule.test) / n_test)
            molecule.test = molecule.test[::test_split]
        print('test_split', test_split)
        print('Check! \nN training2 {} \nN val {} \nN test {}'.format(
            len(molecule.train2), len(molecule.val), len(molecule.test)))
        sys.stdout.flush()

        
        print('\nScale orig energies again, according to training set forces')
        prescale = prescale_energies(molecule, molecule.train)
        print('E SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('prescale value:', prescale)
        sys.stdout.flush()
        Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                bias_type=bias, extra_cols=extra_cols)
        print('N train: {} \nN test: {}'.format(len(molecule.train), 
            len(molecule.test)))
        print('train E min/max', 
                np.min(np.take(molecule.energies, molecule.train)), 
                np.max(np.take(molecule.energies, molecule.train)))
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
        sys.stdout.flush()


        print('\nPlot decompFE vs r and histogram of decompFE '\
                '(if filtered=True, then for refined) data')
        interatomic_measures = Binner()
        interatomic_measures.get_bond_pop(molecule.coords, pairs)
        dists = interatomic_measures.rs.flatten()
        decompFE = molecule.mat_FE[:,:_NC2,].flatten()
        Plotter.xy_scatter([dists], [decompFE], [''], ['k'], '$r_{ij} / \AA$',
                'q / kcal/mol', [10], 'scatter-r-decompFE2.pdf')
        Plotter.hist_1d([molecule.mat_FE], 'q / kcal/mol', 'P(q)', 'hist_q2.pdf')
        sys.stdout.flush()

    '''
    network = Network(molecule)
    if load_model == None: 
        print('\nTrain ANN')
        #bias = 'sum'
        model = Network.get_coord_FE_model(network, molecule, prescale, 
                n_nodes=n_nodes, n_layers=n_layers, grad_loss_w=grad_loss_w, 
                qFE_loss_w=qFE_loss_w, E_loss_w=E_loss_w, bias=bias)
        sys.stdout.flush()
    '''

    dihs = False
    if dihs:
        print('dihedrals:', dihedrals)

        molecule_md17 = Molecule() #initiate molecule class
        NPParser(atom_file, 
                [input_files[0]+'/coords.txt'], 
                [input_files[0]+'/forces.txt'], 
                [input_files[0]+'/energies.txt'], 
                molecule_md17)

        molecule_qm = Molecule() #initiate molecule class
        NPParser(atom_file, 
                [input_files[1]+'/coords.txt'], 
                [input_files[1]+'/forces.txt'], 
                [input_files[1]+'/energies.txt'], 
                molecule_qm)

        measures_md17 = Binner()
        measures_md17.get_dih_pop(molecule_md17.coords, dihedrals)

        measures_qm = Binner()
        measures_qm.get_dih_pop(molecule_qm.coords, dihedrals)

        Plotter.hist_2d([measures_qm.phis.T[0], measures_md17.phis.T[0]], 
                [measures_qm.phis.T[1], measures_md17.phis.T[1]], 
                ['Reds', 'Greys'],
                '$\\tau_1$ (degrees)', '$\\tau_2$ (degrees)', 
                'hist2d-dihs-qm-md17.pdf')

        Plotter.hist_2d([measures_md17.phis.T[0]], 
                [measures_md17.phis.T[1]], 
                ['Greys'], '$\\tau_1$ (degrees)', '$\\tau_2$ (degrees)', 
                'hist2d-dihs-md17.pdf')


        Plotter.hist_2d([measures_qm.phis.T[0]], 
                [measures_qm.phis.T[1]], 
                ['Reds'], '$\\tau_1$ (degrees)', '$\\tau_2$ (degrees)', 
                'hist2d-dihs-qm.pdf')


        Plotter.hist_1d([measures_qm.phis.T[0], measures_md17.phis.T[0]], 
                '$\\tau_1$ (degrees)', 'P($\\tau_1$)', 
                'hist1d-dih1-qm-md17.pdf', color_list=['r', 'k'])

        Plotter.hist_1d([measures_qm.phis.T[1], measures_md17.phis.T[1]], 
                '$\\tau_2$ (degrees)', 'P($\\tau_2$)', 
                'hist1d-dih2-qm-md17.pdf', color_list=['r', 'k'])


    sim = False
    if sim:
        # ML simulation, get time where geometries become
        # unstable and print out these times.
        molecule_sim = Molecule() #initiate molecule class

        if dihs:
            NPParser(atom_file, 
                    [input_files[2]+'/openmm-coords.txt'], 
                    [input_files[2]+'/openmm-forces.txt'], 
                    [input_files[2]+'/openmm-delta_energies.txt'], 
                    molecule_sim)
        else:
            NPParser(atom_file, 
                    [input_files[0]+'/openmm-coords.txt'], 
                    [input_files[0]+'/openmm-forces.txt'], 
                    [input_files[0]+'/openmm-delta_energies.txt'], 
                    molecule_sim)


        adjacency_mat = Molecule.find_bonded_atoms(molecule_sim.atoms,
                molecule_sim.coords[0])
        print('A', adjacency_mat)
        #A_flat = np.tril(adjacency_mat) #lower tri for A
        #print('A_flat', A_flat)

        atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) 
                for z, n in zip(molecule_sim.atoms, 
                range(1,len(molecule_sim.atoms)+1))]
        for i in range(len(molecule_sim.atoms)):
            print(i, atom_names[i])
        print()
        pairs = []
        _N = 0
        for i in range(len(molecule_sim.atoms)):
            for j in range(i):
                if adjacency_mat[i,j] == 1:
                    print(_N, atom_names[i], atom_names[j])
                    pairs.append([i, j])
                _N += 1
        print()

        sim_interatomic_measures = Binner()
        sim_interatomic_measures.get_bond_pop(
                molecule_sim.coords, pairs)
        print(sim_interatomic_measures.rs.shape)

        delta_r = 0.5
        dt = 5 #5_000 #fs 
        fs_to_ns = 1e6
        print('delta_r', delta_r, 'dt', dt)
        for i in range(len(sim_interatomic_measures.rs)):
            diff_r = (sim_interatomic_measures.rs[i] - 
                    sim_interatomic_measures.rs[0])
            max_t = (i * dt) / fs_to_ns
            if max(diff_r) > delta_r:
                max_t = (i * dt) / fs_to_ns
                break
        print(i, max_t)

        if dihs:
            measures_sim = Binner()
            measures_sim.get_dih_pop(molecule_sim.coords[:i], dihedrals)
            Plotter.hist_2d([measures_md17.phis.T[0], measures_sim.phis.T[0]], 
                    [measures_md17.phis.T[1], measures_sim.phis.T[1]], 
                    ['Greys', 'Reds'],
                    '$\\tau_1$ (degrees)', '$\\tau_2$ (degrees)', 
                    'hist2d-dihs-md17-sim.pdf')

            



    if load_model != None:
        network = Network(molecule)
        #model = load_model('best_ever_model_6')
        model = Network.get_coord_FE_model(network, molecule, prescale,
                training_model=False, load_model=load_model)
        sys.stdout.flush()
        print(datetime.now() - startTime)
        #################################################################


        if atom_file and mol:
            print('\ncheck model predictions')
            all_atoms = np.tile(molecule.atoms, (len(molecule.coords), 1))
            prediction = model.predict([molecule.coords, all_atoms])[0]
            for c, f in zip(molecule.coords, prediction):
                print('C', c)
                print('FPRED', f)
                break
            prediction = model.predict([molecule.coords[0].reshape(1,-1,3), 
                    np.array(molecule.atoms).reshape(1,-1)])
            print(prediction)
            print(model)


            ##### S-CURVS AND L1s
            atoms = np.array([float(i) for i in molecule.atoms], 
                    dtype='float32')
            input_coords = molecule.coords#.reshape(-1,n_atoms*3)
            output_E = molecule.energies.reshape(-1,1)
            output_F = molecule.forces.reshape(-1,n_atoms,3)
            output_matFE = molecule.mat_FE.reshape(-1,_NC2+extra_cols)
            test_input_coords = np.take(input_coords, molecule.test, axis=0)
            test_atoms = np.tile(atoms, (len(test_input_coords), 1))
            test_output_E = np.take(output_E, molecule.test, axis=0)
            test_output_E_postscale = ((test_output_E - prescale[2]) / 
                    (prescale[3] - prescale[2]) * 
                    (prescale[1] - prescale[0]) + prescale[0])
            test_output_F = np.take(output_F, molecule.test, axis=0)
            test_output_matFE = np.take(output_matFE, molecule.test, axis=0)

            print('Predict testset data')
            test_prediction = model.predict([test_input_coords, test_atoms])
            test_prediction_E = test_prediction[2].flatten()
            test_prediction_F = test_prediction[0]
            test_prediction_matFE = test_prediction[1]

            bin_edges, hist = Binner.get_scurve(
                    test_output_E_postscale.flatten(), 
                    test_prediction_E.flatten(), 'testset_hist_E.dat')
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'testset_s_curve_E.pdf', log=True)
            ind = [math.floor(i) for i in bin_edges].index(1)
            L1 = hist[ind]
            print('E ind bin L1', ind, bin_edges[ind], L1)
            bin_edges, hist = Binner.get_scurve(test_output_F.flatten(), 
                    test_prediction_F.flatten(), 'testset_hist_F.dat')
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'testset_s_curves_F.pdf', log=True)
            ind = [math.floor(i) for i in bin_edges].index(1)
            L1 = hist[ind]
            print('F ind bin L1', ind, bin_edges[ind], L1)
            bin_edges, hist = Binner.get_scurve(test_output_matFE.flatten(), 
                    test_prediction_matFE.flatten(), 
                    'testset_hist_decompFE.dat')
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'testset_s_curves_decompFE.pdf', log=True)
            ind = [math.floor(i) for i in bin_edges].index(1)
            L1 = hist[ind]
            print('qFE ind bin L1', ind, bin_edges[ind], L1)

            
        cp2k_ = True
        if cp2k_:
            ##### CP2K DATA
            print('\n\n******** CP2K DATA *********')
            molecule_cp2k = Molecule() #initiate molecule class
            #cp2k_path = input_files[0]+'/cp2k_parsed'
            #cp2k_path = input_files[0]+'/cp2k_parsed_all'
            #NPParser(atom_file, [cp2k_path+'/C.txt'], [cp2k_path+'/F.txt'], 
                    #[cp2k_path+'/E.txt'], molecule_cp2k)
            #Writer.write_gaus_cart(molecule_cp2k.coords,#[0].reshape(1,-1,3), 
                    #molecule_cp2k.atoms, '', 'cp2k_all')

            # for cp2k redone in Gaussian
            cp2k_path = input_files[0]+'/aq-rMD17_gauss'
            NPParser(atom_file, [cp2k_path+'/coords.txt'], 
                    [cp2k_path+'/forces.txt'], 
                    [cp2k_path+'/energies.txt'], molecule_cp2k)

            # if N structures in molecule_cp2k == 100,000 then reduce to
            # 20,000 for quicker analysis
            if len(molecule_cp2k.coords) >= 100_000:
                molecule_cp2k.coords = molecule_cp2k.coords[::5]
                molecule_cp2k.energies = molecule_cp2k.energies[::5]
                molecule_cp2k.forces = molecule_cp2k.forces[::5]

            '''
            # cp2k sims in soln or re-done md17 
            molecule_cp2k = Molecule() #initiate molecule class
            OPTParser(
                    [input_files[0]+'/gaus_files2/cp2k_all.out'], 
                    #[input_files[0]+'/gaus_md17/cp2k_all.out'], 
                    molecule_cp2k, 
                    opt=False) #read in FCEZ for SP
            '''

            '''
            # md17 dataset
            molecule = Molecule() #initiate molecule class
            NPParser(atom_file, coord_files, force_files, energy_files, 
                    molecule)

            Writer.write_xyz(molecule.coords, molecule.atoms, 'md17_all.xyz', 
                    'w')

            if n_test == -1 or n_test > len(molecule.coords):
                n_test = len(molecule.coords) - n_training
            print('\nDefine training and test sets')
            train_split = math.ceil(len(molecule.coords) / n_training)
            print('train_split', train_split)
            molecule.train = np.arange(0, len(molecule.coords), 
                    train_split).tolist() 
            val_split = math.ceil(len(molecule.train) / n_val)
            print('val_split', val_split)
            molecule.val = molecule.train[::val_split]
            molecule.train2 = [x for x in molecule.train if x not in 
                    molecule.val]
            molecule.test = [x for x in range(0, len(molecule.coords)) 
                    if x not in molecule.train]
            if n_test > len(molecule.test):
                n_test = len(molecule.test)
            if n_test == 0:
                test_split = 0
                molecule.test = [0]
            if n_test !=0:
                test_split = math.ceil(len(molecule.test) / n_test)
                molecule.test = molecule.test[::test_split]
            print('test_split', test_split)
            print('Check! \nN training2 {} \nN val {} \nN test {}'.format(
                len(molecule.train2), len(molecule.val), len(molecule.test)))
            sys.stdout.flush()

            molecule.coords = np.take(molecule.coords, molecule.test, axis=0)
            molecule.forces = np.take(molecule.forces, molecule.test, axis=0)
            molecule.energies = np.take(molecule.energies, molecule.test, axis=0)
            

            mae, rms, msd = Binner.get_error(
                    molecule.energies[:len(molecule_cp2k.coords)].flatten(), 
                    molecule_cp2k.energies.flatten())
            print('\n{} testset structures, errors for E / kcal/mol:- '\
                    '\nMAE: {:.3f} '\
                    '\nRMS: {:.3f} \nMSD: {:.3f} '\
                    '\nMSE: {:.3f}'.format(
                    len(molecule_cp2k.energies), mae, rms, msd, rms**2))

            mae, rms, msd = Binner.get_error(
                    molecule.forces[:len(molecule_cp2k.coords)].flatten(), 
                    molecule_cp2k.forces.flatten())
            print('\n{} testset structures, errors for gradient (F) / '\
                    'kcal/mol/$\AA$:- '\
                    '\nMAE: {:.3f} \nRMS: {:.3f} '\
                    '\nMSD: {:.3f} \nMSE: {:.3f}'.format(
                    len(molecule_cp2k.forces), mae, rms, msd, rms**2))



            sys.exit()

            molecule_cp2k.coords = np.take(molecule.coords, molecule.test, axis=0)
            molecule_cp2k.forces = np.take(molecule.forces, molecule.test, axis=0)
            molecule_cp2k.energies = np.take(molecule.energies, molecule.test, axis=0)
            
            Writer.write_gaus_cart(molecule_cp2k.coords,#[0].reshape(1,-1,3), 
                    molecule_cp2k.atoms, '', 'cp2k_all')
            '''

            n_atoms = len(molecule_cp2k.atoms)
            atoms = np.array([float(i) for i in molecule_cp2k.atoms], 
                    dtype='float32')
            test_input_coords = molecule_cp2k.coords#[0].reshape(-1,n_atoms,3)
            test_output_E_postscale = molecule_cp2k.energies#[0].reshape(-1,1)
            test_output_F = molecule_cp2k.forces#[0].reshape(-1,n_atoms,3)
            #test_output_matFE = molecule_cp2k.mat_FE.reshape(-1,_NC2+extra_cols)
            test_atoms = np.tile(atoms, (len(test_input_coords), 1))
            test_prediction = model.predict(
                    [test_input_coords, test_atoms])
            test_prediction_E = test_prediction[2].flatten()
            test_prediction_F = test_prediction[0]
            #test_prediction_matFE = test_prediction[1]


            bin_edges, hist = Binner.get_scurve(
                    test_output_E_postscale.flatten(), 
                    test_prediction_E.flatten(), 'testset_hist_E.dat')
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'testset_s_curve_E.pdf', log=True)
            try:
                ind = [math.floor(i) for i in bin_edges].index(1)
            except ValueError:
                if max(bin_edges) <= 1:
                    ind = len(bin_edges)-1
                else:
                    ind = 0
            L1 = hist[ind]
            print('E ind bin L1', ind, bin_edges[ind], round(L1,3))

            bin_edges, hist = Binner.get_scurve(test_output_F.flatten(), 
                    test_prediction_F.flatten(), 'testset_hist_F.dat')
            Plotter.plot_2d([bin_edges], [hist], [''], 
                    'Error', '% of points below error', 
                    'testset_s_curves_F.pdf', log=True)
            try:
                ind = [math.floor(i) for i in bin_edges].index(1)
            except ValueError:
                if max(bin_edges) <= 1:
                    ind = len(bin_edges)-1
                else:
                    ind = 0
            L1 = hist[ind]
            print('F ind bin L1', ind, bin_edges[ind], round(L1,3))


            mae, rms, msd = Binner.get_error(test_output_E_postscale.flatten(), 
                    test_prediction_E.flatten())
            print('\n{} testset structures, errors for E / kcal/mol:- '\
                    '\nMAE: {:.3f} '\
                    '\nRMS: {:.3f} \nMSD: {:.3f} '\
                    '\nMSE: {:.3f}'.format(
                    len(test_output_E_postscale), mae, rms, msd, rms**2))

            mae, rms, msd = Binner.get_error(test_output_F.flatten(), 
                    test_prediction_F.flatten())
            print('\n{} testset structures, errors for gradient (F) / '\
                    'kcal/mol/$\AA$:- '\
                    '\nMAE: {:.3f} \nRMS: {:.3f} '\
                    '\nMSD: {:.3f} \nMSE: {:.3f}'.format(
                        len(test_output_F), mae, rms, msd, rms**2))



            #'''
            ## plot interatomic distances for cp2k sims and ML/MM sims in soln

            molecule_mlmm = Molecule() #initiate molecule class
            mlmm_path = input_files[1]
            NPParser(atom_file, [mlmm_path+'/openmm-coords.txt'], 
                    [mlmm_path+'/openmm-forces.txt'], 
                    [mlmm_path+'/openmm-delta_energies.txt'], molecule_mlmm)


            molecule_md17 = Molecule() #initiate molecule class
            NPParser(atom_file, coord_files, force_files, energy_files, 
                    molecule_md17)

        
            atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) 
                    for z, n in zip(molecule_cp2k.atoms, 
                    range(1,len(molecule_cp2k.atoms)+1))]
            for i in range(len(molecule_cp2k.atoms)):
                print(i, atom_names[i])
            print()
            pairs = []
            _N = 0
            for i in range(len(molecule_cp2k.atoms)):
                for j in range(i):
                    print(_N, atom_names[i], atom_names[j])
                    pairs.append([i, j])
                    _N += 1


            cp2k_interatomic_measures = Binner()
            cp2k_interatomic_measures.get_bond_pop(
                    molecule_cp2k.coords, pairs)

            mlmm_interatomic_measures = Binner()
            mlmm_interatomic_measures.get_bond_pop(
                    molecule_mlmm.coords, pairs)

            md17_interatomic_measures = Binner()
            md17_interatomic_measures.get_bond_pop(
                    molecule_md17.coords[::10], pairs)


            idx = 31 #53
            info = [[
                        cp2k_interatomic_measures.rs.T[idx].flatten(),
                        #mlmm_interatomic_measures.rs.T[idx].flatten(), 
                        md17_interatomic_measures.rs.T[idx].flatten(), 
                        100, '$r / \AA$', '{}'.format(idx)]
                    ]

            for i in info:
                bin_edges, hist = Binner.get_hist(i[0], i[2])
                bin_edges2, hist2 = Binner.get_hist(i[1], i[2])
                Plotter.xy_scatter(
                        [bin_edges2, bin_edges], 
                        [hist2, hist], 
                        ['DFT', 'ML'], ['k', 'r'], i[3], 
                        'P($r$)', [10, 10],
                        'hist-cp2k-md17-r-{}.pdf'.format(i[4]))



            info = [[
                        cp2k_interatomic_measures.rs.T.flatten(),
                        mlmm_interatomic_measures.rs.T.flatten(), 
                        #md17_interatomic_measures.rs.T.flatten(), 
                        1000, '$r / \AA$', 'all']
                    ]

            for i in info:
                bin_edges, hist = Binner.get_hist(i[0], i[2])
                bin_edges2, hist2 = Binner.get_hist(i[1], i[2])
                Plotter.xy_scatter(
                        [bin_edges2, bin_edges], 
                        [hist2, hist], 
                        ['DFT', 'ML'], ['k', 'r'], i[3], 
                        'P($r$)', [10, 10],
                        'hist-cp2k-mlmm-r-{}.pdf'.format(i[4]))

            info = [[
                        #cp2k_interatomic_measures.rs.T.flatten(),
                        md17_interatomic_measures.rs.T.flatten(), 
                        mlmm_interatomic_measures.rs.T.flatten(), 
                        1000, '$r_{ij} / \AA$', 'all']
                    ]

            for i in info:
                bin_edges, hist = Binner.get_hist(i[0], i[2])
                bin_edges2, hist2 = Binner.get_hist(i[1], i[2])
                Plotter.xy_scatter(
                        [bin_edges2, bin_edges], 
                        [hist2, hist], 
                        ['DFT', 'ML'], ['k', 'r'], i[3], 
                        'P($r_{ij}$)', [10, 10],
                        'hist-md17-mlmm-r-{}.pdf'.format(i[4]))



            info = [[
                        cp2k_interatomic_measures.rs.T.flatten(),
                        mlmm_interatomic_measures.rs.T.flatten(), 
                        md17_interatomic_measures.rs.T.flatten(), 
                        1000, '$r /$ \AA', 'all']
                    ]

            for i in info:
                bin_edges, hist = Binner.get_hist(i[0], i[3])
                bin_edges2, hist2 = Binner.get_hist(i[1], i[3])
                bin_edges3, hist3 = Binner.get_hist(i[2], i[3])
                Plotter.xy_scatter(
                        [bin_edges2, bin_edges, bin_edges3], 
                        [hist2, hist, hist3], 
                        ['DFT', 'ML', 'MD17'], ['k', 'r', 'dodgerblue'], i[4], 
                        'P($r$)', [10, 10, 10],
                        'hist-cp2k-mlmm-md17-r-{}.pdf'.format(i[5]))
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
        group = parser.add_argument('-d', '--dihedrals', nargs='+', 
                action='append', type=int, default=[],
                help='list of dihedrals')
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
            filtered=op.filtered, load_model=op.load_model, 
            dihedrals=op.dihedrals)

if __name__ == '__main__':
    main()


