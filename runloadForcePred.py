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
        qFE_loss_w='qFE_loss_w', E_loss_w='E_loss_w', epochs='epochs', 
        bias='bias', filtered='filtered', solution='solution', 
        load_model='load_model', 
        temp='temp', nsteps='nsteps', dt='dt'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class
    prescale = [0, 1, 0, 1, 0, 0]
    #read in input data files
    if list_files:
        input_files = open(list_files).read().split()
    if atom_file or input_files:
        if input_files:
            OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
        else:
            NPParser(atom_file, coord_files, force_files, energy_files, 
                    molecule)
        print(datetime.now() - startTime)
        sys.stdout.flush()

        print(molecule)
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        sys.stdout.flush()

        isExist = os.path.exists('model')
        if not isExist:
            os.makedirs('model')
        isExist = os.path.exists('plots')
        if not isExist:
            os.makedirs('plots')
        # write pdb file for first frame
        Writer.write_pdb(molecule.coords[0], 'MOL', 1, molecule.atoms, 
                'model/molecule.pdb', 'w')
        sys.stdout.flush()

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


            print('\nPlot decompFE vs r and histogram of decompFE')
            Plotter.hist_1d([molecule.mat_FE], 'q / kcal/mol', 'P(q)', 
                    'plots/hist_q.png')
            interatomic_measures = Binner()
            interatomic_measures.get_bond_pop(molecule.coords, pairs)
            dists = interatomic_measures.rs.flatten()
            decompFE = molecule.mat_FE[:,:_NC2,].flatten()
            Plotter.xy_scatter([dists], [decompFE], [''], ['k'], '$r_{ij} / \AA$',
                    'q / kcal/mol', [10], 'plots/scatter-r-decompFE.png')
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
                'q / kcal/mol', [10], 'plots/scatter-r-decompFE2.png')
        Plotter.hist_1d([molecule.mat_FE], 'q / kcal/mol', 'P(q)', 
                'plots/hist_q2.png')
        sys.stdout.flush()

    network = Network(molecule)
    if load_model == None: 
        print('\nTrain ANN')
        #bias = 'sum'
        model = Network.get_coord_FE_model(network, molecule, prescale, 
                n_nodes=n_nodes, n_layers=n_layers, grad_loss_w=grad_loss_w, 
                qFE_loss_w=qFE_loss_w, E_loss_w=E_loss_w, bias=bias, epochs=epochs)
        sys.stdout.flush()

    if load_model != None:
        #model = load_model('best_ever_model_6')
        model = Network.get_coord_FE_model(network, molecule, prescale,
                training_model=False, load_model=load_model, n_nodes=n_nodes, 
                n_layers=n_layers)
        sys.stdout.flush()
        print(datetime.now() - startTime)
        #################################################################

        if atom_file:
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





        print('\nrun MD simulation')
        isExist = os.path.exists('simulation')
        if not isExist:
            os.makedirs('simulation')

        atoms = np.loadtxt(load_model+'/atoms.txt', 
                dtype=np.float32).reshape(-1)
        masses = [Converter._ZM[i] for i in atoms]
        pdb_file=load_model+'/molecule.pdb'


        #nsteps = 200#_000#_000 #100ns #25_000_000 #12.5ns 20000 #10ps
        #dt = 10#_000
        saved_steps = nsteps/dt#/10000#000#2#40#

        print('nsteps: {} saved_steps: {}'.format(nsteps, saved_steps))
        #model = load_model('../../best_ever_model')

        print('\nget MD system')
        #temp = 500
        if solution:
            print("Simulation setup in soln")
            system, simulation, force, integrator, init_positions, init_forces = \
                    OpenMM.get_system(masses, pdb_file, dt, temp,
                    mlff=False, mlmm=True, #for solution
                    )
        else:
            print("Simulation setup in gas")
            system, simulation, force, integrator, init_positions, init_forces = \
                    OpenMM.get_system(masses, pdb_file, dt, temp,
                    #mlff=False, mlmm=True, #for solution
                    )

        print(datetime.now() - startTime)
        sys.stdout.flush()


        #setup object for collecting simulation results
        md = Molecule()
        md.atoms = atoms
        md.coords = []
        md.forces = []
        md.energies = []

        ##write new files here first
        open('simulation/openmm-coords.txt', 'w').close() #needed to continue sims
        open('simulation/openmm-coords.xyz', 'w').close()
        open('simulation/openmm-forces.txt', 'w').close()
        open('simulation/openmm-velocities.txt', 'w').close()
        open('simulation/openmm-delta_energies.txt', 'w').close()
        #open('simulation/openmm-f-curl.txt', 'w').close()
        open('simulation/openmm-mm-forces.txt', 'w').close()

        if solution:
            system, simulation, md = OpenMM.run_md(system, simulation, md, 
                    force, integrator, temp, network, model, 
                    nsteps, saved_steps, init_positions, init_forces, prescale,
                    mlmm=True, mlff=False, #for solution
                    )
        else:
            system, simulation, md = OpenMM.run_md(system, simulation, md, force, 
                    integrator, temp, network, model, 
                    nsteps, saved_steps, init_positions, init_forces, prescale,
                    #mlmm=True, mlff=False, #for solution
                    )


        print(datetime.now() - startTime)
        sys.stdout.flush()

        print('\nchecking invariance of forces')
        md.coords = md.get_3D_array([md.coords])
        md.forces = md.get_3D_array([md.forces])
        md.energies = np.array(md.energies)
        unconserved = md.check_force_conservation()
        sys.stdout.flush()

        Writer.write_gaus_cart(md.coords,#[0].reshape(1,-1,3), 
                md.atoms, '', 'simulation/ml_sim')


    print(datetime.now() - startTime)


def main():

    try:
        usage = 'runloadForcePred.py [-h]'
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
        group = parser.add_argument('-epochs', '--epochs', 
                action='store', default=1_000_000, type=int,
                help='number of epochs for training')
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
        group = parser.add_argument('-solution', '--solution', 
                action='store_true', 
                help='run simulation in solution.')
        group = parser.add_argument('-load_model', '--load_model', 
                action='store', default=None,
                help='load an existing network to perform MD, provide '\
                        'the path+folder to model here')
        group = parser.add_argument('-temp', '--temp', 
                action='store', default=500, type=int,
                help='set temperature (K) of MD simulation')
        group = parser.add_argument('-nsteps', '--nsteps', 
                action='store', default=200, type=int,
                help='set number of steps to take for MD simulation, '\
                        'each time step is 0.5 fs')
        group = parser.add_argument('-dt', '--dt', 
                action='store', default=10, type=int,
                help='interval between number of steps saved to file')

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
            qFE_loss_w=op.qFE_loss_w, E_loss_w=op.E_loss_w, epochs=op.epochs, 
            bias=op.bias, filtered=op.filtered, solution=op.solution, 
            load_model=op.load_model, 
            temp=op.temp, nsteps=op.nsteps, dt=op.dt)

if __name__ == '__main__':
    main()


