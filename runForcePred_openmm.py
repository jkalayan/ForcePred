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
                               
from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, AMBLAMMPSParser, AMBERParser, XYZParser, Binner, \
        Writer, Plotter, Network, Conservation

from keras.models import Model, load_model    
import sys
#import numpy as np
#from itertools import islice

#import os
#os.environ['OMP_NUM_THREADS'] = '8'


def run_force_pred(input_files='input_files',
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files'):


    '''
    #ethanediol    
    scale_NRF = 19911.051275945494
    scale_NRF_min = 11.648288117257646
    scale_F = 105.72339728320982 
    scale_F_min = 0
    '''

    #'''
    #malonaldehyde 
    scale_NRF = 12364.344210692281
    scale_NRF_min = 15.377554794644805
    scale_F = 102.58035961189837
    scale_F_min = 2.4191760310543486e-06
    #'''


    startTime = datetime.now()
    print(startTime)
    print('get molecule')
    molecule = Molecule() #initiate molecule class
    #XYZParser(atom_file, coord_files, force_files, energy_files, molecule)
    NPParser(atom_file, coord_files, force_files, [], molecule)
    pdb_file = 'molecules.pdb'
    Writer.write_pdb(molecule.coords[0], 'MOL', 1, 
            molecule.atoms, pdb_file, 'w')
    masses = [Converter._ZM[i] for i in molecule.atoms]
    sys.stdout.flush()

    print('get network')
    network = Network.get_network(molecule, scale_NRF, scale_NRF_min, 
            scale_F, scale_F_min)
    sys.stdout.flush()

    print('get MD system')
    system, simulation, force = OpenMM.get_system(masses, pdb_file)
    sys.stdout.flush()

    print('run MD simulation')
    nsteps = 100000 #000 #10ns #100ps #20000 #10ps  #15000
    saved_steps = nsteps/5 #/1000
    model = load_model('../../best_ever_model')
    system, simulation, md = OpenMM.run_md(system, simulation, force, 
            molecule.atoms, network, model, nsteps, saved_steps)
    print(datetime.now() - startTime)
    sys.stdout.flush()

    print('checking invariance of forces')
    md.coords = md.get_3D_array([md.coords])
    md.forces = md.get_3D_array([md.forces])
    md.energies = np.array(md.energies)
    unconserved = md.check_force_conservation()
    sys.stdout.flush()

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
        model = load_model('../best_ever_model')
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

    print(datetime.now() - startTime)


def main():

    try:
        usage = 'runForcePred_openmm.py [-h]'
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


