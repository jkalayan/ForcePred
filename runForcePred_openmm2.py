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

#from ForcePred.nn.Network_v2 import Network
#from ForcePred.nn.Network_atomwise import Network
from ForcePred.nn.Network_perminv import Network

from keras.models import Model, load_model    
from keras import backend as K                                              
import sys
import tensorflow as tf
#import numpy as np
#from itertools import islice

import os
#os.environ['OMP_NUM_THREADS'] = '8'

NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


def run_force_pred(input_files='input_files',
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files'):

    prescale = 0
    startTime = datetime.now()
    print(startTime)
    print('\nget molecule')
    molecule = Molecule() #initiate molecule class
    #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #XYZParser(atom_file, coord_files, force_files, energy_files, molecule)
    NPParser(atom_file, coord_files, force_files, energy_files, molecule)
    #NPParser(atom_file, coord_files, force_files, [], molecule)
    #NPParser(atom_file, [coord_files[0]], [force_files[0]], 
            #[energy_files[0]], molecule)
    #molecule2 = Molecule() #initiate molecule class
    #NPParser(atom_file, [coord_files[1]], [force_files[1]], 
            #[energy_files[1]], molecule2)
    print(molecule)

    '''
    A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    indices, pairs_dict = Molecule.find_equivalent_atoms(molecule.atoms, A)
    print('\nZ', molecule.atoms)
    print('A', A) 
    print('indices', indices)
    print('pairs_dict', pairs_dict)
    '''

    sys.stdout.flush()
    #sys.exit()

    '''
    print('get initial pdb structure')
    pdb_file = 'molecules.pdb'
    Writer.write_pdb(molecule.coords[0], 'MOL', 1, 
            molecule.atoms, pdb_file, 'w')
    masses = [Converter._ZM[i] for i in molecule.atoms]
    sys.stdout.flush()
    '''

    pdb_file='../../revised_data/3md.rst.pdb'
    masses = [Converter._ZM[i] for i in molecule.atoms]

    print('\nget MD system')
    system, simulation, force, integrator, init_positions, init_forces = \
            OpenMM.get_system(masses, pdb_file)
    print(datetime.now() - startTime)
    sys.stdout.flush()



    print('\ninternal FE decomposition')
    Converter.get_simultaneous_interatomic_energies_forces(molecule, 
            bias_type='1/r')
    print(datetime.now() - startTime)
    sys.stdout.flush()


    print('\nload network')
    network = Network(molecule)
    prescale_energies = True
    #find max NRF
    Converter.get_all_NRFs(molecule)
    print('max NRF', np.max(np.abs(molecule.mat_NRF)))
    print('max F', np.max(np.abs(molecule.forces)))
    if prescale_energies:
        prescale = [0, 1, 0, 1]
        split = 100 #500 #200 #100
        train = round(len(molecule.coords) / split, 3)
        print('\nget train and test sets, '\
                'training set is {} points'.format(train))
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        train_forces = np.take(molecule.forces, molecule.train, axis=0)
        train_energies = np.take(molecule.energies, molecule.train, axis=0)
        print('E ORIG min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('train E ORIG min: {} max: {}'.format(np.min(train_energies), 
                np.max(train_energies)))
        print('F ORIG min: {} max: {}'.format(np.min(molecule.forces), 
                np.max(molecule.forces)))
        print('train F ORIG min: {} max: {}'.format(np.min(train_forces), 
                np.max(train_forces)))
        print('\nprescale energies so that magnitude is comparable to forces')
        min_e = np.min(train_energies)
        max_e = np.max(train_energies)
        min_f = np.min(train_forces)
        max_f = np.max(train_forces)

        molecule.energies = ((max_f - min_f) * (molecule.energies - min_e) / 
                (max_e - min_e) + min_f)

        prescale[0] = min_e
        prescale[1] = max_e
        prescale[2] = min_f
        prescale[3] = max_f

        print('E SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('prescale value:', prescale)
    print(datetime.now() - startTime)
    sys.stdout.flush()

    #model = load_model('best_ever_model_6')
    model = Network.get_coord_FE_model(network, molecule, prescale)
    sys.stdout.flush()
    print(datetime.now() - startTime)
    #################################################################

    print('\ncheck model predictions')
    prediction = model.predict(molecule.coords)[0]
    for c, f in zip(molecule.coords, prediction):
        print('C', c)
        print('FPRED', f)
        break
    prediction = model.predict(molecule.coords[0].reshape(1,-1,3))
    print(prediction)





    print('\nrun MD simulation')
    nsteps = 200_000#_000 #100ns #25_000_000 #12.5ns 20000 #10ps
    saved_steps = nsteps/40#/10000#000#2
    print('nsteps: {} saved_steps: {}'.format(nsteps, saved_steps))
    #model = load_model('../../best_ever_model')

    #setup object for collecting simulation results
    md = Molecule()
    md.atoms = molecule.atoms
    md.coords = []
    md.forces = []
    md.energies = []




    ##write new files here first
    open('openmm-coords.txt', 'w').close() #needed to continue sims
    open('openmm-coords.xyz', 'w').close()
    open('openmm-forces.txt', 'w').close()
    open('openmm-velocities.txt', 'w').close()
    open('openmm-delta_energies.txt', 'w').close()
    open('openmm-f-curl.txt', 'w').close()
    open('openmm-mm-forces.txt', 'w').close()
    #'''
    #no ramp
    temperature = 300
    system, simulation, md = OpenMM.run_md(system, simulation, md, force, 
            integrator, temperature, network, model, 
            nsteps, saved_steps, init_positions, init_forces, prescale)
    #'''

    '''
    #ramp temp
    for temperature in range(100,400,100):
        #if temperature < 800:
            #nsteps = 5000
        #if temperature >= 800:
            #nsteps = 20000
        nsteps = 2000
        saved_steps = nsteps#/10#000#000
        print('\ntemp: {} nsteps: {} saved_steps: {}'.format(temperature, 
                nsteps, saved_steps))
        system, simulation, md = OpenMM.run_md(system, simulation, md, force, 
                integrator, temperature, network, model, 
                nsteps, saved_steps, init_positions, init_forces, prescale)
    '''


    
    print(datetime.now() - startTime)
    sys.stdout.flush()

    print('\nchecking invariance of forces')
    md.coords = md.get_3D_array([md.coords])
    md.forces = md.get_3D_array([md.forces])
    md.energies = np.array(md.energies)
    unconserved = md.check_force_conservation()
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


