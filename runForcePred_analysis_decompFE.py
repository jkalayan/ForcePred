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

mdanal = True
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

import os
#os.environ['OMP_NUM_THREADS'] = '8'

NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()

    print('Load data')
    molecule = Molecule()


    '''
    print('Gaussian')
    OPTParser(['rmd17_1.out'], molecule, opt=False) #read in FCEZ for SP
    print(molecule.forces)
    print(molecule.energies)
    variance, translations, rotations = Network.check_invariance(
            molecule.coords[0], molecule.forces[0])
    print('variance\n', variance, translations, rotations)

    molecule.forces = (molecule.forces * Converter.Eh2kcalmol / 
            Converter.au2Ang)
    molecule.energies = molecule.energies * Converter.Eh2kcalmol

    print('cp2k')
    molecule2 = Molecule()
    XYZParser(atom_file, [coord_files[0]], [force_files[0]], [energy_files[0]], 
            molecule2)
    molecule2.forces = (molecule2.forces * Converter.Eh2kcalmol / 
            Converter.au2Ang)
    molecule2.energies = molecule2.energies * Converter.Eh2kcalmol
    print(molecule2.forces[0])
    print(molecule2.energies[0])
    variance, translations, rotations = Network.check_invariance(
            molecule2.coords[0], molecule2.forces[0])
    print('variance\n', variance, translations, rotations)
    sys.exit()
    '''
    #OPTParser([input_files[0]], molecule, opt=True) #read in FCEZ for opt

    NPParser(atom_file, [coord_files[0]], [force_files[0]], [energy_files[0]], 
            molecule)
    print('dataset', molecule.coords.shape, 
            molecule.forces.shape, molecule.energies.shape)
    print(molecule)

    n_atoms = len(molecule.atoms)
    _NC2 = int(n_atoms * (n_atoms-1)/2)


    #Writer.write_gaus_cart(molecule.coords[0].reshape(1,-1,3), 
            #molecule.atoms, '', 'rmd17_1')
    #sys.exit()

    '''
    print('!!!use regularly spaced training')
    #molecule.train = np.arange(0, len(molecule.coords), split).tolist() 
    molecule.train = np.arange(2, len(molecule.coords), split).tolist() 
            # includes the structure with highest 
            # force magnitude for malonaldehyde
    molecule.test = [x for x in range(0, len(molecule.coords)) 
            if x not in molecule.train]
    print(len(molecule.train))
    print(len(molecule.test))
    '''

    ##truncate dataset
    molecule.coords = molecule.coords[::100]
    molecule.forces = molecule.forces[::100]
    molecule.energies = molecule.energies[::100]

    molecule.train = [0]
    molecule.test = [0]
    prescale = [0, 1, 0, 1]
    train = True
    if train:
        ###!!!!!!!!
        print('!!!TRAIN OVER-RIDDEN FOR SCALING')
        train_forces = molecule.forces
        train_energies = molecule.energies

        forces_min = np.min(molecule.forces)
        forces_max = np.max(molecule.forces)
        forces_diff = forces_max - forces_min
        #print(forces_diff)
        forces_rms = np.sqrt(np.mean(molecule.forces.flatten()**2))
        energies_rms = np.sqrt(np.mean(molecule.energies.flatten()**2))
        forces_mean = np.mean(molecule.forces.flatten())
        energies_mean = np.mean(molecule.energies.flatten())
        energies_max = np.max(np.absolute(molecule.energies))
        #molecule.energies = (molecule.energies - energies_mean) / energies_rms
        #molecule.forces = (molecule.forces * 0) # forces_rms)


        print('E ORIG min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('train E ORIG min: {} max: {}'.format(np.min(train_energies), 
                np.max(train_energies)))
        print('F ORIG min: {} max: {}'.format(np.min(molecule.forces), 
                np.max(molecule.forces)))
        print('train F ORIG min: {} max: {}'.format(np.min(train_forces), 
                np.max(train_forces)))
        prescale_energies = True
        #prescale = [0, 1, 0, energies_mean, forces_rms, energies_max, 
                #forces_max]
        prescale = [0, 1, 0, 1]
        if prescale_energies:
            print('\nprescale energies so that magnitude is '\
                    'comparable to forces')
            min_e = np.min(train_energies)
            max_e = np.max(train_energies)
            min_f = np.min(train_forces)
            max_f = np.max(train_forces)

            molecule.energies = ((max_f - min_f) * 
                    (molecule.energies - min_e) / 
                    (max_e - min_e) + min_f)

            prescale[0] = min_e
            prescale[1] = max_e
            prescale[2] = min_f
            prescale[3] = max_f

            print('E SCALED min: {} max: {}'.format(
                    np.min(molecule.energies), np.max(molecule.energies)))


        print('prescale value:', prescale)
        sys.stdout.flush()



    bias_type='r'#'1/r'
    print('bias_type:', bias_type)
    Converter.get_simultaneous_interatomic_energies_forces(molecule, 
            bias_type)

    #Converter.get_interatomic_energies_directions(molecule, bias_type)
    #Converter.get_interatomic_energies_forces_directions(molecule, bias_type)

    #Converter.get_atomwise_decompF(molecule, bias_type)

    #sys.exit()


    print(datetime.now() - startTime)
    sys.stdout.flush()


    atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
            zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
    pairs = []
    chosen_inx = []
    _N = 0
    for i in range(len(molecule.atoms)):
        for j in range(i):
            pairs.append([i, j])
            print(_N+1, atom_names[i], atom_names[j])
            if 'O11' in [atom_names[i], atom_names[j]]:
                chosen_inx.append(_N)
                print('\t{}'.format(_N))
            #if 'C8' not in [atom_names[i], atom_names[j]] and \
                    #_N not in chosen_inx:
                #chosen_inx.append(_N)
                #print('\t{}'.format(_N))
            _N += 1


    interatomic_measures = Binner()
    interatomic_measures.get_bond_pop(molecule.coords, pairs)


    print(interatomic_measures.rs.shape, molecule.mat_FE.shape, 
            molecule.mat_F.shape, molecule.mat_E.shape)
        

    dists = interatomic_measures.rs.flatten()
    #decompFE = molecule.mat_E.flatten()
    decompFE = molecule.mat_FE.flatten()

    if len(molecule.mat_FE[0]) == _NC2+1:
        dists = np.ones((len(molecule.coords), _NC2+1))
        dists[:,-1] = interatomic_measures.rs
        dists = dists.flatten()

    #dists = interatomic_measures.rs[:,chosen_inx].flatten()
    #decompFE = molecule.mat_FE[:,chosen_inx].flatten()
    #decompFE = molecule.atom_F[:,10,chosen_inx].flatten()

    #chosen_inx = [i for i in range(21) if i not in [7,10]]
    #chosen_inx = [7,10]
    #dists = molecule.coords[:,chosen_inx,:].flatten()
    #decompFE = molecule.forces[:,chosen_inx,:].flatten()

    #dists = molecule.coords.flatten()
    #decompFE = molecule.forces.flatten()

    #dists = interatomic_measures.rs[:,52].flatten()
    #decompFE = molecule.atom_F[:,10,52].flatten()
    #decompFE = molecule.mat_FE[:,52].flatten()

    '''
    for i in range(len(molecule.coords)):
        #print(i, molecule.atom_F[i,10,52], molecule.atom_F[i,7,52])
        #recompE = np.dot(molecule.mat_F[i], molecule.mat_E[i])
        recompE = np.sum(molecule.mat_FE[i])
        print(i, interatomic_measures.rs[i,52], 
                molecule.mat_FE[i,52], 
                molecule.mat_F[i,52], 
                molecule.mat_E[i,52], 
                molecule.energies[i][0],
                recompE, 
                )
    '''

    print('abs max_decompFE: {} abs min_decompFE: {}'.format(
        max(abs(decompFE)), min(abs(decompFE))))


    Plotter.xy_scatter([dists], [decompFE], [''], ['k'], '$r_{ij} / \AA$',
            'q / kcal/mol', [10], 'scatter-r-decompFE.png')

    Plotter.hist_1d([decompFE], 'q / kcal/mol', 'P(q)', 'hist_q.png')

    Plotter.xy_scatter_density(dists, decompFE, '$r_{ij} / \AA$', 
            'q / kcal/mol', 'density-scatter-r-decompFE.png')

    '''
    network = Network(molecule)
    model = Network.get_coord_FE_model(network, molecule, prescale)

    prediction = model.predict(molecule.coords[0].reshape(1,-1,3))
    print(prediction)
    forces = prediction[0]#[2]
    energies = prediction[2]

    print('C')
    print(molecule.coords[0])
    print('F')
    print(molecule.forces[0])
    print(forces)
    print('E')
    print(molecule.energies[0])
    print(energies)
    '''


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


