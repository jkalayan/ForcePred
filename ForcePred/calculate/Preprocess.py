#!/usr/bin/env python

'''
This module is for preprocessing forces and energies, ready for
decompostion
'''

import math
import numpy as np
import sys
from ..calculate.Converter import Converter
#from ..network.Network import Network

class Preprocess(object):
    '''
    Takes coords, forces, atoms from Molecule class 
    and scales as required
    '''

    def __init__(self):
        pass

    def __str__(self):
        return ("\nPreprocessing class")
    

    def prescale_energies(molecule, prescale, train_idx):
        """Scale energies using min/max forces in training set"""
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
    

    def preprocessFE(molecule, n_training, n_val, n_test, bias):
        """
        
        """
        prescale = [0, 1, 0, 1, 0, 0]
        #n_training = 10000
        #n_val = 50
        if n_test == -1 or n_test > len(molecule.coords):
            n_test = len(molecule.coords) - n_training
        print('\nn_training', n_training, '\nn_val', n_val,
                '\nn_test', n_test, '\nbias', bias)

        # save original energies first, before scaling by forces
        molecule.orig_energies = np.copy(molecule.energies)

        #bias = '1/r' #default for now
        extra_cols = 0 #n_atoms
        pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                pairs.append([i, j])

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
        prescale = Preprocess.prescale_energies(molecule, prescale, molecule.train)
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

        return molecule
    
