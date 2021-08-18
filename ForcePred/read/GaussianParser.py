#!/usr/bin/env python

'''
This module is for reading in forces, coordinates and energies from 
Gaussian output files. 
'''

from itertools import islice
import numpy as np
from ..calculate.Converter import Converter
import sys
import warnings

class OPTParser(object):
    '''
    Reads in a list of Gaussian scan output files, finds flags 
    listed below and saves coordinates, forces and energies of 
    optimised structures.
    To get the correctly formatted file, ensure symmetry is on.

    An example Guassian scan instruction:
    # opt=modredundant b3lyp/6-31g(d)
    
    ________________________________________________________________
    Flags in file:
    'NAtoms' = Counting the number of lines containing this tells us 
    how many structures have been generated in the file.
    'Axes restored to original set' = to find where forces are.
    'Input orientation' = using Input orientation coordinates.
        (used for force decomp)
    'Standard orientation' = using Standard orientation coordinates.
    'Optimization completed' = only extract data preceding
        Optimization Completed
    'SCF Done' = to find when to get energy
    ________________________________________________________________
    '''

    def __init__(self, filenames, molecule, opt=True):
        self.filenames = filenames
        self.structures = 0
        self.opt_structures = 0
        self.natoms = None
        self.atoms = []
        self.new_atoms = []
        self.coords = []
        self.std_coords = []
        self.forces = []
        self.energies = []
        self.charges = []
        self.sorted_i = None
        self.iterate_files(self.filenames, opt)
        molecule.get_ZCFE(self) #populate molecule class

    def __str__(self):
        return ('\nGaussian files: %s, \natoms: %s, N atoms: %s, ' \
                '\nN structures: %s, N opt structures: %s,' \
                '\nN inp coords: %s, N std coords: %s, ' \
                '\nN forces %s, N energies %s' % 
                (', '.join(self.filenames), ' '.join(map(str, self.atoms)), 
                self.natoms, self.structures, self.opt_structures,
                len(self.coords), len(self.std_coords),
                len(self.forces), len(self.energies)))
          
    def iterate_files(self, filenames, opt):
        for filename in filenames:
            print(filename)
            input_ = open(filename, 'r')
            inp_coord, std_coord, force, energy, charges = \
                    None, None, None, None, None
            for line in input_:
                #self.get_counts(line)
                self.get_counts2(line, input_)
                if self.natoms != None:
                    if 'Input orientation:' in line:
                        inp_coord = self.clean(self.extract(4, input_), 3) #\
                                #* Converter.au2Ang
                        #print('inp_coord', inp_coord.shape)
                    if 'Standard orientation:' in line:
                        std_coord = self.clean(self.extract(4, input_), 3) #\
                                #* Converter.au2Ang
                        #print('std_coord', std_coord.shape)
                    if 'SCF Done:' in line:
                        energy = float(line.split()[4]) \
                                * Converter.Eh2kcalmol
                        #print('e', energy)
                    if 'Axes restored to original set' in line:
                        force = self.clean(self.extract(4, input_), 3) \
                                * Converter.au2kJmola
                                #* Converter.au2kcalmola
                        #print('f', force.shape)
                    if 'Mulliken charges:' in line:
                        charges = self.clean(self.extract(1, input_), 1)
                    if 'ESP charges:' in line: #override with ESP
                        charges = self.clean(self.extract(1, input_), 1)
                        #charges = np.repeat(charges.reshape(-1,1), 3)
                        #print(charges)

                    #only save info if structure is optimised
                    save_data = False
                    if opt and 'Optimization completed' in line:
                        save_data = True
                    if opt == False and 'Normal termination' in line:
                        save_data = True
                    if save_data and self.atoms == self.new_atoms:
                        self.opt_structures += 1
                        self.coords.append(inp_coord)
                        self.std_coords.append(std_coord)
                        self.forces.append(force)
                        self.energies.append(energy)
                        self.charges.append(charges)
            #print()
            sys.stdout.flush()
            if self.atoms == self.new_atoms:
                self.new_atoms = []
            else:
                warnings.warn('Structure %s in file %s is different. '\
                        'Saved structure is %s - ensure all files contain '\
                        'this structure.' 
                        % (self.new_atoms, filename, self.atoms))

    def get_counts(self, line):
        if 'NAtoms=' in line:
            self.structures += 1
            if self.natoms == None:
                self.natoms = int(line.split()[1])

    def get_counts2(self, line, input_):
        '''find atom count by counting num atoms after Z-matrix print'''
        if 'Symbolic Z-matrix:' in line:
            self.structures += 1
            atom_count = 0
            input_.readline() #ignore this line
            for line in input_:
                line2 = line.strip('\n').split()
                if len(line2) > 0:
                    atom = line2[0]
                    if len(atom) < 3 and len(atom) > 0: #atom symbols only
                        atom_count += 1
                    else:
                        break
            self.natoms = atom_count

    def extract(self, padding, input_):
        return (list(islice(input_, padding + 
                self.natoms))[-self.natoms:])

    def clean(self, raw, num_cols):
        cleaned = np.empty(shape=[self.natoms, num_cols])
        for i, atom in enumerate(raw):
            cleaned[i] = atom.strip('\n').split()[-num_cols:]
        #get the list of nuclear charges in a molecule
        if len(self.atoms) == 0:
            self.get_atoms(raw, self.atoms)
        if len(self.new_atoms) == 0:
            self.get_atoms(raw, self.new_atoms)
        #print(cleaned.shape)
        return np.array(cleaned)

    def get_atoms(self, raw, atoms):
        for atom in raw:
            atoms.append(int(atom.strip('\n').split()[1]))

     

