#!/usr/bin/env python

'''
This module is for reading in forces, coordinates and energies from 
Gaussian output files. 
'''

from itertools import islice
import numpy as np


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

    def __init__(self, filenames, molecule):
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
        self.sorted_i = None
        self.iterate_files(self.filenames)
        self.sort_by_energy()
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
          
    def iterate_files(self, filenames):
        for filename in filenames:
            input_ = open(filename, 'r')
            inp_coord, std_coord, force, energy = None, None, None, None
            for line in input_:
                self.get_counts(line)
                if self.natoms != None:
                    if 'Input orientation:' in line:
                        inp_coord = self.clean(self.extract(4, input_)) 
                    if 'Standard orientation:' in line:
                        std_coord = self.clean(self.extract(4, input_))
                    if 'SCF Done:' in line:
                        energy = float(line.split()[4])
                    if 'Axes restored to original set' in line:
                        force = self.clean(self.extract(4, input_))
                    #only save info if structure is optimised
                    if 'Optimization completed' in line:
                        self.opt_structures += 1
                        self.coords.append(inp_coord)
                        self.std_coords.append(std_coord)
                        self.forces.append(force)
                        self.energies.append(energy)
            if self.atoms == self.new_atoms:
                self.new_atoms = []
            else:
                raise ValueError('Structure %s in file %s is different. '\
                        'Saved structure is %s - ensure all files contain '\
                        'this structure.' 
                        % (self.new_atoms, filename, self.atoms))

    def get_counts(self, line):
        if 'NAtoms=' in line:
            self.structures += 1
            if self.natoms == None:
                self.natoms = int(line.split()[1])

    def extract(self, padding, input_):
        return (list(islice(input_, padding + 
                self.natoms))[-self.natoms:])

    def clean(self, raw):
        cleaned = np.empty(shape=[self.natoms, 3])
        for i, atom in enumerate(raw):
            cleaned[i] = atom.strip('\n').split()[-3:]
        #get the list of nuclear charges in a molecule
        if len(self.atoms) == 0:
            self.get_atoms(raw, self.atoms)
        if len(self.new_atoms) == 0:
            self.get_atoms(raw, self.new_atoms)
        return cleaned

    def get_atoms(self, raw, atoms):
        for atom in raw:
            atoms.append(int(atom.strip('\n').split()[1]))

    def sort_by_energy(self):
        self.energies = np.array(self.energies)
        self.sorted_i = np.argsort(self.energies)
        self.energies = self.energies[self.sorted_i]
        self.coords = self.order_array(self.coords, 
                self.sorted_i)
        self.std_coords = self.order_array(self.std_coords, self.sorted_i)
        self.forces = self.order_array(self.forces, self.sorted_i)
        return self

    def order_array(self, np_list, sorted_i):
        np_list = self.get_3D_array(np_list)
        np_list = np_list[sorted_i]
        return np_list

    def get_3D_array(self, np_list):
        np_list = np.reshape(np.vstack(np_list), (-1,self.natoms,3))
        return np_list

    def get_2D_array(self, np_list):
        np_list = np.vstack(np_list)
        return np_list        

