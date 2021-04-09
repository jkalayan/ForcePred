#!/usr/bin/env python

'''
This module is used to store information required to predict forces:
    Nuclear charges, coordinates, forces and molecule energies.

NOT CURRENTLY IN USE
'''




class Molecule(object):

    def __init__(self, filenames, atoms, coords, forces, energies):
        self.filenames = filenames
        self.atoms = atoms
        self.coords = coords
        self.forces = forces
        self.energies = energies

    def __str__(self):
        return ('\nInput files: %s, \natoms: %s, N atoms: %s, ' \
                '\nN structures: %s' % 
                (', '.join(self.filenames), ' '.join(map(str, self.atoms)), 
                len(self.atoms), len(self.coords)))
