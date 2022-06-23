#!/usr/bin/env python

__author__ = 'Jas Kalayan'
__license__ = 'GPL'
__maintainer__ = 'Jas Kalayan'
__email__ = 'jkalayan@gmail.com'
__status__ = 'Development'

from datetime import datetime
import argparse
import sys

import MDAnalysis       
from MDAnalysis import *
import numpy as np

from ForcePred.calculate.RAD import *


def run_force_pred(file_topology='file_topology', file_coords='file_coords', 
        file_forces='file_forces', file_energies='file_energies', 
        list_files='list_files'):

    startTime = datetime.now()


    #topo = Universe(file_topology, file_coords, format='TRJ')
    #print(topo.atoms)
    pdbtrj = 'traj_RAD_bfac.pdb'
    coords = Universe(file_topology, file_coords, format='DCD')
    # dynamically add new attributes
    # ('tempfactors' is pre-defined and filled with zeros as default values)
    coords.add_TopologyAttr('tempfactors')
    #coords = Universe(file_topology, file_coords, format='TRJ')
    print(coords.trajectory)
    #sys.exit()
    #forces = Universe(file_topology, file_forces, format='TRJ')
    #print(forces.trajectory[0].positions)
    #print(forces.atoms)
    with MDAnalysis.Writer(pdbtrj, multiframe=True, bonds=None, 
            n_atoms=coords.atoms.n_atoms) as PDB:
        for ts1 in coords.trajectory:
            print('\n', coords.trajectory.time) 
            '''
            print(coords.trajectory.time, 
                    forces.atoms.select_atoms('name H*').masses[5:10], 
                    forces.atoms.select_atoms('mass 1 to 1.1').names[5:10], 
                    coords.select_atoms('name H*').positions[5:10])
            '''

            #hydrogen_coords = coords.select_atoms('mass 1 to 1.1').positions
            #neighbour_coords = ts1.positions

            atom = coords.select_atoms('mass 2 to 999 and not resname WAT')
            atom_coords = atom.positions
            #print(atom_coords.shape)
            #print(atom.names, atom.indices)

            neighbours = coords.select_atoms(
                    'mass 2 to 999 and not index 0 and '\
                    'not bonded index 0')
            neighbour_coords = neighbours.positions
            #print(neighbours.names)
            #print(neighbours.indices)
            RAD_all = []
            for i in atom.indices:
                RAD_all = getUALevelRAD(atom.indices[i], atom.positions[i], 
                        neighbours, 
                        coords.select_atoms('all'), 
                        coords.dimensions, RAD_all)

            #print(RAD_all)
            #print(coords.atoms.tempfactors, coords.atoms.tempfactors.shape)

            zeros = np.zeros((coords.atoms.n_atoms))
            #print(zeros)
            zeros[RAD_all] = 1
            coords.atoms.tempfactors = zeros
            #print(coords.atoms.tempfactors)
            #RAD_atoms = coords.select_atoms('index {}'.format(RAD_all))
            PDB.write(coords.atoms)

            '''
            atom = coords.select_atoms('mass 1 to 1.1 and charge not 0 '\
                    'and charge 0 to 9e9')
            getHB(atom.indices[0], atom.positions[0], neighbours, 
                    coords.select_atoms('all'), coords.dimensions)
            '''
            #sys.exit()
            sys.stdout.flush()


    sys.stdout.flush()
    print('end')
    print(datetime.now() - startTime)


def main():

    try:
        usage = 'runForcePred_RAD.py [-h]'
        parser = argparse.ArgumentParser(description='Program for reading '\
                'in molecule forces, coordinates and energies for '\
                'entropy calculations.', usage=usage, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group('Options')
        group = parser.add_argument('-t', '--file_topology', 
                metavar='file', default=None,
                help='name of file containing system topology.')
        group = parser.add_argument('-c', '--file_coords', 
                metavar='file', default=None,
                help='name of file containing coordinates.')
        group = parser.add_argument('-f', '--file_forces', 
                metavar='file', default=None,
                help='name of file containing forces.')
        group = parser.add_argument('-e', '--file_energies', 
                metavar='file', default=None,
                help='name of file containing energies.')
        group = parser.add_argument('-l', '--list_files', action='store', 
                metavar='file', default=False,
                help='file containing list of file paths.')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    run_force_pred(file_topology=op.file_topology, file_coords=op.file_coords,
            file_forces=op.file_forces, file_energies=op.file_energies, 
            list_files=op.list_files)

if __name__ == '__main__':
    main()


