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
    from ForcePred.calculate.OpenMM_classical import OpenMM_classical

mdanal = False
if mdanal:
    from ForcePred.read.AMBLAMMPSParser import AMBLAMMPSParser
    from ForcePred.read.AMBERParser import AMBERParser
                               
from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, XYZParser, Binner, Writer, Plotter, Network, Conservation

from keras.models import Model, load_model    
from keras import backend as K                                              
import sys
#import numpy as np
#from itertools import islice

#import os
#os.environ['OMP_NUM_THREADS'] = '8'


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
    XYZParser(atom_file, coord_files, [], [], molecule)
    #NPParser(atom_file, coord_files, force_files, energy_files, molecule)
    #NPParser(atom_file, coord_files, force_files, [], molecule)
    #NPParser(atom_file, [coord_files[0]], [force_files[0]], 
            #[energy_files[0]], molecule)

    A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    indices, pairs_dict = Molecule.find_equivalent_atoms(molecule.atoms, A)
    print('\nZ', molecule.atoms)
    print('A', A) 
    print('indices', indices)
    print('pairs_dict', pairs_dict)
    print()
    sys.stdout.flush()


    masses = [Converter._ZM[i] for i in molecule.atoms]

    dihedrals = [[4,0,1,2], [0,1,2,3]] #both OCCC dihs
    measures = Binner()
    measures.get_dih_pop(molecule.coords, dihedrals)
    #print(measures.phis.T[0])
    #print(measures.phis.T[1])
    #print(measures.phis)
    #print(np.radians(measures.phis))
    Plotter.xy_scatter([measures.phis.T[0]], [measures.phis.T[1]], 
            ['omega confs'], ['k'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', 
            [5], 
            'dihs-openbabel.png'
            #'dihs-openmm.png'
            )

    #Plotter.hist_2d([measures.phis.T[0]], [measures.phis.T[1]], 
            #['Greys'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', 
            #'#hist2d-dihs.png')
    print(datetime.now() - startTime)
    #sys.exit()

    #thetas = list(range(-180, 190, 10))
    #thetas = [0]
    k = 100
    #print('theta0s:', thetas)
    print('k:', k)

    '''
    ##write new files here first
    open('openmm-coords.txt', 'w').close() #needed to continue sims
    open('openmm-coords.xyz', 'w').close()
    open('openmm-forces.txt', 'w').close()
    open('openmm-velocities.txt', 'w').close()
    open('openmm-delta_energies.txt', 'w').close()
    '''

    #setup object for collecting simulation results
    md = Molecule()
    md.atoms = molecule.atoms
    md.coords = []
    md.forces = []
    md.energies = []

    for sample, thetas in enumerate(measures.phis):

        ''' #Don't really need this
        print('\nget initial pdb structure')
        pdb_file = 'molecules.pdb'
        Writer.write_pdb(molecule.coords[sample], 'MOL', 1, 
                molecule.atoms, pdb_file, 'w')
        masses = [Converter._ZM[i] for i in molecule.atoms]
        sys.stdout.flush()
        '''

        print('SAMPLE:', sample, thetas)
        thetas = np.radians(thetas)
        #print('\nget MD system')
        system, simulation, integrator = OpenMM_classical.get_system(
                masses, dihedrals, thetas, k)
        #print(datetime.now() - startTime)
        sys.stdout.flush()


        #print('\nrun MD simulation')
        nsteps = 2000 #20000 #10ps
        saved_steps = nsteps#/10
        #print('nsteps: {} saved_steps: {}'.format(nsteps, saved_steps))

        system, simulation, md = OpenMM_classical.run_md(system, simulation, 
                md, integrator, nsteps, saved_steps)

        print(datetime.now() - startTime)
        sys.stdout.flush()


    md.coords = md.get_3D_array([md.coords])
    md.forces = md.get_3D_array([md.forces])
    md.energies = np.array(md.energies)

    print('\nget MD dihs')
    measures2 = Binner()
    measures2.get_dih_pop(md.coords, dihedrals)
    Plotter.xy_scatter([measures2.phis.T[0]], [measures2.phis.T[1]], 
            ['openmm confs'], ['k'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', 
            [5], 'dihs-openmm.png')
    Plotter.hist_2d([measures.phis.T[0]], [measures.phis.T[1]], 
            ['Greys'], '$\zeta_1$ (deg)', '$\zeta_2$ (deg)', 
            'hist2d-dihs.png')

    print('\nchecking invariance of forces')
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


