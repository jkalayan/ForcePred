#!/usr/bin/env python

'''
This module is for running MM simulations using OpenMM.
'''

from ..read.Molecule import Molecule
from ..calculate.Converter import Converter
from ..write.Writer import Writer
from ..nn.Network import Network
from simtk.openmm.app import *
from simtk.openmm import * 
from simtk.unit import *
import openmmtools
from keras.models import Model, load_model                                   
import numpy as np
import sys


class OpenMM(object):
    '''
    '''
    def __init__(self):
        self.coords = []
        self.forces = []
        self.velocities = []


    def get_system(masses, pdb_file):
        '''setup system for MD with OpenMM'''

        gaff = True
        mlff = False

        if mlff:
            #intial coords need to be in pdb format or amber, gromacs, charmm
            pdb = PDBFile(pdb_file)
            n_atoms = len(pdb.getPositions())
            #create a system of n_atoms with zero intial forces
            system = System()
            for j in range(n_atoms):
                system.addParticle(masses[j])


        if gaff:
            ##setup separate system with GAFF parameters
            prmtop = AmberPrmtopFile('molecules.prmtop')
            inpcrd = AmberInpcrdFile('molecules.inpcrd')
            system = prmtop.createSystem()

        ##set initial forces as zero for mlff or keep as is for gaff
        force = CustomExternalForce('-fx*x-fy*y-fz*z')
        system.addForce(force)
        force.addPerParticleParameter('fx')
        force.addPerParticleParameter('fy')
        force.addPerParticleParameter('fz')
        for j in range(system.getNumParticles()):
            force.addParticle(j, (0,0,0))

        ## setup simulation conditions
        ts = (0.5*femtoseconds).in_units_of(picoseconds) #1 ps == 1e3 fs
        coupling = 1 #2 and 100K, 50 and 300K 
        temperature = 300
        print('temperature {}K coupling {} ps^-1'.format(temperature, 
            coupling))

        #NVE Verlet only
        #integrator = VerletIntegrator(ts)

        #NVE Velocity-Verlet
        #integrator = openmmtools.integrators.VelocityVerletIntegrator(ts)

        #Langevin
        print('Langevin thermostat, coupling {}'.format(coupling))
        integrator = LangevinIntegrator(temperature*kelvin, 
                coupling/picoseconds, ts)

        '''
        #NoseHoover
        #http://docs.openmm.org/latest/api-c++/generated/NoseHooverIntegrator.html
        chain_length = 2 #default 5, 
                #number of beads in the Nose-Hoover chain
        collision_frequency = (50)/picoseconds #default is 50
                #freq of interaction with the heat bath
        num_mts = 5 #default 5
                #number of step in the multiple timestep chain 
                #propagation algorithm
        num_yoshidasuzuki = 5 #1 #default 5
                #num terms in the Yoshida-Suzuki multi timestep
                #decomposition used in the chain propagation algorithm
                #must be 1,3 or 5
        print('NHC thermostat, chain_length {}, collision_freq {}, '\
                'num_mts {}, num_yosh {}'.format(chain_length, 
                collision_frequency, num_mts, num_yoshidasuzuki))
        integrator = openmmtools.integrators.NoseHooverChainVelocityVerletIntegrator(
                system, temperature*kelvin, collision_frequency, ts, 
                chain_length, num_mts, num_yoshidasuzuki)
        '''

        '''
        #BAOAB
        integrator = openmmtools.integrators.GeodesicBAOABIntegrator(
                K_r=2, temperature=temperature, 
                collision_rate=(1*1000)/picoseconds, timestep=ts)
        '''

        integrator.setRandomNumberSeed(22)
        if mlff:
            simulation = Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
        if gaff:
            simulation = Simulation(prmtop.topology, system, integrator)
            simulation.context.setPositions(inpcrd.positions)
        #simulation.minimizeEnergy()
        #simulation.reporters.append(PDBReporter('openmm.pdb', 1))
        simulation.reporters.append(StateDataReporter('openmm.csv', 1000, 
                step=True, potentialEnergy=True, kineticEnergy=True, 
                temperature=True))
        #simulation.context.setVelocitiesToTemperature(1*kelvin, 22) 
                #temp, random seed
        CMMotionRemover(1) #prevents COM of system from drifting

        return system, simulation, force


    def run_md(system, simulation, force, atoms, network, model, nsteps,
            saved_steps):
        '''Run MD with OpenMM for nsteps'''
        #setup object for collecting simulation results
        md = Molecule()
        md.atoms = atoms
        md.coords = []
        md.forces = []
        md.energies = []
        ## get initial conditions
        init_positions = simulation.context.getState(
                getPositions=True).getPositions(asNumpy=True).in_units_of(
                angstrom) #in angstrom
        init_forces = simulation.context.getState(
                getForces=True).getForces(asNumpy=True).in_units_of(
                kilocalories_per_mole/angstrom) #in kcal/mol/A
        open('openmm-coords.xyz', 'w').close()
        #open('openmm-coords.txt', 'w')       
        open('openmm-forces.txt', 'w')             
        open('openmm-velocities.txt', 'w')  
        open('openmm-delta_energies.txt', 'w') 
        #f1 = open('openmm-coords.txt', 'ab')          
        f2 = open('openmm-forces.txt', 'ab') 
        f3 = open('openmm-velocities.txt', 'ab') 
        for i in range(nsteps):
            system, simulation, force, positions = OpenMM.get_forces(
                    i, system, simulation, 
                    force, atoms, init_forces, init_positions, network, model)
            velocities = simulation.context.getState(
                    getVelocities=True).getVelocities(
                    asNumpy=True).in_units_of(angstrom/picosecond) #in Ang/ps
            forces = simulation.context.getState(
                    getForces=True).getForces(asNumpy=True).in_units_of(
                    kilocalories_per_mole/angstrom) #in kcal/(mole angstrom)
            dE = (simulation.integrator.getStepSize() * np.sum(forces * 
                    (velocities)))/kilocalories_per_mole 
                    #kcal/mol

            if i%(nsteps/saved_steps) == 0:
                Writer.write_xyz([positions/angstrom], atoms, 
                        'openmm-coords.xyz', 'a', i)
                #np.savetxt(f1, positions)
                np.savetxt(f2, forces)
                np.savetxt(f3, velocities)
                open('openmm-delta_energies.txt', 'a').write('{}\n'.format(dE))
            md.coords.append(positions/angstrom)
            md.forces.append(forces/kilocalories_per_mole/angstrom)
            md.energies.append(dE)
            ##take simulation step
            simulation.step(1)
            sys.stdout.flush()
        
        return system, simulation, md


    def get_forces(time, system, simulation, force, atoms, init_forces, 
            init_coords, network, model):
        '''Run MD for one step, using predicted forces'''

        positions = simulation.context.getState(
                getPositions=True).getPositions(
                asNumpy=True).in_units_of(angstrom) #in angstrom

        mlff = False
        if mlff:
            #'''
            #translate (and rotate if applied) coords back to center
            positions = OpenMM.translate_coords(init_coords/angstrom,
                    positions/angstrom, atoms)
            simulation.context.setPositions(positions*angstrom)
            positions = simulation.context.getState(
                    getPositions=True).getPositions(
                    asNumpy=True).in_units_of(angstrom) #in angstrom
            #'''
            ##predict forces here
            pred_forces = OpenMM.predict_force(positions, network, model)
            '''
            ##check invariance
            invariance, trans, rot = OpenMM.check_invariance(positions/angstrom, 
                    pred_forces[0])
            if invariance:
                print('{} invariant structure {}, trans: {} rot {}'.format(
                        time, invariance, trans, rot))
                pred_forces = OpenMM.get_conservation(positions/angstrom, 
                        pred_forces, atoms, 
                        network.scale_NRF, network.scale_NRF_min, 
                        network.scale_F, model, molecule=None, 
                        dr=0.001, NRF_scale_method='A', qAB_factor=0)
            '''
            pred_forces = pred_forces*kilocalories_per_mole/angstrom
            ##update simulation forces
            diff_forces = pred_forces[0] - init_forces
            for j in range(system.getNumParticles()):
                force.setParticleParameters(j,j, diff_forces[j])
            force.updateParametersInContext(simulation.context)

        return system, simulation, force, positions


    def predict_force(positions, network, model):
        '''Use NN model to predict forces of current molecule geom'''
        mat_NRF = Network.get_NRF_input([positions], network.atoms, 
                network.n_atoms, network._NC2)
        mat_NRF_scaled, max_, min_ = \
                Network.get_scaled_values(mat_NRF, network.scale_NRF, 
                network.scale_NRF_min, method='A')
        prediction_scaled = model.predict(mat_NRF_scaled)
        prediction = Network.get_unscaled_values(prediction_scaled, 
                network.scale_F, network.scale_F_min, method='B')
        recomp_forces = Network.get_recomposed_forces([positions], 
                [prediction], network.n_atoms, network._NC2)

        return recomp_forces

    def translate_coords(init_coords, coords, atoms):
        n_atoms = len(atoms)
        masses = np.array([Converter._ZM[a] for a in atoms])
        _PA, _MI, com = Converter.get_principal_axes(coords, masses)
        c_translated = np.zeros((n_atoms,3))
        c_rotated = np.zeros((n_atoms,3))
        for i in range(n_atoms):
            c_translated[i] = coords[i] - com
            for j in range(3):
                c_rotated[i][j] = np.dot(c_translated[i], _PA[j])
        return c_translated
        #return c_rotated


    def check_invariance(coords, forces):
        ''' Ensure that forces in each structure translationally and 
        rotationally sum to zero (energy is conserved) i.e. translations
        and rotations are invariant. '''
        #translations
        translations = []
        trans_sum = np.sum(forces,axis=0) #sum columns
        for x in range(3):
            x_trans_sum = abs(round(trans_sum[x], 0))
            translations.append(x_trans_sum)
        translations = np.array(translations)
        #rotations
        cm = np.average(coords, axis=0)
        i_rot_sum = np.zeros((3))
        for i in range(len(forces)):
            r = coords[i] - cm
            diff = np.cross(r, forces[i])
            i_rot_sum = np.add(i_rot_sum, diff)
        rotations = np.round(np.abs(i_rot_sum), 0)
        #check if invariant
        invariance = False
        if np.all(rotations != 0):
            invariance = True
        if np.all(translations != 0):
            invariance = True
        return invariance, translations, rotations

