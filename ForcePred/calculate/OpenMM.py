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
from keras.models import Model, load_model                                   
import numpy as np


class OpenMM(object):
    '''
    '''
    def __init__(self):
        self.coords = []
        self.forces = []
        self.velocities = []


    def get_system(masses, pdb_file):
        '''setup system for MD with OpenMM'''
        #intial coords need to be in pdb format or amber, gormacs, charmm
        pdb = PDBFile(pdb_file)
        n_atoms = len(pdb.getPositions())
        #create a system of n_atoms with zero intial forces
        system = System()
        for j in range(n_atoms):
            system.addParticle(masses[j])
        force = CustomExternalForce('-fx*x-fy*y-fz*z')
        system.addForce(force)
        force.addPerParticleParameter('fx')
        force.addPerParticleParameter('fy')
        force.addPerParticleParameter('fz')
        for j in range(system.getNumParticles()):
            force.addParticle(j, (0,0,0))
        ## setup simulation conditions
        ts = (0.5*femtoseconds).in_units_of(picoseconds) #1 ps == 1e3 fs
        coupling = 1000 #2 and 100K, 50 and 300K 
        temp = 300
        print('temperature {}K coupling {} ps^-1'.format( temp, coupling))
        integrator = LangevinIntegrator(temp*kelvin, coupling/picoseconds, ts)
        integrator.setRandomNumberSeed(22)
        #integrator = VerletIntegrator(ts)
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.reporters.append(PDBReporter('openmm.pdb', 1))
        simulation.reporters.append(StateDataReporter('openmm.csv', 1, 
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
        ## get initial conditions
        init_forces = simulation.context.getState(
                getForces=True).getForces(asNumpy=True).in_units_of(
                kilocalories_per_mole/angstrom) #in kcal/mol/A
        open('openmm-coords.xyz', 'w').close()
        open('openmm-coords.txt', 'w')       
        open('openmm-forces.txt', 'w')             
        open('openmm-velocities.txt', 'w')  
        open('openmm-delta_energies.txt', 'w') 
        #f1 = open('openmm-coords.txt', 'ab')          
        f2 = open('openmm-forces.txt', 'ab') 
        f3 = open('openmm-velocities.txt', 'ab') 
        for i in range(nsteps):
            system, simulation, force = OpenMM.get_step(system, simulation, 
                    force, init_forces, network, model)
            positions = simulation.context.getState(
                    getPositions=True).getPositions(
                    asNumpy=True).in_units_of(angstrom) #in Ang
            forces = simulation.context.getState(
                    getForces=True).getForces(asNumpy=True).in_units_of(
                    kilocalories_per_mole/angstrom) #in kcal/(mole angstrom)
            velocities = simulation.context.getState(
                    getVelocities=True).getVelocities(
                    asNumpy=True).in_units_of(angstrom/picosecond) #in Ang/ps
            dE = (simulation.integrator.getStepSize() * 
                    np.sum(forces * (velocities)))/kilocalories_per_mole 
                    #kcal/mol

            if i%(nsteps/saved_steps) == 0:
                Writer.write_xyz([positions], atoms, 
                        'openmm-coords.xyz', 'a', i)
                #np.savetxt(f1, positions)
                np.savetxt(f2, forces)
                np.savetxt(f3, velocities)
                open('openmm-delta_energies.txt', 'a').write('{}\n'.format(dE))
            md.coords.append(positions/angstrom)
            md.forces.append(forces/kilocalories_per_mole/angstrom)
        
        return system, simulation, md


    def get_step(system, simulation, force, init_forces, network, model):
        '''Run MD for one step, using predicted forces'''

        positions = simulation.context.getState(
                getPositions=True).getPositions(
                asNumpy=True).in_units_of(angstrom) #in angstrom
        #translate and rotate coords back to center
        positions = Converter.translate_coords(
                positions/angstrom, atoms)
        simulation.context.setPositions(positions)
        ##predict forces here
        pred_forces = OpenMM.predict_force(
                positions, network, model)*kilocalories_per_mole/angstrom
        pred_forces = pred_forces.in_units_of(kilojoules_per_mole/nanometer)
        diff_forces = pred_forces[0] - init_forces
        for j in range(system.getNumParticles()):
            force.setParticleParameters(j,j, diff_forces[j])
        force.updateParametersInContext(simulation.context)
        simulation.step(1)

        return system, simulation, force


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

