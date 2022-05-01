#!/usr/bin/env python

'''
This module is for running MD simulations using OpenMM.
'''

from ..read.Molecule import Molecule
from ..calculate.Converter import Converter
from ..calculate.Conservation import Conservation
from ..write.Writer import Writer
from ..nn.Network_v2 import Network
#from ..nn.openmm_mlpotential import *
#from ..nn.Network import Network
#from simtk.openmm.app import *
#from simtk.openmm import * 
#from simtk.unit import *
from openmm.app import *
from openmm import * 
from simtk.unit import *
import openmmtools
#from tensorflow import keras #newer version
#from tensorflow.keras.models import Model, load_model #newer version 
#import keras
#import tensorflow as tf
#from keras.models import Model, load_model                                   
import numpy as np
import sys


class OpenMM_classical(object):
    '''
    '''
    def __init__(self):
        self.coords = []
        self.forces = []
        self.velocities = []


    def get_system(masses, dihedrals, thetas, k):
        '''setup system for MD with OpenMM'''

        #prmtop = AmberPrmtopFile('data/molecules.prmtop')
        #inpcrd = AmberInpcrdFile('data/molecules.inpcrd')
        prmtop = AmberPrmtopFile('data/molecules-solv.prmtop')
        inpcrd = AmberInpcrdFile('data/molecules-solv.inpcrd')
        system = prmtop.createSystem()

        #dihedrals = [[4,0,1,2], [0,1,2,3]]

        #torsion = CustomTorsionForce('k*((theta-theta0)**2)')
        harmonic_torsion = CustomTorsionForce('0.5*k*min(dtheta, '\
                '2*pi-dtheta)^2; dtheta = abs(theta-theta0); '\
                'pi = 3.1415926535')
        #harmonic_torsion.addGlobalParameter('theta0', theta0)
        #harmonic_torsion.addGlobalParameter('k', k)
        harmonic_torsion.addPerTorsionParameter('theta0')
        harmonic_torsion.addPerTorsionParameter('k')
        system.addForce(harmonic_torsion)
        for i, dih in enumerate(dihedrals):
            harmonic_torsion.addTorsion(dih[0], dih[1], dih[2], dih[3], 
                    [thetas[i], k])

        xml = openmm.XmlSerializer.serialize(system)
        open('mm.xml', 'w').write(xml)

        ## setup simulation conditions
        ts = (2*femtoseconds).in_units_of(picoseconds) #1 ps == 1e3 fs
        coupling = 1 #50 #2 and 100K, 50 and 300K 
        temperature = 300 #300 #500 #300
        #print('temperature {}K coupling {} ps^-1'.format(temperature, 
            #coupling))

        #NVE Verlet only
        #integrator = VerletIntegrator(ts)

        '''
        #NVE Velocity-Verlet
        print('NVE velocity-verlet')
        integrator = openmmtools.integrators.VelocityVerletIntegrator(ts)
        '''

        #'''
        #Langevin
        #print('Langevin thermostat, coupling {}'.format(coupling))
        integrator = LangevinIntegrator(temperature*kelvin, 
                coupling/picoseconds, ts)
        #'''

        '''
        #NoseHoover
        #http://docs.openmm.org/latest/api-c++/generated/NoseHooverIntegrator.html
        chain_length = 5 #n_atoms #default 5, 
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

        #integrator.setRandomNumberSeed(22)
        simulation = Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(inpcrd.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(StateDataReporter('openmm.csv', 1,
                #1000, #
                step=True, potentialEnergy=True, kineticEnergy=True, 
                temperature=True))
        #simulation.context.setVelocitiesToTemperature(1*kelvin, 22) 
                #temp, random seed
        CMMotionRemover(1) #prevents COM of system from drifting

        return system, simulation, integrator


    def run_md(system, simulation, md, integrator, nsteps, saved_steps):
        '''Run MD with OpenMM for nsteps'''

        n_atoms_all = system.getNumParticles()
        n_atoms_model = len(md.atoms)
        #print(n_atoms_all)
        #print(n_atoms_model)
        all_atoms = []
        for n in range(n_atoms_all):
            m = system.getParticleMass(n)
            m = int(round(m._value,0))
            if m != 1:
                m = int(0.5 * m)
            all_atoms.append(m)


        for i in range(nsteps):
            f1 = open('openmm-coords.txt', 'a')          
            f2 = open('openmm-forces.txt', 'a') 
            f3 = open('openmm-velocities.txt', 'a') 
            f4 = open('openmm-delta_energies.txt', 'a')

            time = simulation.context.getState().getTime().in_units_of(
                femtoseconds)
            time = int(time/femtoseconds)
            positions = simulation.context.getState(
                    getPositions=True).getPositions(
                    asNumpy=True).in_units_of(angstrom) #in angstrom
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
                Writer.write_xyz([positions[:n_atoms_model]/angstrom], 
                        md.atoms, 'openmm-coords.xyz', 'a', time)
                np.savetxt(f1, positions[:n_atoms_model])
                np.savetxt(f2, forces[:n_atoms_model])
                np.savetxt(f3, velocities[:n_atoms_model])
                f4.write('{}\n'.format(dE))
                md.coords.append([positions[:n_atoms_model]]/angstrom)
                md.forces.append([forces[:n_atoms_model]]/
                        kilocalories_per_mole/angstrom)
                md.energies.append(dE)
            ##take simulation step
            simulation.step(1)
            sys.stdout.flush()

            f1.close()
            f2.close()
            f3.close()
            f4.close()
        
        return system, simulation, md


