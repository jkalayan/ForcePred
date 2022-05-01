#!/usr/bin/env python

'''
This module is for running MM simulations using OpenMM.
'''

from ..read.Molecule import Molecule
from ..calculate.Converter import Converter
from ..calculate.Conservation import Conservation
from ..write.Writer import Writer
from ..nn.Network_v2 import Network
from ..nn.openmm_mlpotential import *
#from ..nn.Network import Network
#from simtk.openmm.app import *
#from simtk.openmm import * 
#from simtk.unit import *
from openmm.app import *
from openmm import * 
from simtk.unit import *
import openmmtools
from tensorflow import keras #newer version
from tensorflow.keras.models import Model, load_model #newer version 
#import keras
#import tensorflow as tf
#from keras.models import Model, load_model                                   
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

        gaff = False
        mlff = False
        mlmm = True
        force = None

        if mlmm:
            pdb = PDBFile(pdb_file)
            '''
            pdb = PDBFile(pdb_file)
            forcefield = ForceField('openmmforcefields/gaff-edit4.xml', 
                    'openmmforcefields/tip3p_standard.xml')
            [templates, residues] = \
                    forcefield.generateTemplatesForUnmatchedResidues(
                    pdb.topology)
            '''

            '''
            atom_types = np.genfromtxt('data/atom-types2.txt',dtype='str')
            n_atoms_model = len(atom_types)
            print(atom_types)
            for template in templates:
                for i, atom in enumerate(template.atoms):
                    atom.type = atom_types[i]
                forcefield.registerResidueTemplate(template)
            '''
    
            '''
            print(pdb.topology)
            print(pdb.topology.atoms)
            for a in pdb.topology.atoms():
                print(a)
                #a.charge = 0
                #print(a.charge)

            print(residues)
            system = forcefield.createSystem(pdb.topology, 
                    nonbondedMethod=PME, nonbondedCutoff=0.8*nanometer)
            '''

            prmtop = AmberPrmtopFile('../data/molecules.prmtop')
            inpcrd = AmberInpcrdFile('../data/molecules.inpcrd')
            mm_system = prmtop.createSystem()
            print(mm_system.getNumParticles())

            residues = list(prmtop.topology.residues())
            #print(residues)
            ml_atoms = [atom.index for atom in residues[0].atoms()]
            n_atoms_model = len(ml_atoms)
            print(ml_atoms)
            #forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
            #forcefield = ForceField()


            import xml.etree.ElementTree as ET
            xml = openmm.XmlSerializer.serialize(mm_system)
            root = ET.fromstring(xml)
            open('mm.xml', 'w').write(xml)
            #print('\nxml', xml)
            #print('root', root)

            # Create the new System, removing bonded interactions 

            # within the ML subset.
            atomSet = set(ml_atoms)

            def shouldRemove(termAtoms, removeInSet=True):
                return all(a in atomSet for a in termAtoms) == removeInSet

            for bonds in root.findall('./Forces/Force/Bonds'):
                for bond in bonds.findall('Bond'):
                    bondAtoms = [int(bond.attrib[p]) for p in ('p1', 'p2')]
                    if shouldRemove(bondAtoms):
                        bonds.remove(bond)

            for angles in root.findall('./Forces/Force/Angles'):
                for angle in angles.findall('Angle'):
                    angleAtoms = [int(angle.attrib[p]) 
                            for p in ('p1', 'p2', 'p3')]
                    if shouldRemove(angleAtoms):
                        angles.remove(angle)
            for torsions in root.findall('./Forces/Force/Torsions'):
                for torsion in torsions.findall('Torsion'):
                    torsionAtoms = [int(torsion.attrib[p]) 
                            for p in ('p1', 'p2', 'p3', 'p4')]
                    if shouldRemove(torsionAtoms):
                        torsions.remove(torsion)
            '''
            #remove all distance constraints on water distances
            for constraints in root.findall('.Constraints'):
                for constraint in constraints.findall('Constraint'):
                    constraintAtoms = [int(constraint.attrib[p]) 
                            for p in ('p1', 'p2')]
                    if shouldRemove(constraintAtoms) == False:
                        constraints.remove(constraint)
            '''


            ml_system = openmm.XmlSerializer.deserialize(ET.tostring(
                    root, encoding='unicode'))

            # Add nonbonded exceptions and exclusions.

            atomList = list(ml_atoms)
            for force in ml_system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    for i in range(len(atomList)):
                        for j in range(i):
                            force.addException(i, j, 0, 1, 0, True)
                elif isinstance(force, openmm.CustomNonbondedForce):
                    existing = set(tuple(force.getExclusionParticles(i)) 
                            for i in range(force.getNumExclusions()))
                    for i in range(len(atomList)):
                        for j in range(i):
                            if (i, j) not in existing and (j, i) \
                                    not in existing:
                                force.addExclusion(i, j, True)

            xml2 = openmm.XmlSerializer.serialize(ml_system)
            open('ml.xml', 'w').write(xml2)
            #print('\nxml2', xml2)
            system = ml_system

            #print('exit')
            #sys.exit()

            #for ml, use amber top of solute only
            #potential = MLPotential('ani2x')
            #fact = MLPotentialImplFactory()
            #pot = MLPotential.registerImplFactory('pairF-Net', fact)
            #pot.addForces(prmtop.topology, mm_system, ml_atoms, 0)
            #ml_system = MLPotential.createMixedSystem(pot, prmtop.topology, 
                    #mm_system, ml_atoms)



        if mlff:
            #intial coords need to be in pdb format or amber, gromacs, charmm
            pdb = PDBFile(pdb_file)
            n_atoms = len(pdb.getPositions())
            #create a system of n_atoms with zero intial forces
            system = System()
            for j in range(n_atoms):
                #system.addParticle(1)
                system.addParticle(masses[j])


        if gaff:
            ##setup separate system with GAFF parameters
            prmtop = AmberPrmtopFile('molecules.prmtop')
            inpcrd = AmberInpcrdFile('molecules.inpcrd')
            system = prmtop.createSystem()


        if mlff or mlmm:
            ##set initial forces as zero for mlff or keep as is for gaff
            force = CustomExternalForce('-fx*x-fy*y-fz*z')
            system.addForce(force)
            force.addPerParticleParameter('fx')
            force.addPerParticleParameter('fy')
            force.addPerParticleParameter('fz')
            #for j in range(system.getNumParticles()):
            for j in range(len(masses)):
                force.addParticle(j, (0,0,0))


        if mlmm:
            forces = { force.__class__.__name__ : force 
                    for force in system.getForces() }
            nbforce = forces['NonbondedForce']
            #for index in range(nbforce.getNumParticles()-1):
            for index in range(nbforce.getNumExceptions()):
                (p1, p2, q, sig, eps) = nbforce.getExceptionParameters(index)
                #print(index, p1, p2, 1, sig, eps)

            '''
            nb_force = NonbondedForce()
            count = -1
            for k in range(n_atoms_model):
                for l in range(n_atoms_model):
                    count += 1
                    nb_force.setExceptionParameters(index=count, particle1=k, 
                            particle2=l, chargeProd=0, sigma=0, epsilon=0)
            system.addForce(nb_force)
            '''


        ## setup simulation conditions
        ts = (0.5*femtoseconds).in_units_of(picoseconds) #1 ps == 1e3 fs
        coupling = 1 #50 #2 and 100K, 50 and 300K 
        temperature = 300 #300 #500 #300
        print('temperature {}K coupling {} ps^-1'.format(temperature, 
            coupling))

        #NVE Verlet only
        #integrator = VerletIntegrator(ts)

        '''
        #NVE Velocity-Verlet
        print('NVE velocity-verlet')
        integrator = openmmtools.integrators.VelocityVerletIntegrator(ts)
        '''

        #'''
        #Langevin
        print('Langevin thermostat, coupling {}'.format(coupling))
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
        if mlff:
            simulation = Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
        if mlmm:
            simulation = Simulation(prmtop.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
        if gaff:
            simulation = Simulation(prmtop.topology, system, integrator)
            simulation.context.setPositions(inpcrd.positions)
        #simulation.minimizeEnergy()
        #simulation.reporters.append(PDBReporter('openmm.pdb', 1))
        simulation.reporters.append(StateDataReporter('openmm.csv', 1,
                #1000, #
                step=True, potentialEnergy=True, kineticEnergy=True, 
                temperature=True))
        #simulation.context.setVelocitiesToTemperature(1*kelvin, 22) 
                #temp, random seed
        CMMotionRemover(1) #prevents COM of system from drifting

        ## get initial conditions
        init_positions = simulation.context.getState(
                getPositions=True).getPositions(asNumpy=True).in_units_of(
                angstrom) #in angstrom

        init_forces = simulation.context.getState(
                getForces=True).getForces(asNumpy=True).in_units_of(
                kilocalories_per_mole/angstrom) #in kcal/mol/A
        print('init_forces', init_forces[:n_atoms_model])

        return system, simulation, force, integrator, \
                init_positions, init_forces


    def run_md(system, simulation, md, force, integrator, temperature, 
            network, model, nsteps, saved_steps, init_positions, 
            init_forces, prescale):
        '''Run MD with OpenMM for nsteps'''

        mlmm = True
        NVT = False
        if NVT:
            #set temperature for integrator
            integrator.setTemperature(temperature)
            print(integrator.getTemperature())
        equiv_atoms = False
        pairs_dict = None
        if equiv_atoms:
            print('equiv_atoms', equiv_atoms)
            A = Molecule.find_bonded_atoms(md.atoms, init_positions)
            indices, pairs_dict = Molecule.find_equivalent_atoms(
                    md.atoms, A)

        n_atoms_all = system.getNumParticles()
        n_atoms_model = len(md.atoms)
        print(n_atoms_all)
        print(n_atoms_model)
        all_atoms = []
        for n in range(n_atoms_all):
            m = system.getParticleMass(n)
            m = int(round(m._value,0))
            if m != 1:
                m = int(0.5 * m)
            all_atoms.append(m)
        #all_atoms = [1] * n_atoms_all
        #print(all_atoms)


        for i in range(nsteps):
            f1 = open('openmm-coords.txt', 'a')          
            f2 = open('openmm-forces.txt', 'a') 
            f3 = open('openmm-velocities.txt', 'a') 
            f4 = open('openmm-delta_energies.txt', 'a')
            f5 = open('openmm-f-curl.txt', 'a')
            f6 = open('openmm-mm-forces.txt', 'a')

            #print(i)
            time = simulation.context.getState().getTime().in_units_of(
                femtoseconds)
            time = int(time/femtoseconds)# * 2) #fs * 2
            #print(time)
            system, simulation, force, curl, positions, pred_forces, \
                    mm_forces = OpenMM.get_forces(i, system, simulation, 
                    force, md.atoms, init_forces, init_positions, network, 
                    model, prescale, pairs_dict)
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
                if mlmm:
                    Writer.write_xyz([positions/angstrom], all_atoms, 
                            'openmm-coords-all.xyz', 'a', time)
                    Writer.write_xyz([positions[:n_atoms_model]/angstrom], 
                            all_atoms[:n_atoms_model], 
                            'openmm-coords.xyz', 'a', time)
                    np.savetxt(f6, mm_forces[:n_atoms_model])
                if mlmm == False:
                    Writer.write_xyz([positions/angstrom], md.atoms, 
                            'openmm-coords.xyz', 'a', time)
                np.savetxt(f1, positions[:n_atoms_model])
                np.savetxt(f2, forces[:n_atoms_model])
                np.savetxt(f3, velocities[:n_atoms_model])
                f4.write('{}\n'.format(dE))
                f5.write('{}\n'.format(curl))
                md.coords.append([positions[:n_atoms_model]]/angstrom)
                md.forces.append([pred_forces[:n_atoms_model]]/
                        kilocalories_per_mole/angstrom)
                md.energies.append(dE)
            ##take simulation step
            simulation.step(1)
            sys.stdout.flush()

            if mlmm:
                #remove ml contribution, keeping original mm forces
                #after the step is taken
                #n_atoms_model = len(pred_forces)
                minus_ml_forces = forces[:n_atoms_model] - pred_forces 
                for j in range(n_atoms_model):
                    force.setParticleParameters(j,j, [0, 0, 0]) #minus_ml_forces[j]) #
                force.updateParametersInContext(simulation.context)

            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
        
        return system, simulation, md


    def get_forces(time, system, simulation, force, atoms, init_forces, 
            init_coords, network, model, prescale, pairs_dict):
        '''Run MD for one step, using predicted forces'''

        n_atoms_model = len(atoms)
        #print('n_atoms_model', n_atoms_model)
        positions = simulation.context.getState(
                getPositions=True).getPositions(
                asNumpy=True).in_units_of(angstrom) #in angstrom

        mlff = False
        mlmm = True
        equiv_atoms = False
        conservation = False
        from_FE = True
        mm_forces = None

        if mlff or mlmm:
            curl = None
            #'''
            #translate (and rotate if applied) coords back to center
            #positions = OpenMM.translate_coords(init_coords/angstrom,
                    #positions/angstrom, atoms)
            #simulation.context.setPositions(positions*angstrom)
            positions = simulation.context.getState(
                    getPositions=True).getPositions(
                    asNumpy=True).in_units_of(angstrom) #in angstrom
            #'''

            ##predict forces here
            if from_FE:
                pred_forces, curl = OpenMM.predict_force_internal_FE(
                        positions[:n_atoms_model]/angstrom, model)
                #pred_forces, curl = OpenMM.predict_force_from_FE(
                        #positions/angstrom, network, model, prescale)
                #pred_forces2 = OpenMM.predict_force(positions, 
                        #network, model)
                #print('\nA', pred_forces, curl)
                #print('B', pred_forces2)
            else:
                if conservation:
                    pred_forces = OpenMM.predict_scaled_force(
                            positions/angstrom, network, model)
                else:
                    pred_forces = OpenMM.predict_force(positions, 
                            network, model, equiv_atoms, pairs_dict)

            '''
            ##check invariance
            variance, trans, rot = OpenMM.check_invariance(positions/angstrom, 
                    pred_forces[0])
            if variance:
                print('{} variant structure {}, trans: {} rot {}'.format(
                        time, invariance, trans, rot))
                pred_forces = OpenMM.get_conservation(positions/angstrom, 
                        pred_forces, atoms, 
                        network.scale_NRF, network.scale_NRF_min, 
                        network.scale_F, model, molecule=None, 
                        dr=0.001, NRF_scale_method='A', qAB_factor=0)
            '''


            pred_forces = pred_forces*kilocalories_per_mole/angstrom

            forces = simulation.context.getState(
                    getForces=True).getForces(
                    asNumpy=True).in_units_of(kilocalories_per_mole/angstrom) 
                    #in angstrom
            mm_forces = forces[:n_atoms_model]
            print('step', time)
            print('C', positions[:n_atoms_model])
            print('PRED', pred_forces)
            print('MM F', mm_forces)
            print('SUM F', pred_forces + mm_forces)

            ##update simulation forces
            if mlff:
                new_forces = pred_forces[0] - init_forces
            if mlmm:
                new_forces = pred_forces[0] #- mm_forces
            #for j in range(system.getNumParticles()):
            for j in range(n_atoms_model):
                #force.setParticleParameters(j,j, diff_forces[j])
                force.setParticleParameters(j,j, new_forces[j])
            force.updateParametersInContext(simulation.context)

        return system, simulation, force, curl, positions, pred_forces[0], \
                mm_forces


    def predict_force(positions, network, model, equiv_atoms, pairs_dict):
        '''Use NN model to predict forces of current molecule geom'''
        mat_NRF = Network.get_NRF_input([positions], network.atoms, 
                network.n_atoms, network._NC2)
        #print('nrf', mat_NRF)
        if equiv_atoms:
            all_sorted_list, all_resorted_list = Molecule.get_sorted_pairs(
                    mat_NRF, pairs_dict)
            mat_NRF = np.take_along_axis(mat_NRF, all_sorted_list, axis=1)
        mat_NRF_scaled, max_, min_ = \
                Network.get_scaled_values(mat_NRF, 
                #network.scale_NRF, 
                #network.scale_NRF_min, 
                ##hardcoded ismaeels aspirin
                #13036.551114036025, 0,
                ##hardcoded ismaeels malonaldehyde
                12364.334358774337, 0,
                method='A')
        #print('nrf scaled', mat_NRF_scaled)
        prediction_scaled = model.predict(mat_NRF_scaled)
        #print('pred scaled', prediction_scaled)
        prediction = Network.get_unscaled_values(prediction_scaled, 
                #network.scale_F, 
                #network.scale_F_min, 
                ##hardcoded ismaeels aspirin
                #228.19031799443, 0, 
                ##hardcoded ismaeels malonaldehyde
                205.1607192237968, 0,
                method='B')
        #print('pred unscaled', prediction)

        if equiv_atoms:
            mat_NRF = np.take_along_axis(mat_NRF, all_resorted_list, axis=1)
            prediction = np.take_along_axis(prediction, 
                    all_resorted_list, axis=1)
        recomp_forces = Network.get_recomposed_forces([positions], 
                [prediction], network.n_atoms, network._NC2)
        #print('recomp Fs', recomp_forces)
        #sys.exit()

        return recomp_forces

    def predict_scaled_force(positions, network, model):
        forces = None
        molecule = None
        dr = 0.001
        pred_F, scaled_F = Conservation.get_conservation(positions, 
                forces, network.atoms, network.scale_NRF, 0, network.scale_F, 
                model, molecule, dr, NRF_scale_method='A')
        return scaled_F, curl

    def predict_force_from_FE(positions, network, model, prescale):
        molecule = None
        dr = 0.001
        bias_type = '1/r'
        scaled_F, curl = Conservation.get_forces_from_energy(
                positions, network.atoms, 
                network.scale_NRF, network.scale_NRF_min, 
                network.scale_F, model, dr, bias_type, molecule, prescale)
        return scaled_F, curl

    def predict_force_internal_FE(positions, model):
        '''forces are from gradient of energy'''
        prediction = model.predict(positions.reshape(1,-1,3))
        forces = prediction[2]
        return forces, None

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
        variance = False
        if np.all(rotations != 0):
            variance = True
        if np.all(translations != 0):
            variance = True
        return variance, translations, rotations



