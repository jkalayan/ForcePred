#!/usr/bin/env python


from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import sys
#from sys import stdout

kcal2kj = 4.1840000000000005
ang2nm = 1e-1

print(1*(kcal2kj * ang2nm))
print(1/(kcal2kj * ang2nm))
print()

print(1/kilocalories_per_mole/angstrom)
print(1/kilojoules_per_mole/nanometer)
print()

print((1/kilocalories_per_mole/angstrom) * (1/kilojoules_per_mole/nanometer))
print((1*kilocalories_per_mole/angstrom).in_units_of(kilojoules_per_mole/nanometer))
print()


prmtop = AmberPrmtopFile('molecules.prmtop')
inpcrd = AmberInpcrdFile('molecules.inpcrd')
#system = prmtop.createSystem()

pdb = PDBFile('molecules_tleap.pdb')
masses = [6, 6, 8, 1, 1, 1, 8, 1, 1, 1] 
#system = System()
system = System()


for j in range(10):
    #print(j, masses[j])
    system.addParticle(masses[j])


#for j in range(10):
    #system.addForce(0)



#print(system.getForces())
#print(system.getNumParticles())

#'''
force = CustomExternalForce('-fx*x-fy*y-fz*z')
#force = CustomExternalForce('fx fy fz')
system.addForce(force)
force.addPerParticleParameter('fx')
force.addPerParticleParameter('fy')
force.addPerParticleParameter('fz')

for j in range(system.getNumParticles()):
    force.addParticle(j, (0,0,0)*kilocalories_per_mole/angstrom)
#force.addParticle(5, (0,0,0)*kilojoules_per_mole/nanometer)
#force.addParticle(9, (0,0,0)*kilojoules_per_mole/nanometer)
#force.updateParametersInContext(simulation.context)
#'''




#integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
integrator = VerletIntegrator(0.002*picoseconds)
#integrator = VerletIntegrator(0.0005*picoseconds)
#simulation = Simulation(prmtop.topology, system, integrator)
simulation = Simulation(pdb.topology, system, integrator)
#simulation.context.setPositions(inpcrd.positions)
simulation.context.setPositions(pdb.positions)
#if inpcrd.boxVectors is not None:
    #simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output.pdb', 1))
simulation.reporters.append(StateDataReporter('out.csv', 1, step=True, 
        potentialEnergy=True, kineticEnergy=True, temperature=True))
#simulation.step(1000)
#simulation.system.getForces()


#simulation.context.setVelocitiesToTemperature(1*kelvin, 22) #temp, random seed

#PDBFile.writeFile(simulation.topology, positions, open('output2.pdb', 'w'))


#help(simulation.context.setForces)

positions1 = simulation.context.getState(getPositions=True).getPositions(
        asNumpy=True) #in nanometers
forces1 = simulation.context.getState(getForces=True).getForces(asNumpy=True)
        #in kJ/(mole nm)
velocities1 = simulation.context.getState(getVelocities=True).getVelocities(
        asNumpy=True)
        #in nm/ps

#print('forces1', forces1)

open('out-coords.txt', 'w')       
open('out-forces.txt', 'w')             
open('out-velocities.txt', 'w')             
f1 = open('out-coords.txt', 'ab')          
f2 = open('out-forces.txt', 'ab') 
f3 = open('out-velocities.txt', 'ab') 
for i in range(1000):
    #simulation.setParameter()
    #simulation.context.getState(setForces=True).setForces(
            #[Vec3(0, 0, 0) * 10])
    #simulation.context.setForces(forces1) #doesn't work
    #simulation.context.setPositions([Vec3(0, 0, 0) * 10]) #doesn't work
    #simulation.context.setPositions(positions1) #works
    #simulation.context.setVelocities(velocities1) #doesn't work


    positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True) #in nanometers
    forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)
            #in kJ/(mole nm)
    velocities = simulation.context.getState(getVelocities=True).getVelocities(
            asNumpy=True)
            #in nm/ps

    pred_force = (np.array([i,i,i])*kilocalories_per_mole/angstrom).in_units_of(kilojoules_per_mole/nanometer)
    #pred_force = np.array([i,i,i])*kilojoules_per_mole/nanometer
    diff_force = pred_force - forces1[9] 
    print(forces[5], forces[9], diff_force)

    '''
    #print(system.getNumParticles())
    #for j in range(system.getNumParticles()):
    #for j in range(1):
        #force.setParticleParameters(j, j, (10,0,0)*kilojoules_per_mole/nanometer)
    force.setParticleParameters(0, 0, (10,0,0)*kilojoules_per_mole/nanometer)
    force.setParticleParameters(1, 1, (-10,0,0)*kilojoules_per_mole/nanometer)
    #print(system.getNumParticles())
    force.updateParametersInContext(simulation.context)
    '''

    #if i > 9:
    for j in range(system.getNumParticles()):
        #force.setParticleParameters(5, 5, [-4,0,0]*kilojoules_per_mole/nanometer)
        force.setParticleParameters(j,j, diff_force)
        #force.addParticle(5, (2,0,0)*kilojoules_per_mole/nanometer)
    force.updateParametersInContext(simulation.context)

    positions2 = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True) #in nanometers
    forces2 = simulation.context.getState(getForces=True).getForces(asNumpy=True)
            #in kJ/(mole nm)
    velocities2 = simulation.context.getState(getVelocities=True).getVelocities(
            asNumpy=True)
            #in nm/ps



    np.savetxt(f1, positions2.in_units_of(angstrom)) #Ang 
    np.savetxt(f2, forces2.in_units_of(kilocalories_per_mole/angstrom)) #kcal/(mol Ang)
    np.savetxt(f3, velocities2.in_units_of(angstrom/picosecond)) #Ang/ ps
    #print(positions)
    #print(forces)
    #print(velocities)
    #print()
    #PDBFile.writeFile(simulation.topology, positions, open('output2.pdb', 'a'))

    simulation.step(1)
    #simulation.context.setForces(forces1)







'''
print([Vec3(0, 0, 0)] * 10)
print()
print([Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), 
        Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), 
        Vec3(0, 0, 0), Vec3(0, 0, 0)])
'''

#help(simulation.context.getState)
#help(simulation.context.getState(setForces))
#help(setParameter)
#help(Force)
#help(AmberPrmtopFile)
#help(Simulation)
#help(VerletIntegrator)
#help(System)
