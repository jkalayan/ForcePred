#!/usr/bin/env python

'''
This module is for running MM simulations.
'''

import numpy as np
from ..calculate.Converter import Converter

class MM(object):
    '''
    '''
    def __init__(self):
        self.coords = []
        self.forces = []

    def calculate_verlet_step(coords_current, coords_prev, forces, 
            masses, timestep):
        '''Calculate coords for next timestep using Verlet
        algorithm. Coords are assumed to be in Angstrom,
        forces in kcal/(mol Ang), mass in amu and time in 
        femtoseconds'''
        _C = coords_current * Converter.ang2m
        _Cprev = coords_prev * Converter.ang2m
        dt = timestep * Converter.fsec2sec
        _F = (forces * Converter.kcal2kj * 1000) / \
                (Converter.ang2m * Converter._NA)
        m = masses * Converter.au2kg
        a = _F / (2 * m)
        v = ((_C - _Cprev) / dt) + a * dt
        _Cnew = _C + v * dt + a * dt ** 2
        #print(_Cnew)

        #a = _F / (2 * m)
        #_Cnew = _C + (((_C - _Cprev) / dt) + a * dt) * dt + a * dt ** 2
        #print(_Cnew)

        dE = dt * np.sum(forces * (v / Converter.ang2m))
        return _Cnew / Converter.ang2m, dE, v




    #def calculate_step(coords,coords_prev,forces,delta_t,types,timestep):
    def calculate_step(coords,coords_prev,forces,delta_t,weights,timestep):
        #this code assumes that forces are in kcal mol-1 A-1
        #and coords are in A and delta t is in femtoseconds
        C = coords*1*10**-10
        prev_C = coords_prev*1*10**-10
        #weights = np.zeros((len(types),3))
        forces_scaling = 4.184*1000*10**10/(6.02214076*10**23)
        mass_scaling = 1.66053907*10**-27
        #for i in range(0,weights.shape[0]):
            #weights[i,:]=get_mol_weight(types[i])
        time = timestep*10**-15
        dt = delta_t*10**-15
        grad = (forces*forces_scaling)/(2*weights*mass_scaling)
        steps = C + ( ((C-prev_C)/time) + grad*time)*dt + grad*dt**2
        #print('steps', steps)
        steps = steps*10**10 #convert back to angstroms
        return steps


