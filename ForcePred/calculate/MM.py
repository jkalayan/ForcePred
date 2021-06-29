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
            masses, timestep, temp):
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
        v_scaled, current_T = MM.scale_velocities(v, m, temp, timestep, 0.5)
        #v = v_scaled
        _KE = 0.5 * m * v ** 2
        sum_KE = np.sum(_KE)
        #print('_KE', sum_KE)
        #print('current_T', current_T)
        _Cnew = _C + v * dt + a * dt ** 2
        #print(_Cnew)

        #a = _F / (2 * m)
        #_Cnew = _C + (((_C - _Cprev) / dt) + a * dt) * dt + a * dt ** 2
        #print(_Cnew)

        dE = dt * np.sum(forces * (v / Converter.ang2m))
        return _Cnew / Converter.ang2m, dE, v, current_T, sum_KE


    def scale_velocities(v, m, target_T, dt, tau):
        '''Scale velocities using Berendsen thermostat
        https://www2.mpip-mainz.mpg.de/~andrienk/journal_club/thermostats.pdf
        Don't use leapfrog i.e. dt/2
        '''
        current_T = (m * v ** 2) / (3 * Converter.kB)
        #print('ave_current_T', current_T, np.average(current_T))
        scale_factor = (target_T / current_T) ** 0.5 #simple velocity scaling
        #scale_factor = (1 + dt / tau * (target_T / current_T - 1)) ** 0.5
                #choose how often velocities are scaled with tau value
        v_scaled = v * scale_factor
        return v_scaled, np.average(current_T)


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


