#!/usr/bin/env python

'''
This module is for running MM simulations.
'''

import numpy as np
from ..calculate.Converter import Converter
from ..calculate.Binner import Binner

class MM(object):
    '''
    '''
    def __init__(self):
        self.coords = []
        self.forces = []

    def calculate_verlet_step(coords_current, coords_prev, forces, 
            masses, timestep, dt, temp):
        '''Calculate coords for next timestep using Verlet
        algorithm. Coords are assumed to be in Angstrom,
        forces in kcal/(mol Ang), mass in amu and time in 
        femtoseconds'''
        _C = coords_current * Converter.ang2m
        _Cprev = coords_prev * Converter.ang2m
        timestep = timestep * Converter.fsec2sec
        dt = dt * Converter.fsec2sec
        _F = (forces * Converter.kcal2kj * 1000) / \
                (Converter.ang2m * Converter._NA)
        m = masses * Converter.au2kg
        a = _F / (2 * m)
        v = ((_C - _Cprev) / timestep) + a * timestep #?? eq 5
        v_scaled, current_T = MM.scale_velocities(v, m, temp, timestep, 0.5)
        #v = v_scaled
        _KE = 0.5 * m * v ** 2
        sum_KE = np.sum(_KE)
        #print('_KE', sum_KE)
        #print('current_T', current_T)
        _Cnew = _C + v * dt + a * dt ** 2 ## eq 1, 3=verlet
        #print(_Cnew)

        #a = _F / (2 * m)
        #_Cnew = _C + (((_C - _Cprev) / dt) + a * dt) * dt + a * dt ** 2
        #print(_Cnew)

        dE = timestep * np.sum(forces * (v / Converter.ang2m))
        return _Cnew / Converter.ang2m, dE, v, current_T, sum_KE

    #def calculate_step(coords,coords_prev,forces,delta_t,types,timestep):
    def calculate_step(coords,coords_prev,forces,delta_t,weights,timestep):
        '''
        Not used anymore, replaced by OpenMM, was used to for MD with 
        predicted forces.
        this code assumes that forces are in kcal mol-1 A-1
        and coords are in A and delta t is in femtoseconds.
        '''
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


    def use_gaff(coords):
        '''
        Never used, Amber and OpenMM used instead for GAFF implementation
        CHARGES (E = q1*q2/r^2, where E in kcal/mol, r in Ang)
        C   = 1.99351962E+00
        O   = -1.10390693E+01
        HC  = 7.82647785E-01
        HO  = 7.47114300E+00

        MASSES
        C   = 1.20100000E+01
        O   = 1.60000000E+01
        H   = 1.00800000E+00

        BOND FORCE CONSTANTS (kcal/(mol Ang^2))
        HO_OH   = 3.71400000E+02
        C3_OH   = 3.16700000E+02
        C3_H1   = 3.30600000E+02
        C3_C3   = 3.00900000E+02

        EQUILIBRIUM BOND LENGTH (Ang)
        HO_OH   = 9.73000000E-01 
        C3_OH   = 1.42330000E+00
        C3_H1   = 1.09690000E+00
        C3_C3   = 1.53750000E+00

        ANGLE FORCE CONSTANTS (kcal/(mol radian^2))
        H1_C3_H1    = 3.92000000E+01
        H1_C3_OH    = 5.09000000E+01
        C3_C3_OH    = 6.75000000E+01
        C3_C3_H1    = 4.64000000E+01
        C3_OH_HO    = 4.74000000E+01 

        ANGLE EQUILIBRIUM VALUE (radian)
        H1_C3_H1    = 1.89298492E+00
        H1_C3_OH    = 1.92440086E+00
        C3_C3_OH    = 1.92317913E+00
        C3_C3_H1    = 1.91218355E+00
        C3_OH_HO    = 1.87204096E+00

        DIHEDRAL FORCE CONSTANT (kcal/mol)
        HC_C3_C3_C3 = 1.66666667E-01
        H1_C3_C3_OH = 2.50000000E-01
        H1_C3_C3_OH = 0.00000000E+00
        HC_C3_C3_HC = 1.55555556E-01
        OH_C3_C3_OH = 1.17500000E+00
        OH_C3_C3_OH = 1.44000000E-01 
        HO_OH_C3_C3 = 1.60000000E-01

        DIHEDRAL_PERIODICITY
        HC_C3_C3_C3 = 3.00000000E+00
        H1_C3_C3_OH = 1.00000000E+00
        H1_C3_C3_OH = 3.00000000E+00
        HC_C3_C3_HC = 3.00000000E+00
        OH_C3_C3_OH = 2.00000000E+00
        OH_C3_C3_OH = 3.00000000E+00 
        HO_OH-C3_C3 = 3.00000000E+00

        ATOM TYPE INDEX
        C0  = 1
        C1  = 1
        O2  = 2
        H3  = 3
        H4  = 3
        H5  = 4
        O6  = 2 
        H7  = 3
        H8  = 3
        H9  = 4

        LENNARD JONES ACOEF
        A_1_1 = 1.04308023E+06
        A_1_2 = 7.91544157E+05
        A_1_3 = 5.81803229E+05
        A_1_4 = 6.78771368E+04
        A_2_2 = 4.66922514E+04
        A_2_3 = 3.25969625E+03
        A_2_4 = 0.00000000E+00
        A_3_3 = 0.00000000E+00
        A_3_4 = 0.00000000E+00
        A_4_4 = 0.00000000E+00

        LENNARD JONES BCOEF
        B_1_1 = 6.75612247E+02
        B_1_2 = 6.93079947E+02
        B_1_3 = 6.99746810E+02
        B_1_4 = 1.06076943E+02
        B_2_2 = 1.03606917E+02
        B_2_3 = 1.43076527E+01
        B_2_4 = 0.00000000E+00
        B_3_3 = 0.00000000E+00
        B_3_4 = 0.00000000E+00
        B_4_4 = 0.00000000E+00

        RADII
        R_C   = 1.70000000E+00 
        R_O   = 1.50000000E+00
        R_HC  = 1.30000000E+00
        R_HO  = 8.00000000E-01

        '''

        atom_type_indices = {
                0:1, 1:1, 2:2, 3:3, 4:3, 5:4, 6:2, 7:3, 8:3, 9:4
                } #1 = C, 2 = O, 3 = HC, 4 = HO

        bonded_list = [[0,1], [0,2], [0,3], [0,4], 
                [1,6], [1,7], [1,8], 
                [2,5],
                [6,9]]

        bond_force_constants = {
                (1,1): 3.00900000E+02,  #C_C
                (1,2): 3.16700000E+02,  #C_O
                (1,3): 3.30600000E+02,  #C_HC
                (2,4): 3.71400000E+02   #O_HO
                }

        bond_eq_lengths = {
                (1,1): 1.53750000E+00,  #C_C
                (1,2): 1.42330000E+00,  #C_O
                (1,3): 1.09690000E+00,  #C_HC
                (2,4): 9.73000000E-01   #O_HO
                }

        angle_list = [[1,0,2], [1,0,3], [1,0,4], [2,0,3], [2,0,4], [3,0,4],
                [0,1,6], [0,1,7], [0,1,8], [6,1,7], [6,1,8], [7,1,8],
                [0,2,5],
                [1,6,9]]

        angle_force_constants = {
                (1,3,3): 3.92000000E+01,    #H1_C3_H1 
                (1,2,3): 5.09000000E+01,    #H1_C3_OH
                (1,1,2): 6.75000000E+01,    #C3_C3_OH
                (1,1,3): 4.64000000E+01,    #C3_C3_H1 
                (1,2,4): 4.74000000E+01     #C3_OH_HO
                }

        angle_eq_values = {
                (1,3,3): 1.89298492E+00,    #H1_C3_H1 
                (1,2,3): 1.92440086E+00,    #H1_C3_OH
                (1,1,2): 1.92317913E+00,    #C3_C3_OH
                (1,1,3): 1.91218355E+00,    #C3_C3_H1 
                (1,2,4): 1.87204096E+00     #C3_OH_HO
                }



        n_atoms = len(coords)
        _E = 0
        for pair in bonded_list:
            i = pair[0]
            j = pair[1]
            type_i = atom_type_indices[i]
            type_j = atom_type_indices[j]
            bond_type = (type_i, type_j)
            r = Converter.get_r(coords[i], coords[j])
            bond_fc = bond_force_constants[bond_type]
            bond_eq = bond_eq_lengths[bond_type]
            _E += bond_fc * (r - bond_eq) ** 2

        for triplet in angle_list:
            i = triplet[0]
            j = triplet[1]
            k = triplet[2]
            type_i = atom_type_indices[i]
            type_j = atom_type_indices[j]
            type_k = atom_type_indices[k]
            angle_type = tuple(sorted([type_i, type_j, type_k]))
            theta = get_angles(coords, triplet)
            theta = np.radians(theta)
            angle_fc = angle_force_constants[angle_type]
            angle_eq = angle_eq_values[angle_type]
            _E += angle_fc * (theta - anlge_eq) ** 2





