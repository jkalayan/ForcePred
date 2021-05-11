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
        a = _F / m
        v = (_C - _Cprev) / dt + a * dt
        _Cnew = _C + v * dt + a * dt ** 2
        return _Cnew / Converter.ang2m









