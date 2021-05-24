#!/usr/bin/env python

'''
This module is for converting forces and coordinates/nuclear charges 
into pair-wise and nuclear repulsive forces respectively.
'''

import numpy as np

class Converter(object):
    '''
    Takes coords, forces, atoms from Molecule class 
    and converts into desired format
    e.g. interatomic foces, nuclear repulsive forces,
    '''

    #constants
    au2Ang = 0.529177 #bohr to Angstrom
    kcal2kj = 4.184
    Eh2kcalmol = 627.5095 #hartree to kcal/mol
    Eh2kJmol = Eh2kcalmol * kcal2kj #hartree to kJ/mol
    au2kJmola = Eh2kJmol * au2Ang #convert from Eh/a_0 to kJ/(mol Ang)
    au2kcalmola = Eh2kcalmol * au2Ang #convert from Eh/a_0 to kcal/(mol Ang)
    rad2deg = float(180) / float(np.pi) #radians to degrees
    ang2m = 1e-10
    m2ang = 1e10
    fsec2sec = 1e-15
    _NA = 6.02214076e23
    au2kg = 1.66053907e-27
    _ZM = {1:1.008, 6:12.011, 8:15.999}
    _ZSymbol = {1:'H', 6:'C', 8:'O'}
 
    def __init__(self, molecule):
        #molecule.check_force_conservation()
        self.coords = molecule.coords
        self.forces = molecule.forces
        self.atoms = molecule.atoms
        self.mat_r = None
        self.mat_NRF = None
        self.get_interatomic_forces(molecule)

    def __str__(self):
        return ('\nN structures: %s,' % (len(self.coords)))

    def get_interatomic_forces(self, molecule):
        if len(self.atoms) == len(self.coords[0]):
            n_atoms = len(self.atoms)
            _NC2 = int(n_atoms * (n_atoms-1)/2)
            n_structures = len(self.coords)
            self.mat_r = np.zeros((n_structures, _NC2))
            self.mat_NRF = np.zeros((n_structures, _NC2))
            mat_vals = np.zeros((n_structures, n_atoms, 3, _NC2))
            mat_F = [] 
            for s in range(n_structures):
                _N = -1
                for i in range(n_atoms):
                    zi = self.atoms[i]
                    for j in range(i):
                        _N += 1
                        zj = self.atoms[j]
                        r = Converter.get_r(self.coords[s][i], 
                                self.coords[s][j])
                        self.mat_r[s,_N] = r
                        self.mat_r[s,_N] = r
                        if i != j:
                            self.mat_NRF[s,_N] = Converter.get_NRF(zi, zj, r)
                        for x in range(0, 3):
                            val = ((self.coords[s][i][x] - 
                                    self.coords[s][j][x]) /
                                    self.mat_r[s,_N]) * self.mat_NRF[s,_N]
                            mat_vals[s,i,x,_N] = val
                            mat_vals[s,j,x,_N] = -val

                mat_vals2 = mat_vals[s].reshape(n_atoms*3,_NC2)
                forces2 = self.forces[s].reshape(n_atoms*3)
                _F = np.matmul(np.linalg.pinv(mat_vals2), forces2)
                #'''##### if using scale factor, need to remove.
                _N2 = -1
                for i in range(n_atoms):
                    for j in range(i):
                        _N2 += 1
                        _F[_N2] = _F[_N2] * self.mat_NRF[s,_N2]
                #'''#####
                mat_F.append(_F)


            mat_F = np.reshape(np.vstack(mat_F), (n_structures,_NC2))
            molecule.mat_F = mat_F
            molecule.mat_NRF = self.mat_NRF
            #print(molecule.mat_F.shape)
            #print(molecule.mat_NRF.shape)

        else:
            raise ValueError('Number of atoms does not match '\
                    'number of coords.')

    def get_NRF(zA, zB, r):
        return zA * zB * Converter.au2kcalmola / (r ** 2)
        #return zA * zB / (r ** 2) 

    def get_r(coordsA, coordsB):
        return np.linalg.norm(coordsA-coordsB)

    def get_com(coords, masses):
        com = np.zeros((3))
        for coord, mass in zip(coords, masses):
            for i in range(3):
                com[i] += mass * coord[i]
        com = com / sum(masses)
        return np.array(com)

    def get_centroid(coords):
        centroid = np.zeros((3))
        for coord in coords:
            for i in range(3):
                centroid[i] += coord[i]
        centroid = centroid / len(coords)
        return centroid

    def get_moi_tensor(com, coords, masses):
        x_cm, y_cm, z_cm = com[0], com[1], com[2]
        _I = np.zeros((3,3))
        for coord, mass in zip(coords, masses):
            _I[0][0] += (abs(coord[1] - y_cm)**2 + \
                    abs(coord[2] - z_cm)**2) * mass
            _I[0][1] -= (coord[0] - x_cm) * (coord[1] - y_cm) * mass
            _I[1][0] -= (coord[0] - x_cm) * (coord[1] - y_cm) * mass

            _I[1][1] += (abs(coord[0] - x_cm)**2 + \
                    abs(coord[2] - z_cm)**2) * mass
            _I[0][2] -= (coord[0] - x_cm) * (coord[2] - z_cm) * mass
            _I[2][0] -= (coord[0] - x_cm) * (coord[2] - z_cm) * mass

            _I[2][2] += (abs(coord[0] - x_cm)**2 + \
                    abs(coord[1] - y_cm)**2) * mass
            _I[1][2] -= (coord[1] - y_cm) * (coord[2] - z_cm) * mass
            _I[2][1] -= (coord[1] - y_cm) * (coord[2] - z_cm) * mass
        return _I

    def get_principal_axes(coords, masses):
        com  = Converter.get_com(coords, masses)
        moi = Converter.get_moi_tensor(com, coords, masses)
        eigenvalues, eigenvectors = np.linalg.eig(moi) #diagonalise moi
        transposed = np.transpose(eigenvectors) #turn columns to rows

        ##find min and max eigenvals (magnitudes of principal axes/vectors)
        min_eigenvalue = abs(eigenvalues[0])
        if eigenvalues[1] < min_eigenvalue:
            min_eigenvalue = eigenvalues[1]
        if eigenvalues[2] < min_eigenvalue:
            min_eigenvalue = eigenvalues[2]

        max_eigenvalue = abs(eigenvalues[0])
        if eigenvalues[1] > max_eigenvalue:
            max_eigenvalue = eigenvalues[1]
        if eigenvalues[2] > max_eigenvalue:
            max_eigenvalue = eigenvalues[2]

        #PA = principal axes
        _PA = np.zeros((3,3)) 
        #MI = moment of inertia
        _MI = np.zeros((3))
        for i in range(3):
            if eigenvalues[i] == max_eigenvalue:
                _PA[i] = transposed[i]
                _MI[i] = eigenvalues[i]
            elif eigenvalues[i] == min_eigenvalue:
                _PA[i] = transposed[i]
                _MI[i] = eigenvalues[i]
            else:
                _PA[i] = transposed[i]
                _MI[i] = eigenvalues[i]

        return _PA, _MI, com

    def transform_coords(ref, other):
        '''http://nghiaho.com/?page_id=671'''
        ref = ref.reshape(3,-1)
        other = other.reshape(3,-1)
        #find column means
        centre_ref = np.mean(ref, axis=1).reshape(-1,1)
        centre_other = np.mean(other, axis=1).reshape(-1,1)
        #subtract mean
        ref_m = ref - centre_ref
        other_m = other - centre_other
        _H = ref_m @ np.transpose(other_m)
        #get rotation
        _U, _S, _Vt = np.linalg.svd(_H)
        _R = np.transpose(_Vt) @ np.transpose(_U)
        #check for reflecton
        if np.linalg.det(_R) < 0:
            _Vt[2,] *= -1 #multiply 3rd column by -1
            _R = np.transpose(_Vt) @ np.transpose(_U)
        t = -_R @ centre_ref + centre_other
        return _R, t

    def transform_coords2(ref, other, masses):
        '''
        https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
        '''
        #translate each molecule
        ref_com = Converter.get_com(ref, masses)
        #ref_com = Converter.get_centroid(ref)
        ref_translated = ref - ref_com
        other_com = Converter.get_com(other, masses)
        #other_com = Converter.get_centroid(other)
        other_translated = other - other_com
        #rotate molecule
        _C = np.dot(np.transpose(other_translated), ref_translated)
        _V, _S, _W = np.linalg.svd(_C)
        d = (np.linalg.det(_V) * np.linalg.det(_W)) < 0.0
        if d:
            _S[-1] = -_S[-1]
            _V[:,-1] = -_V[:,-1]
        _U = np.dot(_V, _W) #rotation matrix
        other_rotated = np.dot(other_translated, _U)
        return other_rotated, _U

    def rotate_forces(forces, coords, masses, n_atoms):
        _PA, _MI, com = Converter.get_principal_axes(coords, masses)
        n_f_rotated = np.zeros((n_atoms,3))
        n_c_rotated = np.zeros((n_atoms,3))
        #n_c_translated = np.zeros((n_atoms,3))
        for i in range(n_atoms):
            translated_c = coords[i] - com
            #n_c_translated[i] = translated_c
            for j in range(3):
                rotated_f = np.dot(forces[i], _PA[j])
                n_f_rotated[i][j] = rotated_f
                rotated_c = np.dot(translated_c, _PA[j])
                n_c_rotated[i][j] = rotated_c
        return n_f_rotated, n_c_rotated

    def get_rotated_forces(molecule):
        n_structures = len(molecule.coords)
        n_atoms = len(molecule.atoms)
        masses = np.array([Converter._ZM[a] for a in molecule.atoms])
        molecule.rotated_forces = np.zeros_like(molecule.forces)
        molecule.rotated_coords = np.zeros_like(molecule.coords)
        for n in range(n_structures):
            coords = molecule.coords[n]
            forces = molecule.forces[n]
            #'''
            n_f_rotated, n_c_rotated = Converter.rotate_forces(forces, 
                    coords, masses, n_atoms)
            molecule.rotated_forces[n] = n_f_rotated
            molecule.rotated_coords[n] = n_c_rotated
            #'''
            '''
            _R, t = Converter.transform_coords(
                    molecule.rotated_coords[0], n_c_rotated)
            n_c_transformed = np.dot(n_c_rotated, _R)
            molecule.rotated_coords[n] = n_c_transformed
            '''
            '''
            n_c_transformed, _U = Converter.transform_coords2(
                    molecule.coords[0], coords, masses)
            molecule.rotated_coords[n] = n_c_transformed
            #n_f_transformed, _Uf = Converter.transform_coords2(
                    #molecule.coords[n], forces, masses)
            #molecule.rotated_forces[n] = n_f_transformed
            forces_translated = forces - Converter.get_com(coords, masses)
            molecule.rotated_forces[n] = np.dot(forces_translated, _U)
            '''
        molecule.rotated_forces = molecule.get_3D_array(
                molecule.rotated_forces)
        molecule.rotated_coords = molecule.get_3D_array(
                molecule.rotated_coords)

