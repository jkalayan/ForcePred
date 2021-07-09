#!/usr/bin/env python

'''
This module is for converting forces and coordinates/nuclear charges 
into pair-wise and nuclear repulsive forces respectively.
'''

import numpy as np
#from ..network.Network import Network

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
    kB = 1.38064852e-23 
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
        if hasattr(molecule, 'charges'):
            self.get_interatomic_charges(molecule)

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
            molecule.mat_r = self.mat_r
            #print(molecule.mat_F.shape)
            #print(molecule.mat_NRF.shape)

        else:
            raise ValueError('Number of atoms does not match '\
                    'number of coords.')

    def get_interatomic_charges(self, bias_type='NRF'):
        '''
        This function takes molecule coords (C), atomic charges (Q) and 
        nuclear charges (Z) and decomposes atomic charges with a bias 
        if selected.
        Decomposed pairwise charges are recomposed and checked with 
        original atomic charges to ensure the decomposition scheme is 
        performed correctly.

        Variables are - 

        self:       molecule object containing ZCFEQ information
        bias_type:  the bias required for decomposed charges, options are:

                    '1':   No bias (bias=1)
                    '1/r':  1/r bias
                    '1/r2': 1/r^2 bias
                    'NRF':  bias using nuclear repulsive forces (zA zB/r^2)
                    'r':    r bias
        '''
        if len(self.atoms) == len(self.coords[0]):
            n_atoms = len(self.atoms)
            _NC2 = int(n_atoms * (n_atoms-1)/2)
            n_structures = len(self.coords)
            self.mat_NRF = np.zeros((n_structures, _NC2))
            self.mat_bias = np.zeros((n_structures, n_atoms, _NC2))
            self.mat_bias2 = np.zeros((n_structures, _NC2))
            mat_Q = [] 
            for s in range(n_structures):
                _N = -1
                for i in range(n_atoms):
                    zi = self.atoms[i]
                    for j in range(i):
                        _N += 1
                        zj = self.atoms[j]
                        r = Converter.get_r(self.coords[s][i], 
                                self.coords[s][j])
                        if i != j:
                            self.mat_NRF[s,_N] = Converter.get_NRF(zi, zj, r)
                            bias = Converter.get_bias(zi, zj, r, bias_type)
                            if bias_type == 'NRF':
                                bias = self.mat_NRF[s,_N]
                            self.mat_bias[s,i,_N] = bias
                            self.mat_bias[s,j,_N] = -bias
                            self.mat_bias2[s,_N] = bias

                charges2 = self.charges[s].reshape(n_atoms)
                _Q = np.matmul(np.linalg.pinv(self.mat_bias[s]), charges2)
                #rescale using bias
                _N2 = -1
                for i in range(n_atoms):
                    for j in range(i):
                        _N2 += 1
                        #_Q[_N2] = _Q[_N2] * self.mat_NRF[s,_N2]
                        _Q[_N2] = _Q[_N2] * self.mat_bias2[s,_N2]
                mat_Q.append(_Q)

            mat_Q = np.reshape(np.vstack(mat_Q), (n_structures,_NC2))
            self.mat_Q = mat_Q
            recomp_Q = Converter.get_recomposed_charges(self.coords, 
                    self.mat_Q, n_atoms, _NC2)
            ##check recomp_Q is the same as Q
            if np.array_equal(np.round(recomp_Q, 1), 
                    np.round(self.charges, 1)) == False:
                raise ValueError('Recomposed charges {} do not '\
                        'equal initial charges {}'.format(
                        recomp_Q, self.charges))
        else:
            raise ValueError('Number of atoms does not match '\
                    'number of coords.')

    def get_recomposed_charges(all_coords, all_prediction, n_atoms, _NC2):
        '''Take pairwise decomposed charges and convert them back into 
        atomic charges.'''
        all_recomp_charges = []
        for coords, prediction in zip(all_coords, all_prediction):
            eij = np.zeros((n_atoms, n_atoms))
                #normalised interatomic vectors
            q_list = []
            for i in range(1,n_atoms):
                for j in range(i):
                    eij[i,j] = 1
                    eij[j,i] = -eij[i,j]
                    q_list.append([i,j])
            _T = np.zeros((n_atoms, _NC2))
            for i in range(int(_T.shape[0])):
                for k in range(len(q_list)):
                    if q_list[k][0] == i:
                        _T[range(i, (i+1)), k] = \
                                eij[q_list[k][0],q_list[k][1]]
                    if q_list[k][1] == i:
                        _T[range(i, (i+1)), k] = \
                                eij[q_list[k][1],q_list[k][0]]
            recomp_charges = np.zeros((n_atoms))
            recomp_charges = np.dot(_T, prediction.flatten())
            all_recomp_charges.append(recomp_charges)
        return np.array(all_recomp_charges).reshape(-1,n_atoms)

    def get_bias(zA, zB, r, bias_type):
        bias = 1
        if bias_type == '1/r':
            bias = 1 / r
        if bias_type == '1/r2':
            bias = 1 / r ** 2
        if bias_type == 'r':
            bias = r
        return bias

    def get_NRF(zA, zB, r):
        return zA * zB * Converter.au2kcalmola / (r ** 2)
        #return zA * zB / (r ** 2) 
        #return 1 / (r ** 2) 

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

    def translate_coords(coords, atoms):
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

    def get_r_from_NRF(_NRF, atoms):
        rs = np.zeros_like(_NRF)
        n_atoms = len(atoms)
        n = -1
        for i in range(n_atoms):
            zA = atoms[i]
            for j in range(i):
                n += 1
                zB = atoms[j]
                r = ((zA * zB) / _NRF[n]) ** 0.5
                rs[n] = r
        return rs


    def get_coords_from_NRF(_NRF, atoms, coords, scale, scale_min):

        n_atoms = len(atoms)
        #print(_NRF)
        #scale_NRF = _NRF / np.amax(scale)
        #scale_min_NRF = _NRF / np.amin(scale_min)
        scale_NRF = _NRF / scale
        scale_min_NRF = _NRF / scale_min
        #r_min = (1 / scale) ** 0.5
        #r_max = (1 / scale_min) ** 0.5

        #print('scale_NRF', scale_NRF)
        #print('scale_NRF_min', scale_min_NRF)
        #print(coords)
        n = -1
        for i in range(n_atoms):
        #for i in range(n_atoms-1):
            zA = atoms[i]
            for j in range(i):
            #for j in range(i+1, n_atoms):
                n += 1
                zB = atoms[j]
                r = Converter.get_r(coords[i], coords[j])
                #r_min = ((zA * zB) / scale) ** 0.5
                #r_max = ((zA * zB) / scale_min) ** 0.5
                r_min = ((zA * zB) / scale[n]) ** 0.5
                r_max = ((zA * zB) / scale_min[n]) ** 0.5
                #print(i, j, n)
                #print('\t', r)
                s = None
                if scale_NRF[n] > 1:
                    s = r_min
                if scale_min_NRF[n] < 1:
                    s = r_max
                if s != None:
                    #print(_NRF[n], s)
                    coords1 = coords[i]
                    coords2 = coords[j]
                    v = coords2 - coords1
                    new_r = s #use max or min value
                    #print(n+1, ':', i+1, j+1, ':', zA, zB, ':', 
                            #r_min, r_max, r, new_r)
                    u = v / r
                    new_coords = coords1 + new_r * u
                    coords[j] = new_coords
        #print(coords)
        return coords

