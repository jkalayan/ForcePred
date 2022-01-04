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
    au2kJmola = Eh2kJmol / au2Ang #convert from Eh/a_0 to kJ/(mol Ang)
    au2kcalmola = Eh2kcalmol * au2Ang ###convert from Eh/a_0 to kcal/(mol Ang)
    au2kcalmola = np.float64(au2kcalmola)
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
            self.charges = molecule.charges
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
            mat_bias = np.zeros((n_structures, _NC2))
            mat_F = [] 
            for s in range(n_structures):
                _N = -1
                for i in range(n_atoms):
                    zi = self.atoms[i]
                    for j in range(i):
                        _N += 1
                        zj = self.atoms[j]
                        cutoff = 3
                        r = Converter.get_r(self.coords[s][i], 
                                self.coords[s][j])
                        self.mat_r[s,_N] = r
                        self.mat_r[s,_N] = r
                        self.mat_NRF[s,_N] = Converter.get_NRF(zi, zj, r)
                        #mat_bias[s,_N] = self.mat_NRF[s,_N]
                        mat_bias[s,_N] = 1 #Converter.get_bias(zi, zj, r, 
                                #'1/r')
                        #self.mat_NRF[s,_N] = mat_bias[s,_N]
                        #if r > cutoff:
                            #mat_bias[s,_N] = 0
                            #self.mat_NRF[s,_N] = 0
                        for x in range(0, 3):
                            val = ((self.coords[s][i][x] - 
                                    self.coords[s][j][x]) /
                                    self.mat_r[s,_N]) * mat_bias[s,_N]
                                        # self.mat_NRF[s,_N]
                            mat_vals[s,i,x,_N] = val
                            mat_vals[s,j,x,_N] = -val

                mat_vals2 = mat_vals[s].reshape(n_atoms*3,_NC2)
                forces2 = self.forces[s].reshape(n_atoms*3)
                _F = np.matmul(np.linalg.pinv(mat_vals2), forces2)
                ''' #for atomwise decomposition
                print('decompF_all', _F)
                print()
                #recompF = np.dot(mat_vals2, _F)
                #print(forces2, recompF)
                for i in range(n_atoms):
                    e1 = mat_vals[s][i].reshape(3,_NC2)
                    f1 = self.forces[s][i].reshape(3)
                    _f1 = np.matmul(np.linalg.pinv(e1), f1)
                    recompF1 = np.dot(e1, _f1)
                    print('e1', e1)
                    print('atomF and atomRecompF', f1, recompF1)
                    print('atom decompF', _f1)
                    print()
                print()
                '''
                #'''##### if using scale factor, need to remove it.
                _N2 = -1
                for i in range(n_atoms):
                    for j in range(i):
                        _N2 += 1
                        _F[_N2] *= mat_bias[s,_N]
                            # self.mat_NRF[s,_N2]
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

    def get_decomposition(atoms, coords, var):
        '''forces'''
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        mat_r = np.zeros((_NC2))
        mat_vals = np.zeros((n_atoms, 3, _NC2))
        mat_F = [] 
        _N = -1
        for i in range(n_atoms):
            zi = atoms[i]
            for j in range(i):
                _N += 1
                zj = atoms[j]
                r = Converter.get_r(coords[i], coords[j])
                mat_r[_N] = r
                mat_r[_N] = r
                for x in range(0, 3):
                    val = ((coords[i][x] - coords[j][x]) / mat_r[_N])
                    mat_vals[i,x,_N] = val
                    mat_vals[j,x,_N] = -val

        mat_vals2 = mat_vals.reshape(n_atoms*3,_NC2)
        var2 = var.reshape(n_atoms*3)
        decomposed = np.matmul(np.linalg.pinv(mat_vals2), var2)
        return decomposed

    def get_interatomic_energies(molecule, bias_type='NRF'):
        '''decompose molecular energy into pairwise energies'''
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        n_structures = len(molecule.coords)
        molecule.mat_NRF = np.zeros((n_structures, _NC2))
        molecule.mat_E = np.zeros((n_structures, _NC2))
        molecule.mat_r = np.zeros((n_structures, _NC2))
        molecule.mat_bias = np.zeros((n_structures, _NC2))
        for s in range(n_structures):
            #mat_bias = np.zeros((_NC2))
            #mat_r = np.zeros((_NC2))
            _N = -1
            for i in range(n_atoms):
                zi = molecule.atoms[i]
                for j in range(i):
                    _N += 1
                    zj = molecule.atoms[j]
                    r = Converter.get_r(molecule.coords[s,i], 
                            molecule.coords[s,j])
                    molecule.mat_r[s,_N] = r
                    molecule.mat_NRF[s,_N] = Converter.get_NRF(zi, zj, r)
                    bias = molecule.mat_NRF[s,_N]
                    if bias_type != 'NRF':
                        bias = Converter.get_bias(zi, zj, r, bias_type)
                    molecule.mat_bias[s,_N] = bias

            mat_bias2 = molecule.mat_bias[s].reshape((1,_NC2))
            _E = molecule.energies[s].reshape(1)
            decomp_E = np.matmul(np.linalg.pinv(mat_bias2), _E)
            #rescale using bias
            _N2 = -1
            for i in range(n_atoms):
                for j in range(i):
                    _N2 += 1
                    decomp_E[_N2] *= molecule.mat_bias[s,_N2]
            molecule.mat_E[s] = decomp_E
            #for e in molecule.mat_E[s]:
                #print(e)
            #print(mat_r)
            #print()
        #molecule.mat_E = np.reshape(np.vstack(mat_E), (n_structures,_NC2))


    def get_atomwise_decompF(molecule, bias_type='NRF'):
        '''Get decomposed forces for each pair separately. This is done by
        averaging pair contributions for each atom both for inputs and eij
        and then using the averages in x,z,z to get single decompF. Could
        then train these all separately as with ANI-1.'''

        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        n_structures = len(molecule.coords)
        molecule.atom_NRF = np.zeros((n_structures, n_atoms, _NC2))
        #molecule.mat_bias = np.zeros((n_structures, n_atoms))
        molecule.atom_F = np.zeros((n_structures, n_atoms, _NC2))
        for s in range(n_structures):
            mat_eij = np.zeros((n_atoms, 3, _NC2))
            _N = -1
            for i in range(n_atoms):
                zi = molecule.atoms[i]
                for j in range(i):
                    _N += 1
                    zj = molecule.atoms[j]
                    r = Converter.get_r(molecule.coords[s][i], 
                            molecule.coords[s][j])
                    molecule.atom_NRF[s,i,_N] = \
                            Converter.get_NRF(zi, zj, r)
                    molecule.atom_NRF[s,j,_N] = molecule.atom_NRF[s,i,_N] 
                    for x in range(0, 3):
                        eij = ((molecule.coords[s][i][x] - 
                                molecule.coords[s][j][x]) / r)
                        mat_eij[i,x,_N] = eij
                        mat_eij[j,x,_N] = -eij

            for i in range(n_atoms):
                #print(i)
                mat_eij2 = mat_eij[i].reshape(3,_NC2)
                forces2 = molecule.forces[s,i].reshape(3)
                decomp_atom_F = np.matmul(np.linalg.pinv(mat_eij2), forces2)
                recomp_F = np.dot(mat_eij2, decomp_atom_F)
                #print('NRF', molecule.atom_NRF[s,i])
                #print('decompF', decomp_atom_F)
                #print('F', forces2, recomp_F)
                molecule.atom_F[s,i] = decomp_atom_F
            #print(molecule.atom_F[s])
            #print(mat_eij)
            #print()

    def get_atomwise_recompF(all_coords, all_prediction, atoms, n_atoms, 
            _NC2):
        '''Take per-atom pairwise decomposed F and convert them back into 
        Cart forces.'''
        all_recomp_F = np.zeros((len(all_coords), n_atoms, 3))
        s = -1
        for coords, prediction in zip(all_coords, all_prediction):
            mat_eij = np.zeros((n_atoms, 3, _NC2))
            _N = -1
            for i in range(n_atoms):
                zi = atoms[i]
                for j in range(i):
                    _N += 1
                    zj = atoms[j]
                    r = Converter.get_r(coords[i], coords[j])
                    for x in range(0, 3):
                        eij = ((coords[i][x] - coords[j][x]) / r)
                        mat_eij[i,x,_N] = eij
                        mat_eij[j,x,_N] = -eij

            for i in range(n_atoms):
                mat_eij2 = mat_eij[i].reshape(3,_NC2)
                recomp_F = np.dot(mat_eij2, prediction[i])
                all_recomp_F[s,i] = recomp_F
        return all_recomp_F


    def get_simultaneous_interatomic_energies_forces(molecule, 
            bias_type='NRF'):
        '''Get decomposed energies and forces from the same simultaneous
        equation'''

        force_bias = False

        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        n_structures = len(molecule.coords)
        molecule.mat_NRF = np.zeros((n_structures, _NC2))
        molecule.mat_bias = np.zeros((n_structures, _NC2))
        molecule.mat_FE = np.zeros((n_structures, _NC2))
        molecule.mat_eij = np.zeros((n_structures, n_atoms*3+1, _NC2))
        for s in range(n_structures):
            mat_r = np.zeros((_NC2))
            mat_Fvals = np.zeros((n_atoms, 3, _NC2))
            _N = -1
            for i in range(n_atoms):
                zi = molecule.atoms[i]
                for j in range(i):
                    _N += 1
                    zj = molecule.atoms[j]
                    r = Converter.get_r(molecule.coords[s][i], 
                            molecule.coords[s][j])
                    mat_r[_N] = r
                    molecule.mat_NRF[s,_N] = Converter.get_NRF(zi, zj, r)
                    bias = molecule.mat_NRF[s,_N]
                    if bias_type != 'NRF':
                        bias = Converter.get_bias(zi, zj, r, bias_type)
                    molecule.mat_bias[s,_N] = bias
                    molecule.mat_eij[s,n_atoms*3,_N] = bias
                    for x in range(0, 3):
                        val = ((molecule.coords[s][i][x] - 
                                molecule.coords[s][j][x]) / mat_r[_N])
                        if force_bias:
                            val *= bias
                        mat_Fvals[i,x,_N] = val
                        mat_Fvals[j,x,_N] = -val
                        molecule.mat_eij[s,i*3+x,_N] = val
                        molecule.mat_eij[s,j*3+x,_N] = -val

            mat_Fvals2 = mat_Fvals.reshape(n_atoms*3,_NC2)
            forces2 = molecule.forces[s].reshape(n_atoms*3)
            #decomp_F = np.matmul(np.linalg.pinv(mat_Fvals2), forces2)

            mat_bias2 = molecule.mat_bias[s].reshape((1,_NC2))
            _E = molecule.energies[s].reshape(1)
            #decomp_E = np.matmul(np.linalg.pinv(mat_bias2), _E)

            mat_FE = np.concatenate((mat_Fvals2, mat_bias2), axis=0)
            _FE = np.concatenate([forces2.flatten(), _E.flatten()])
            decomp_FE = np.matmul(np.linalg.pinv(mat_FE), _FE)


            '''
            #not sure how to rescale using bias
            _N2 = -1
            for i in range(n_atoms):
                for j in range(i):
                    _N2 += 1
                    decomp_FE[_N2] *= mat_bias[_N2]
                    if force_bias:
                        decomp_FE[_N2] *= mat_bias[_N2]
            '''
            molecule.mat_FE[s] = decomp_FE


    def get_recomposed_FE(all_coords, all_prediction, atoms, n_atoms, _NC2, 
            bias_type):
        '''Take pairwise decomposed FE and convert them back into 
        Cart forces and molecular energy.'''
        all_recomp_F = np.zeros((len(all_coords), n_atoms, 3))
        all_recomp_E = np.zeros((len(all_coords), 1))
        s = -1
        for coords, prediction in zip(all_coords, all_prediction):
            s += 1
            #print(coords.shape)
            #print(prediction.shape)
            rij = np.zeros((3, n_atoms, n_atoms))
                #interatomic vectors from col to row
            eij = np.zeros((3, n_atoms, n_atoms))
                #normalised interatomic vectors
            bias_ij = np.zeros((n_atoms, n_atoms))
            q_list = []
            for i in range(n_atoms):
                zi = atoms[i] 
                for j in range(i):
                    zj = atoms[j]
                    r = Converter.get_r(coords[i], coords[j])
                    bias = Converter.get_bias(zi, zj, r, bias_type)
                    bias_ij[i,j] = bias
                    rij[:,i,j] = (coords[i,:] - coords[j,:])
                    rij[:,j,i] = -rij[:,i,j]
                    eij[:,i,j] = rij[:,i,j] / np.reshape(
                            np.linalg.norm(rij[:,i,j], axis=0), (-1,1))
                    eij[:,j,i] = -eij[:,i,j]
                    q_list.append([i,j])
            _T = np.zeros((3*n_atoms+1, _NC2))
            for i in range(n_atoms):
                for k in range(_NC2):
                    if q_list[k][0] == i:
                        _T[range(i*3, (i+1)*3), k] = \
                                eij[:,q_list[k][0],q_list[k][1]]
                    if q_list[k][1] == i:
                        _T[range(i*3, (i+1)*3), k] = \
                                eij[:,q_list[k][1],q_list[k][0]]
            k = -1
            for i in range(1,n_atoms):
                for j in range(i):
                    k += 1
                    _T[-1,k] = bias_ij[i,j]

            recomp_FE = np.dot(_T, prediction.flatten())
            recomp_F = recomp_FE[:-1].reshape(-1,3)
            recomp_E = recomp_FE[-1]
            #print('recomp_F', recomp_F.shape, recomp_F)
            #print('recomp_E', recomp_E.shape, recomp_E)
            all_recomp_F[s] = recomp_F
            all_recomp_E[s] = recomp_E
        return all_recomp_F, all_recomp_E


    def get_energy_decomposition(atoms, coords, var):
        '''test to see if one energy term can be decomposed into 
        pairwise energies'''
        n_atoms = len(atoms)
        _NC2 = int(n_atoms * (n_atoms-1)/2)
        mat_r = np.zeros((_NC2))
        mat_vals = np.zeros((_NC2))
        mat_F = [] 
        _N = -1
        for i in range(n_atoms):
            zi = atoms[i]
            for j in range(i):
                _N += 1
                zj = atoms[j]
                r = Converter.get_r(coords[i], coords[j])
                #mat_r[_N] = r
                #mat_r[_N] = r
                #val = ((coords[i] - coords[j]) / mat_r[_N])
                mat_vals[_N] = 1/r #val

        mat_vals2 = mat_vals.reshape((1,_NC2))
        var2 = var.reshape(1)
        decomposed = np.matmul(np.linalg.pinv(mat_vals2), var2)
        #rescale using bias
        _N2 = -1
        for i in range(n_atoms):
            for j in range(i):
                _N2 += 1
                decomposed[_N2] = decomposed[_N2] * mat_vals[_N2]
        return decomposed

    def get_recomposed_energy(coords, prediction, n_atoms, _NC2):
        '''Take pairwise decomposed energies and convert them back into 
        molecular energy.'''
        rij = np.zeros((n_atoms, n_atoms))
        eij = np.zeros((n_atoms, n_atoms))
            #normalised interatomic vectors
        q_list = []
        for i in range(1,n_atoms):
            for j in range(i):
                r = Converter.get_r(coords[i], coords[j])
                eij[i,j] = 1 #r
                eij[j,i] = -eij[i,j] #-r
                q_list.append([i,j])
        _T = np.zeros((1, _NC2))
        for i in range(int(_T.shape[0])):
            for k in range(len(q_list)):
                #if q_list[k][0] == i:
                _T[i,k] = eij[q_list[k][0],q_list[k][1]]
                '''
                if q_list[k][1] == i:
                    _T[range(i, (i+1)), k] = \
                            eij[q_list[k][1],q_list[k][0]]
                '''
        #print(_T)
        recomp = np.zeros((1))
        recomp = np.dot(_T, prediction.flatten())
        return recomp 


    def get_bias(zA, zB, r, bias_type):
        bias = 1
        if bias_type == '1/r':
            bias = 1 / r
        if bias_type == '1/r2':
            bias = 1 / r ** 2
        if bias_type == 'r':
            bias = r
        if bias_type == 'r/qq':
            bias = r / zA * zB
        if bias_type == 'NRF':
            bias = Converter.get_NRF(zA, zB, r)
        if bias_type == '1/qqr2':
            bias = 1 / zA * zB * r ** 2
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
        #masses = np.array([1]*n_atoms)
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

        orig_coords = np.copy(coords)
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
                r_min = ((zA * zB * Converter.au2kcalmola) / scale[n]) ** 0.5
                r_max = ((zA * zB * Converter.au2kcalmola) / scale_min[n]) ** 0.5
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
        if np.array_equal(orig_coords, coords) == False:
            print (orig_coords)
            print(coords)
            print()
        #print(coords)
        return coords

