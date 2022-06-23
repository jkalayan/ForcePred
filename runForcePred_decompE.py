#!/usr/bin/env python

__author__ = 'Jas Kalayan'
__credits__ = ['Jas Kalayan', 'Ismaeel Ramzan', 
        'Neil Burton',  'Richard Bryce']
__license__ = 'GPL'
__maintainer__ = 'Jas Kalayan'
__email__ = 'jkalayan@gmail.com'
__status__ = 'Development'

from datetime import datetime
import argparse
import numpy as np

openmm = False
if openmm:
    from ForcePred.calculate.OpenMM import OpenMM

mdanal = False
if mdanal:
    from ForcePred.read.AMBLAMMPSParser import AMBLAMMPSParser
    from ForcePred.read.AMBERParser import AMBERParser
                               
from ForcePred import Molecule, OPTParser, NPParser, Converter, \
        Permuter, XYZParser, Binner, Writer, Plotter, Conservation
        #Network


#from ForcePred.nn.Network_v2 import Network
#from ForcePred.nn.Network_atomwise import Network
from ForcePred.nn.Network_perminv import Network
from keras.models import Model, load_model    
from keras import backend as K                                              
import sys
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

#import numpy as np
#from itertools import islice

import os
import math
#os.environ['OMP_NUM_THREADS'] = '8'

# Get number of cores reserved by the batch system
# ($NSLOTS is set by the batch system, or use 1 otherwise)
NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


tf.config.threading.set_inter_op_parallelism_threads(NUMCORES) 
tf.config.threading.set_intra_op_parallelism_threads(NUMCORES)
tf.config.set_soft_device_placement(1)


def run_force_pred(input_files='input_files', 
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files'):


    startTime = datetime.now()

    print(startTime)
    molecule = Molecule() #initiate molecule class

    if list_files:
        input_files = open(list_files).read().split()
    #OPTParser(input_files, molecule, opt=False) #read in FCEZ for SP
    #OPTParser(input_files, molecule, opt=True) #read in FCEZ for opt
    #AMBERParser('molecules.prmtop', coord_files, force_files, molecule)
    #XYZParser(atom_file, coord_files, force_files, energy_files, molecule)
    NPParser(atom_file, coord_files, force_files, energy_files, molecule)
    #write pdb file for first frame
    Writer.write_pdb(molecule.coords[0], 'MOL', 1, molecule.atoms, 
            'molecule.pdb', 'w')


    '''
    for Rs in range(1):
        Converter.get_acsf(molecule, Rs)

        atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
        atom_pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                atom_pairs.append('{}_{}'.format(atom_names[i], 
                    atom_names[j]))
        input_header = ['input_' + s for s in atom_names]
        input_header = ','.join(input_header)
        output_header = ['output_' + s for s in atom_pairs]
        output_header = ','.join(output_header)
        prediction_header = ['prediction_' + s for s in atom_pairs]
        prediction_header = ','.join(prediction_header)
        header = input_header #+ ',' + output_header + ',' + \
                #prediction_header

        Writer.write_csv([
                molecule.acsf,
                #output_atomNRF, 
                #output_atomFE, 
                #output_matFE,
                #prediction_atomFE
                #prediction_matFE
                ], 'acsf-{}'.format(Rs), 
                header)

    print(np.min(molecule.mat_qqr2), np.max(molecule.mat_qqr2))
    '''

    #sys.exit()

    '''
    ##shorten num molecules here:
    end = len(molecule.coords)
    step = 500
    molecule.coords = molecule.coords[0:end:step]
    molecule.forces = molecule.forces[0:end:step]
    molecule.energies = molecule.energies[0:end:step]
    '''



    '''
    atoms_flat = tf.reshape(atoms_, [-1])
    Ztypes, idx, count = tf.unique_with_counts(atoms_flat)
    idx = tf.reshape(idx, shape=tf.shape(atoms_))
    print('Ztypes', sess.run(Ztypes))
    print('idx', sess.run(idx))
    sort_idx = tf.argsort(idx)
    print('sort_idx', sess.run(sort_idx))
    print('count', sess.run(count))

    coords = tf.gather(coords, sort_idx, axis=1, batch_dims=-1)
    atoms_ = tf.gather(atoms_, sort_idx, batch_dims=-1)
    print('coords', coords.shape, sess.run(coords))
    print('atoms_', sess.run(atoms_))
    '''

    '''
    print('\ntensorflow')
    sess = tf.compat.v1.Session()
    coords = np.copy(molecule.coords[0:2])#.reshape(1, n_atoms, 3)
    #atoms = np.array([float(i) for i in molecule.atoms] * len(coords), 
            #dtype='float32')
    atoms = np.float32(np.array(molecule.atoms * len(coords)).reshape(
            len(coords),-1))
    coords[1,[0,8]] = coords[1,[8,0]] 
    atoms[1,[0,8]] = atoms[1,[8,0]] 

    coords = tf.convert_to_tensor(coords, np.float32)
    atoms_ = tf.convert_to_tensor(atoms, dtype=tf.float32)
    print('coords', coords.shape, sess.run(coords))
    print('atoms_', sess.run(atoms_))
    batch = coords.shape.as_list()[0]
    n_atoms_ = coords.shape.as_list()[1]



    a = tf.expand_dims(coords, 2)
    b = tf.expand_dims(coords, 1)
    diff = a - b
    r = tf.reduce_sum(diff**2, axis=-1) ** 0.5 #get sqrd diff
    print('r', r.shape)
    print(sess.run(r))
    min_r = tf.constant([1.6])
    bools = tf.math.less(r, min_r)
    print(sess.run(bools))
    A = tf.where(bools, 1, 0)
    A = tf.linalg.set_diag(A, tf.zeros([A.shape[0], A.shape[1]], 
            dtype=tf.int32))
    A = tf.cast(A, tf.float32)
    print(sess.run(A))

    atoms_flat = tf.reshape(atoms, [-1])
    Ztypes, idx, count = tf.unique_with_counts(atoms_flat)
    #idx = tf.reshape(idx, shape=tf.shape(atoms_))
    idx = tf.reshape(idx, shape=(batch,n_atoms_,1))
    print('Ztypes', sess.run(Ztypes))
    print('idx', sess.run(idx))
    print('count', sess.run(count))

    Ztypes_I = tf.eye(tf.shape(Ztypes)[0], batch_shape=[tf.shape(coords)[0]])
    print('Ztypes_I', Ztypes_I.shape)
    print(sess.run(Ztypes_I))
    N_atomTypes = tf.cast(Ztypes_I.shape.as_list()[1], tf.float32)
    Ztypes_I2 = tf.expand_dims(Ztypes_I, axis=1)


    X = tf.tile(Ztypes_I2, [1,n_atoms_,1,1])
    X = tf.gather(X, idx, batch_dims=-1, axis=2)
    X = tf.squeeze(X, [2])
    print('X', X.shape)
    print(sess.run(X[0]))

    n = -1
    for row in range(n_atoms_):
        n += 1
        r = tf.reshape(tf.repeat(A[:,row], repeats=[N_atomTypes], axis=0), 
                shape=(batch,N_atomTypes,n_atoms_))
        r = tf.transpose(r, perm=[0,2,1])
        #print(n, 'r', r.shape, sess.run(r))
        #print('X', sess.run(X))
        h1 = r * X
        #print('h1', sess.run(h1))
        sum1 = tf.reduce_sum(h1, axis=1)
        #print('sum1', sess.run(sum1))
        all_sum = 0
        for i in range(n_atoms_):
            h = h1[:,i]
            #row2 = A[:,i]
            r2 = tf.reshape(tf.repeat(A[:,n], repeats=[N_atomTypes], axis=0), 
                    shape=(batch,N_atomTypes,n_atoms_))
            r2 = tf.transpose(r2, perm=[0,2,1])
            #print('r2', sess.run(r2))
            red_h = tf.reduce_sum(h, axis=1)
            red_h = tf.expand_dims(red_h, axis=1)
            red_h = tf.expand_dims(red_h, axis=2)
            h2 = r2 * X * red_h

            #print('h', sess.run(tf.reduce_sum(h, axis=1)))
            print('h2', sess.run(tf.reduce_sum(h2, axis=1)))
            #all_sum += tf.reduce_sum(h2, axis=1)
        #print('all_sum', sess.run(all_sum))
    '''



    '''
    print('\nnumpy')

    n_atoms = len(molecule.atoms)
    coords = np.copy(molecule.coords[0])
    atoms = np.copy(molecule.atoms)
    A = Molecule.find_bonded_atoms(atoms, coords)
    indices, pairs_dict = Molecule.find_equivalent_atoms(atoms, A)
    print('Z\n', atoms)
    print('A\n', A) 
    print('indices\n', indices)
    print('pairs_dict\n', pairs_dict)
    print()



    #A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    #Z_types = list(set(molecule.atoms))
    Z_types = [6,8,1]
    print('Z_types', Z_types)
    N_atomTypes = len(Z_types)
    N = np.eye(N_atomTypes)    
    print('N', N)
    X = []
    for z in atoms:
        ind = Z_types.index(z)
        X.append(N[ind])
    X = np.array(X)
    print('X', X)
    E = []
    n = -1
    for row in A:
        n += 1
        r = np.transpose(np.array([row,]*N_atomTypes))
        #print(n, 'r', r.shape, r)
        #print('X', X)
        h1 = r * X
        #print('h1', h1)
        sum1 = np.sum(h1, axis=0)
        #print('sum1', sum1)
        all_sum = 0
        for i in range(len(h1)):
            h = h1[i]
            #print('h', h)
            is_zero = np.all(h == 0)
            #print('is_zero', is_zero)
            #if is_zero == False:
            row2 = A[i]
            r2 = np.transpose(np.array([row2,]*N_atomTypes))
            #print('r2', r2)
            h2 = r2 * X #* np.sum(h)
            #print('h', np.sum(h))
            print('h2', np.sum(h2, axis=0))
            all_sum += np.sum(h2, axis=0)
            #print()
        #print('all_sum', all_sum)
        cross = np.cross(sum1, all_sum.T) * X[n] #Z_types[n] #amended here
        E.append(cross)
    E = np.array(E)
    #print('E', E)
    u, indices = np.unique(E, axis=0, return_inverse=True)

    pairs = []
    pairs_dict = {}
    _N = -1
    for i in range(n_atoms):
        for j in range(i):
            _N += 1
            pair = (indices[i], indices[j])
            pairs.append(pair)
            if pair not in pairs_dict:
                pairs_dict[pair] = []
            pairs_dict[pair].append(_N)

    print(indices, '\n', pairs_dict)
    '''


    print(datetime.now() - startTime)
    sys.stdout.flush()
    #sys.exit()
    #'''

    print(molecule)
    n_atoms = len(molecule.atoms)
    _NC2 = int(n_atoms * (n_atoms-1)/2)

    '''
    A = Molecule.find_bonded_atoms(molecule.atoms, molecule.coords[0])
    print(molecule.atoms)
    print(A)
    indices, pairs_dict = Molecule.find_equivalent_atoms(
            molecule.atoms, A)
    print('indices', indices)
    print('pairs_dict', pairs_dict)
    '''
    sys.stdout.flush()




    split = 100 #500 #200 #2
    train = round(len(molecule.coords) / split, 3)
    print('\nget train and test sets, '\
            'training set is {} points'.format(train))
    Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
            split) #get train and test sets
    '''
    print('!!!use regularly spaced training')
    molecule.train = np.arange(2, len(molecule.coords), split).tolist() 
            #includes the structure with highest force magnitude for malonaldehyde
    molecule.test = [x for x in range(0, len(molecule.coords)) 
            if x not in molecule.train]
    print(len(molecule.train))
    print(len(molecule.test))
    '''


    '''
    print('!!!Using electronic energies rather than QM: E_qm = E_elec + E_nr')
    molecule.energies_orig = np.copy(molecule.energies)
    Converter.get_all_NRFs(molecule)
    sumNRFs = np.sum(molecule.mat_NRF, axis=1).reshape(-1,1)
    molecule.energies_elec = molecule.energies_orig - sumNRFs
    molecule.energies = np.copy(molecule.energies_elec)
    print(molecule.energies_orig[0], sumNRFs[0], molecule.energies[0])
    print(molecule.energies_orig.shape, sumNRFs.shape, molecule.energies.shape)
    #sys.exit()
    '''

    train_forces = np.take(molecule.forces, molecule.train, axis=0)
    train_energies = np.take(molecule.energies, molecule.train, axis=0)

    ###!!!!!!!!
    #print('!!!TRAIN OVER-RIDDEN FRO SCALING')
    #train_forces = molecule.forces
    #train_energies = molecule.energies

    forces_min = np.min(molecule.forces)
    forces_max = np.max(molecule.forces)
    forces_diff = forces_max - forces_min
    #print(forces_diff)
    forces_rms = np.sqrt(np.mean(molecule.forces.flatten()**2))
    energies_rms = np.sqrt(np.mean(molecule.energies.flatten()**2))
    forces_mean = np.mean(molecule.forces.flatten())
    energies_mean = np.mean(molecule.energies.flatten())
    energies_max = np.max(np.absolute(molecule.energies))
    #molecule.energies = (molecule.energies - energies_mean) / energies_rms
    #molecule.forces = (molecule.forces * 0) # forces_rms)


    print('E ORIG min: {} max: {}'.format(np.min(molecule.energies), 
            np.max(molecule.energies)))
    print('train E ORIG min: {} max: {}'.format(np.min(train_energies), 
            np.max(train_energies)))
    print('F ORIG min: {} max: {}'.format(np.min(molecule.forces), 
            np.max(molecule.forces)))
    print('train F ORIG min: {} max: {}'.format(np.min(train_forces), 
            np.max(train_forces)))
    prescale_energies = True
    #prescale = [0, 1, 0, energies_mean, forces_rms, energies_max, forces_max]
    prescale = [0, 1, 0, 1]
    if prescale_energies:
        #'''
        print('\nprescale energies so that magnitude is comparable to forces')
        '''
        forces_rms =np.sqrt(np.mean(molecule.forces.flatten()**2))
        energies_mean = np.mean(molecule.energies.flatten())
        abs_max_e = np.max(np.absolute(molecule.energies))
        abs_max_f = np.max(np.absolute(molecule.forces))
        prescale[0] = abs_max_e
        prescale[1] = abs_max_f
        #molecule.energies = np.add(molecule.energies, prescale[0]) * \
                #prescale[1] #orig
        molecule.energies = molecule.energies + prescale[0]
        prescale[2] = np.max(np.absolute(molecule.energies))
        molecule.energies = molecule.energies - prescale[2]/2
        '''
        min_e = np.min(train_energies)
        max_e = np.max(train_energies)
        min_f = np.min(train_forces)
        max_f = np.max(train_forces)

        molecule.energies = ((max_f - min_f) * (molecule.energies - min_e) / 
                (max_e - min_e) + min_f)

        prescale[0] = min_e
        prescale[1] = max_e
        prescale[2] = min_f
        prescale[3] = max_f

        print('E SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))

        #'''

        '''
        print('\nprescale energies so that magnitude is comparable to forces')
        print('ORIG min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        forces_rms = 1 #np.sqrt(np.mean(molecule.forces.flatten()**2))
        energies_mean = np.mean(molecule.energies.flatten())
        print(molecule.energies[0])
        molecule.energies = (molecule.energies - energies_mean) / forces_rms
        print(molecule.energies[0])
        prescale = [energies_mean, forces_rms]
        print((molecule.energies[0] * prescale[1]) + prescale[0])
        print('SCALED min: {} max: {}'.format(np.min(molecule.energies), 
                np.max(molecule.energies)))
        print('prescale value:', prescale)
        '''
    print('prescale value:', prescale)
    sys.stdout.flush()


    #Converter(molecule) #get pairwise forces
    #sys.exit()

    #get atomwise decomposed forces and NRF inputs
    get_atomF = False
    if get_atomF:
        Converter.get_atomwise_decompF(molecule, bias_type='NRF')
        recompF = Converter.get_atomwise_recompF([molecule.coords[0]], 
                [molecule.atom_F[0]], molecule.atoms, n_atoms, _NC2)
        print('atom_NRF', molecule.atom_NRF[0])
        print('atom_F', molecule.atom_F[0])
        print('recompF', recompF)
        print()
        print('F', molecule.forces[0])
        print(molecule.atom_NRF.shape, molecule.atom_F.shape)
        #molecule.atom_NRF = molecule.atom_NRF[:,0]
        #molecule.atom_F = molecule.atom_F[:,0]
        #sys.exit()
        sys.stdout.flush()

    get_decompF = False
    if get_decompF:
        print(molecule)
        print('\ncheck and get force decomp')
        unconserved = Molecule.check_force_conservation(molecule) #
        Converter(molecule) #get pairwise forces
        print('mat_r', molecule.mat_r[0])
        print('mat_NRF', molecule.mat_NRF[0])
        print('mat_F', molecule.mat_F[0])
        recompF = Conservation.get_recomposed_forces([molecule.coords[0]], 
                [molecule.mat_F[0]], n_atoms, _NC2)
        print('recompF', recompF)
        print('forces', molecule.forces[0])
        print(datetime.now() - startTime)
        sys.stdout.flush()
        #sys.exit()

    get_decompE = False
    bias_type = '1/r' # 'NRF' # 'qq/r2' #
    print('\nenergy bias type: {}'.format(bias_type))
    if get_decompE:
        print('\nget mat_E')
        Converter.get_interatomic_energies(molecule, bias_type)
        print('molecule energy', molecule.energies[0])
        print('mat_E', molecule.mat_E[0])
        print('sum mat_E', np.sum(molecule.mat_E[0]))
        print('recompE', np.dot(molecule.mat_bias[0], molecule.mat_E[0]))

        lower_mask = np.tri(n_atoms, dtype=bool, k=-1) #True False mask
        #print(lower_mask)
        out = np.zeros((n_atoms, n_atoms))

        molecule.atomE = np.zeros((len(molecule.mat_E), n_atoms))
        for i in range(len(molecule.mat_E)):
            out_copy = np.copy(out)
            out_copy[lower_mask] = molecule.mat_E[i]
            ult = out_copy + out_copy.T
            atomE = np.sum(ult, axis=0) / 2
            molecule.atomE[i] = atomE

        out[lower_mask] = molecule.mat_E[0]
        out3 = out + out.T
        print('upper lower triangle out3\n', out3)
        print('molecule.atomE[0]\n', molecule.atomE[0])
        atomE = np.sum(out3, axis=0) / 2
        print('column sums, atomE\n', atomE)
        print(molecule.atoms)
        print('sum atomE\n', np.sum(atomE))

        print(datetime.now() - startTime)
        sys.stdout.flush()
        #sys.exit()


    get_decompFE = True
    if get_decompFE:

        print('\nget decomposed forces and energies simultaneously')

        #for some reason, using r as the bias does not give back recomp 
        #values, no idea why!
        Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                bias_type)
        '''
        #check rot invariance
        print('coords\n', molecule.coords[0])
        Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                bias_type)
        print('mat_FE\n', molecule.mat_FE[0])
        recompF = np.dot(molecule.mat_eij[0][:-1], molecule.mat_FE[0] / 
                molecule.mat_eij[0][-1]) #add back bias
        print('recompF\n', recompF)
        print('forces\n', molecule.forces[0])
        Writer.write_xyz([molecule.coords[0]], molecule.atoms, 
                'rot-coords.xyz', 'w', 1)
        print('\n\n\n')
        #rotate molecule
        molecule.coords[0] = molecule.coords[0] * -1
        molecule.forces[0] = molecule.forces[0] * -1
        print('coords\n', molecule.coords[0])
        Converter.get_simultaneous_interatomic_energies_forces(molecule, 
                bias_type)
        print('mat_FE\n', molecule.mat_FE[0])
        recompF = np.dot(molecule.mat_eij[0][:-1], molecule.mat_FE[0] / 
                molecule.mat_eij[0][-1]) #add back bias
        print('recompF\n', recompF)
        print('forces\n', molecule.forces[0])
        Writer.write_xyz([molecule.coords[0]], molecule.atoms, 
                'rot-coords.xyz', 'a', 2)
        '''

        '''
        #check angleNRF
        coords = molecule.coords[0].reshape(n_atoms, 3)
        atoms = np.ones(n_atoms).reshape(n_atoms, 1)
        print('coords\n', coords)
        print('atoms\n', atoms)
        com = np.sum(coords * atoms, axis=0) / np.sum(atoms)
        print('com\n', com)
        com2 = Converter.get_com(coords, molecule.atoms)
        print('com2\n', com2)
        vectors = molecule.coords[0] - com
        print('vectors\n', vectors)

        #a = np.expand_dims(vectors, axis=1)
        #b = np.expand_dims(vectors, axis=0)
        top = np.dot(vectors, vectors.T)
        print('top\n', top.shape, top)
        unit_mag = np.linalg.norm(vectors, axis=1).reshape(n_atoms, 1)
        print('unit_mag\n', unit_mag.shape, unit_mag)
        unit_mag2 = np.sum(vectors**2, axis=1) ** 0.5
        print('unit_mag2\n', unit_mag2.shape, unit_mag2)
        #bottom = np.einsum('ij,ji', unit_mag,  unit_mag.T)
        bottom = unit_mag * unit_mag.T
        print('bottom\n', bottom.shape, bottom)
        cos_theta = top / bottom
        print('cos_theta\n', cos_theta)



        for i in range(n_atoms):
            for j in range(i):
                r = Converter.get_r(coords[i], coords[j])
                NRF = Converter.get_NRF(molecule.atoms[i], 
                        molecule.atoms[j], r)
                cos_theta = Converter.get_angle(coords[i], com, coords[j])

                v1 = coords[i] - com
                v2 = coords[j] - com
                r1 = np.linalg.norm(v1)
                r2 = np.linalg.norm(v2)
                top = np.dot(v1, v2)
                bottom = np.multiply(r1, r2)
                val = np.divide(top, bottom)
                print('v1 {}, v2 {}, r1 {}, r2 {}, top {}, bottom {}, '\
                        'cos_theta {}'.format(
                        v1, v2, r1, r2, top, bottom, val))

                #print(NRF, cos_theta)
                angleNRF = cos_theta * NRF
                #print(angleNRF)
        '''



        '''
        atom_names = ['{}{}'.format(Converter._ZSymbol[z], n) for z, n in 
                zip(molecule.atoms, range(1,len(molecule.atoms)+1))]
        atom_pairs = []
        for i in range(len(molecule.atoms)):
            for j in range(i):
                atom_pairs.append('{}_{}'.format(atom_names[i], 
                    atom_names[j]))
        input_header = ['input_' + s for s in atom_pairs]
        input_header = ','.join(input_header)
        output_header = ['output_' + s for s in atom_pairs]
        output_header = ','.join(output_header)
        prediction_header = ['prediction_' + s for s in atom_pairs]
        prediction_header = ','.join(prediction_header)
        header = input_header #+ ',' + output_header + ',' + \
                #prediction_header

        Writer.write_csv([
                molecule.mat_NRF,
                #output_atomNRF, 
                #output_atomFE, 
                #output_matFE,
                #prediction_atomFE
                #prediction_matFE
                ], 'inp', 
                header)



        for k in range(3):
            Writer.write_csv([
                    #molecule.mat_NRF,
                    molecule.vector_NRF[:,k,],
                    #output_atomNRF, 
                    #output_atomFE, 
                    #output_matFE,
                    #prediction_atomFE
                    #prediction_matFE
                    ], 'inp{}'.format(k), 
                    header)
        '''
        #sys.exit()



        '''
        ###tensorflow
        print('tensorflow')
        coords = tf.convert_to_tensor(coords, np.float32)
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r = diff_flat**0.5
        print(r.shape)

        centre = tf.reduce_sum(coords, axis=0) / \
                tf.cast(tf.shape(coords)[0], tf.float32)
        #print(coords.shape, centre.shape)
        vectors = coords - centre
        #print(vectors.shape)
        #unit_mags = tf.einsum('bij, bij -> b', vectors, vectors) ** 0.5
        unit_mags1 = tf.reduce_sum(vectors**2, axis=1) ** 0.5
        unit_mags = tf.expand_dims(unit_mags1, -1)
        numerator = tf.einsum('ij, jk -> ik', vectors, 
                tf.transpose(vectors))
        denominator = tf.einsum('ij, jk -> ik', unit_mags, 
                tf.transpose(unit_mags))
        #denominator = tf.matmul(unit_mags, tf.transpose(unit_mags)) #works too
        cos_theta = numerator / denominator

        tri1 = tf.linalg.band_part(cos_theta, -1, 0) #lower
        diag = tf.linalg.band_part(cos_theta, 0, 0) #diag of ones
        tri2 = tri1 - diag

        #print(tri1.shape, diag.shape, tri2.shape)


        nonzero_indices2 = tf.where(tf.not_equal(tri2, tf.zeros_like(tri2)))
        nonzero_values2 = tf.gather_nd(tri2, nonzero_indices2)
        cos_theta_flat = tf.reshape(nonzero_values2, 
                shape=(1, _NC2))  #reshape to _NC2
        #print(nonzero_indices2.shape, nonzero_values2.shape)
        print(cos_theta.shape, cos_theta_flat.shape)
        #with tf.Session() as sess: #to get tensor to numpy
            #sess.run(tf.global_variables_initializer())
            #result = sess.run(cos_theta_flat)
            #print('result\n', result)
        sess = tf.compat.v1.Session()
        print(sess.run(cos_theta_flat))
        print('exit')
        sys.exit()
        '''


        def Triangle(decompFE):
            '''Convert flat NC2 to lower and upper triangle sq matrix, this
            is used in get_FE_eij_matrix to get recomposedFE
            https://stackoverflow.com/questions/40406733/
                    tensorflow-equivalent-for-this-matlab-code
            '''
            #decompFE = tf.convert_to_tensor(decompFE, dtype=tf.float32)
            #rescale decompFE
            #decompFE = ((decompFE - 0.5) * (2 * self.max_FE))

            #decompFE.get_shape().with_rank_at_least(1)

            #put batch dimensions last
            decompFE = tf.transpose(decompFE, tf.concat([[tf.rank(decompFE)-1],
                    tf.range(tf.rank(decompFE)-1)], axis=0))
            input_shape = tf.shape(decompFE)[0]
            #compute size of matrix that would have this upper triangle
            matrix_size = (1 + tf.cast(tf.sqrt(tf.cast(input_shape*8+1, 
                    tf.float32)), tf.int32)) // 2
            matrix_size = tf.identity(matrix_size)
            #compute indices for whole matrix and upper diagonal
            index_matrix = tf.reshape(tf.range(matrix_size**2), 
                    [matrix_size, matrix_size])


            tri1 = tf.linalg.band_part(index_matrix, -1, 0) #lower
            diag = tf.linalg.band_part(index_matrix, 0, 0) #diag of ones
            tri2 = tri1 - diag #get lower without diag of ones
            nonzero_indices = tf.where(tf.not_equal(tri2, tf.zeros_like(tri2)))
            nonzero_values = tf.gather_nd(tri2, nonzero_indices)
            reshaped_nonzero_values = tf.reshape(nonzero_values, [-1])

            '''
            diagonal_indices = (matrix_size * tf.range(matrix_size)
                    + tf.range(matrix_size))
            upper_triangular_indices, _ = tf.unique(tf.reshape(
                    #tf.matrix_band_part(index_matrix, -1, 0) #v1
                    tf.linalg.band_part(index_matrix, -1, 0)
                    - tf.linalg.diag(diagonal_indices), [-1]))
            '''
            batch_dimensions = tf.shape(decompFE)[1:]
            return_shape_transposed = tf.concat([[matrix_size, matrix_size],
                    batch_dimensions], axis=0)
            #fill everything else with zeros; later entries get priority
            #in dynamic_stitch
            result_transposed = tf.reshape(tf.dynamic_stitch([index_matrix,
                    #upper_triangular_indices[1:]
                    reshaped_nonzero_values
                    ],
                    [tf.zeros(return_shape_transposed, dtype=decompFE.dtype),
                    decompFE]), return_shape_transposed)
            #Transpose the batch dimensions to be first again
            Q = tf.transpose(result_transposed, tf.concat(
                    [tf.range(2, tf.rank(decompFE)+1), [0,1]], axis=0))
            Q2 = tf.transpose(result_transposed, tf.concat(
                    [tf.range(2, tf.rank(decompFE)+1), [1,0]], axis=0))
            Q3 = Q + Q2

            return Q3


        def FlatTriangle(tri):
            tri = tf.linalg.band_part(tri, -1, 0)
            nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
            nonzero_values = tf.gather_nd(tri, nonzero_indices)
            reshaped_nonzero_values = tf.reshape(nonzero_values, 
                    shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
            return reshaped_nonzero_values



        print('tensorflow2')
        sess = tf.compat.v1.Session()
        n_atoms = len(molecule.atoms)
        _NC2 = int(n_atoms*(n_atoms-1)/2)

        atoms_flat = []
        for i in range(n_atoms):
            for j in range(i):
                ij = molecule.atoms[i] * molecule.atoms[j]
                atoms_flat.append(ij)
        atoms_flat = tf.convert_to_tensor(atoms_flat, dtype=tf.float32) #_NC2



        coords = molecule.coords[0:3].reshape(-1, n_atoms, 3)
        F = molecule.forces[0:3].reshape(-1, n_atoms, 3)
        E = molecule.energies[0:3].reshape(-1, 1)
        coords = tf.convert_to_tensor(coords, np.float32)
        F = tf.convert_to_tensor(F, np.float32)
        E = tf.convert_to_tensor(E, np.float32)
        F_reshaped = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        FE = tf.concat([F_reshaped, E], axis=1)
        #print('FE', FE.shape)

        coords_F_E = coords, F, E

        coords, F, E  = coords_F_E
        F_reshaped = tf.reshape(F, shape=(tf.shape(F)[0], -1))
        FE = tf.concat([F_reshaped, E], axis=1)


        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, 
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r_flat = diff_flat**0.5
        au2kcalmola = 627.5095 * 0.529177
        au2kcalmola = tf.constant(au2kcalmola, dtype=tf.float32)
        _NRF = ((atoms_flat) / (r_flat ** 2))

        '''
        triangle_qqr2 = Triangle(_NRF)
        triangle_r = Triangle(r_flat)
        Rc = 5
        nu = 4
        Rs = 0
        less_mask = tf.math.less_equal(triangle_r, Rc) #Bools inside
        less_r = triangle_r * tf.cast(less_mask, dtype=tf.float32)
        fc_all = 0.5 * tf.math.cos(math.pi * triangle_r / Rc) + 0.5
        fc = fc_all * tf.cast(less_mask, dtype=tf.float32) #only keep less rc
        zeros = tf.zeros([fc.shape[0], fc.shape[1]])
        fc = tf.linalg.set_diag(fc, zeros) #make diags zero
        Gij = tf.math.exp(-nu * (triangle_qqr2 - Rs) ** 2) * fc
        Gij = tf.reduce_sum(Gij, 1)

        #print('zeros', zeros.shape, sess.run(zeros))
        print('Gij', sess.run(Gij))
        print(molecule.acsf[0:3])
        print('fc', sess.run(fc))
        print('less_r', sess.run(less_r))
        print('triangle_r', sess.run(triangle_r))
        #sys.exit()
        '''


        print('_NRF', _NRF.shape)
        print(sess.run(_NRF))
        print(molecule.mat_NRF[:3])


        #FOR ENERGY get energy 1/r_ij eij matrix
        recip_r = 1 / r_flat
        recip_r2 = 1 / r_flat ** 2
        if bias_type == '1/r':
            eij_E = tf.expand_dims(recip_r, 1)
        if bias_type == 'qq/r2':
            eij_E = tf.expand_dims(_NRF, 1)
        #print('eij_E', sess.run(eij_E))
        norm_recip_r = tf.reduce_sum(recip_r2, axis=1) ** 0.5
        norm_recip_r = tf.expand_dims(norm_recip_r, 1)
        norm_recip_r = tf.expand_dims(norm_recip_r, 2)
        #print('!!!!!! normalising eij_E with norm(1/R)')
        #eij_E = eij_E / norm_recip_r
        #print('norm', sess.run(norm_recip_r))
        #sys.exit()

        #### FOR FORCES
        r = Triangle(r_flat)
        r2 = tf.expand_dims(r, 3)
        eij_F2 = diff / r2
        eij_F = tf.where(tf.math.is_nan(eij_F2), tf.zeros_like(eij_F2), 
                eij_F2) #remove nans 

        new_eij_F = []
        n_atoms = coords.shape.as_list()[1]
        for i in range(n_atoms):
            for x in range(3):
                atom_i = eij_F[:,i,:,x]
                s = []
                _N = -1
                count = 0
                a = [k for k in range(n_atoms) if k != i]
                for i2 in range(n_atoms):
                    for j2 in range(i2):
                        _N += 1
                        if i2 == i:
                            s.append(atom_i[:,a[count]])
                            count += 1
                        elif j2 == i:
                            s.append(atom_i[:,a[count]])
                            count += 1
                        else:
                            s.append(tf.zeros_like(atom_i[:,0]))
                s = tf.stack(s)
                s = tf.transpose(s)
                new_eij_F.append(s)

        eij_F = tf.stack(new_eij_F)
        #eij_F = eij_F * recip_r2
        #print('!!! 1/r2 bias included to eij_F matrix')
        eij_F = tf.transpose(eij_F, perm=[1,0,2])
        eij_FE = tf.concat([eij_F, eij_E], axis=1)


        inv_eij = tf.linalg.pinv(eij_FE)
        qs = tf.linalg.matmul(inv_eij, tf.transpose(FE))
        qs = tf.transpose(qs, perm=[1,0,2])
        qs = tf.linalg.diag_part(qs)
        qs = tf.transpose(qs) #* recip_r

        print('qs', qs.shape)
        print(sess.run(qs))
        print()
        print(molecule.mat_FE[0:3])
        print('max_FE', np.max(np.abs(molecule.mat_FE.flatten())))



        if bias_type == '1/r':
            Q3 = Triangle(recip_r)
        if bias_type == 'qq/r2':
            Q3 = Triangle(_NRF)
        #Q3 = Triangle(_NRF)
        #Q3 = Triangle(recip_r)
        eij_E = tf.expand_dims(Q3, 3)
        #dot product of
        qs_tri = Triangle(qs)
        E2 = tf.einsum('bijk, bij -> bk', eij_E, qs_tri)
        E_tf = E2/2
        #'''

        print('E_tf', E_tf.shape)
        print(sess.run(E_tf))
        print()
        print(molecule.energies[0:3])



        print(datetime.now() - startTime)
        #sys.exit()


        '''
        #save to file for Neil
        kj2kcal = 1/4.184
        au2kcalmola = Converter.Eh2kcalmol / Converter.au2Ang
        print(molecule.forces[-1])
        #np.savetxt('matrix_FE.dat', (molecule.mat_FE[-1]/kj2kcal).reshape(-1,_NC2))
        #np.savetxt('F.dat', (molecule.forces[-1]/au2kcalmola).reshape(-1,3))
        #np.savetxt('E.dat', (molecule.energies[-1]/kj2kcal).reshape(-1,1))
        #np.savetxt('matrix_eij.dat', molecule.mat_eij[-1].reshape(-1,_NC2))

        np.savetxt('q.dat', (molecule.mat_FE[:]).reshape(-1,_NC2))
        np.savetxt('C.dat', (molecule.coords[:]).reshape(-1,3))
        np.savetxt('F.dat', (molecule.forces[:]).reshape(-1,3))
        np.savetxt('E.dat', (molecule.energies[:]).reshape(-1,1))
        np.savetxt('eij_FE.dat', molecule.mat_eij[:].reshape(-1,_NC2))
        #sys.exit()
        '''
        for i in range(1):
            print('\ni', i)
            print('\nmat_NRF', molecule.mat_NRF[i])
            print('\nmat_FE', molecule.mat_FE[i])
            print('\nsum mat_FE', np.sum(molecule.mat_FE[i]))
            print('\nget recomposed FE')
            print('actual')
            print(molecule.forces[i])
            print(molecule.energies[i])
            n_atoms = len(molecule.atoms)
            _NC2 = int(n_atoms*(n_atoms-1)/2)
            recompF, recompE = Converter.get_recomposed_FE(
                    [molecule.coords[i]], 
                    [molecule.mat_FE[i]], 
                    #[molecule.mat_FE[i] / molecule.mat_eij[i][-1]], #add back bias 
                    molecule.atoms, n_atoms, _NC2, bias_type)
            print('\nrecomp from FE')
            print(recompF)
            print(recompE)
            print('\nrecomp from FE with dot')
            #recompF = np.dot(molecule.mat_eij[i][:-1], molecule.mat_FE[i] / 
                    #molecule.mat_eij[i][-1]) #add back bias
            recompF = np.dot(molecule.mat_eij[i][:-1], molecule.mat_FE[i])
            recompE = np.dot(molecule.mat_bias[i], molecule.mat_FE[i])

            print(recompF)
            print(recompE)
            '''
            print('\nrecomp from F only')
            recompF2 = Conservation.get_recomposed_forces(
                [molecule.coords[i]], [molecule.mat_F[i]], n_atoms, _NC2)
            print(recompF2)
            '''


        #'''
        lower_mask = np.tri(n_atoms, dtype=bool, k=-1) #True False mask
        #print(lower_mask)
        out = np.zeros((n_atoms, n_atoms))

        molecule.atomFE = np.zeros((len(molecule.mat_FE), n_atoms))
        molecule.atomNRF = np.zeros((len(molecule.mat_NRF), n_atoms))
        for j in range(len(molecule.mat_FE)):
            out_copy = np.copy(out)
            out_copy[lower_mask] = molecule.mat_FE[j]
            ult = out_copy + out_copy.T
            atomFE = np.sum(ult, axis=0) / 2
            molecule.atomFE[j] = atomFE
            if j == 0:
                print('upper lower triangle ult atomFE\n', ult)
            out_copy2 = np.copy(out)
            out_copy2[lower_mask] = molecule.mat_NRF[j]
            ult = out_copy2 + out_copy2.T
            atomNRF = np.sum(ult, axis=0) / 2
            molecule.atomNRF[j] = atomNRF
            if j == 0:
                print('upper lower triangle ult atomNRF\n', ult)
        #np.savetxt('atomFE.txt', molecule.atomFE)
        #np.savetxt('atomNRF.txt', molecule.atomNRF)
        print('column sums, molecule.atomFE\n', molecule.atomFE[0])
        out[lower_mask] = molecule.mat_FE[0]
        out3 = out + out.T
        print('\nupper lower triangle out3\n', out3)
        atomFE = np.sum(out3, axis=0) / 2
        print('column sums, atomFE', atomFE)
        print(molecule.atoms)
        print('sum atomFE', np.sum(atomFE))
        print('atomNRF', molecule.atomNRF[0])
        #'''

        print(datetime.now() - startTime)
        sys.stdout.flush()
        #sys.exit()

    #molecule.mat_FE = molecule.mat_F
    #print('\n!!!!!!!!! *** using mat_E instead of mat_FE *** !!!!!!!!!!!')
    #sys.exit()

    '''
    decompE = Converter.get_energy_decomposition(molecule.atoms, 
            molecule.coords[0], molecule.energies[0])
    recompE = Converter.get_recomposed_energy(molecule.coords[0], 
            decompE, n_atoms, _NC2)
    print(molecule.energies[0])
    print(decompE)
    print(recompE)
    sys.exit()
    '''

    #'''
    print('\ninternal FE decomposition')
    network = Network(molecule)
    model = Network.get_coord_FE_model(network, molecule, prescale)
    print('exit')
    print(datetime.now() - startTime)
    sys.stdout.flush()
    sys.exit()
    #'''

    run_net = True
    split = 2#0 #2 4 5 20 52 260
    if run_net:
        train = round(len(molecule.coords) / split, 3)
        nodes = 1000
        input = molecule.mat_NRF #atom_NRF #
        output = molecule.mat_FE #atom_F #
        #output = molecule.energies
        print('\nget train and test sets, '\
                'training set is {} points.'\
                '\nNumber of nodes is {}'.format(train, nodes))
        Molecule.make_train_test_old(molecule, molecule.energies.flatten(), 
                split) #get train and test sets
        print('\nget ANN model')
        sys.stdout.flush()
        network = Network(molecule) #initiate network class
        train_prediction, test_prediction = Network.get_variable_depth_model(
                network, molecule, nodes, input, output) #train std NN
        #Network.get_decompE_sum_model(network, molecule, 
                #nodes, input, output) #train NN with sum energies

    sys.stdout.flush()


    train_input = np.take(molecule.coords, molecule.train, axis=0)
    train_e = np.take(molecule.energies, molecule.train, axis=0)
    train_f = np.take(molecule.forces, molecule.train, axis=0)

    test_input = np.take(molecule.coords, molecule.test, axis=0)
    test_e = np.take(molecule.energies, molecule.test, axis=0)
    test_f = np.take(molecule.forces, molecule.test, axis=0)


    if get_atomF:
        #n_atoms = 1
        #train_f = train_f[:,0]
        #test_f = test_f[:,0]
        train_pred_f = Converter.get_atomwise_recompF(
                train_input, train_prediction.reshape(-1,n_atoms,_NC2), 
                molecule.atoms, n_atoms, _NC2)
        test_pred_f = Converter.get_atomwise_recompF(
                test_input, test_prediction.reshape(-1,n_atoms,_NC2), 
                molecule.atoms, n_atoms, _NC2)

        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                    train_pred_f.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                    test_pred_f.flatten())
        print('\nForces:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

    if get_decompE:
        train_pred_e = np.sum(train_prediction, axis=1)
        test_pred_e = np.sum(test_prediction, axis=1)
        train_mae, train_rms = Binner.get_error(train_e.flatten(), 
                    train_pred_e.flatten())
        test_mae, test_rms = Binner.get_error(test_e.flatten(), 
                    test_pred_e.flatten())
        print('\nEnergies:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))
        #print(train_e, train_pred_e)

    if get_decompF:
        train_pred_f = Conservation.get_recomposed_forces(train_input, 
                train_prediction, n_atoms, _NC2)
        test_pred_f = Conservation.get_recomposed_forces(test_input, 
                test_prediction, n_atoms, _NC2)
        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                    train_pred_f.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                    test_pred_f.flatten())
        print('\nForces:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

    if get_decompFE:
        train_pred_f, train_pred_e = Converter.get_recomposed_FE(train_input, 
                train_prediction, molecule.atoms, n_atoms, _NC2, bias_type)
        test_pred_f, test_pred_e = Converter.get_recomposed_FE(test_input, 
                test_prediction, molecule.atoms, n_atoms, _NC2, bias_type)

        #print(train_e, train_pred_e)
        train_mae, train_rms = Binner.get_error(train_e.flatten(), 
                    train_pred_e.flatten())
        test_mae, test_rms = Binner.get_error(test_e.flatten(), 
                    test_pred_e.flatten())
        print('\nEnergies:\nTrain MAE: {} \nTrain RMS: {} \nTest MAE: {} '\
                '\nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))

        train_e = np.subtract(train_e / prescale[1], prescale[0])
        train_pred_e = np.subtract(train_pred_e / prescale[1], prescale[0])
        test_e = np.subtract(test_e / prescale[1], prescale[0])
        test_pred_e = np.subtract(test_pred_e / prescale[1], prescale[0])
        #print(train_e, train_pred_e)
        #print(train_f[0], train_pred_f[0])
        train_mae, train_rms = Binner.get_error(train_e.flatten(), 
                    train_pred_e.flatten())
        test_mae, test_rms = Binner.get_error(test_e.flatten(), 
                    test_pred_e.flatten())
        print('\nEnergies kcal/mol Ang:\nTrain MAE: {} \nTrain RMS: {} '\
                '\nTest MAE: {} \nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))
        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                    train_pred_f.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                    test_pred_f.flatten())
        print('\nForces from mat_FE:\nTrain MAE: {} \nTrain RMS: {} '\
                '\nTest MAE: {} \nTest RMS: {}'.format(train_mae, train_rms, 
                test_mae, test_rms))


    scurves = True
    if scurves:
        train = None
        test = None
        train_pred = None
        test_pred = None
        if get_atomF or get_decompF or get_decompFE:
            train = train_f
            test = test_f
            train_pred = train_pred_f
            test_pred = test_pred_f
        if get_decompE: #get_decompFE or 
            train = train_e
            test = test_e
            train_pred = train_pred_e
            test_pred = test_pred_e

        print(train.shape, train_pred.shape, test.shape, test_pred.shape)

        train_bin_edges, train_hist = Binner.get_scurve(train.flatten(), 
                train_pred.flatten(), 'train-hist.txt')

        test_bin_edges, test_hist = Binner.get_scurve(test.flatten(), #actual
                test_pred.flatten(), #prediction
                'test-hist.txt')
       
        Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                [train_hist, test_hist], ['train', 'test'], 
                'Error', '% of points below error', 's-curves-recomp.png')

    #'''
    print('check decompE')
    ###malonaldehyde 
    #scale_NRF = 13088.721638547617
    #scale_NRF_min = 14.486865558162252
    ###decompE, 1/r 
    #scale_E = 18838.72967827923
    #scale_E_min = 679.9222840668109
    ###decompE, 1/qqr2
    #scale_E = 81838.51564994424 
    #scale_E_min = 1.8099282995681707

    #ethanediol r_6layers, syn scan
    #scale_NRF = 7990.182422267673 
    #scale_NRF_min = 16.93017848874296
    #bias r
    #scale_E = 10391.176054613952 
    #scale_E_min = 496.02300749772684 
    #bias 1/r
    #scale_E = 11947.537628797449 
    #scale_E_min = 570.8028197171167

    scale_NRF = network.scale_input_max 
    scale_NRF_min = network.scale_input_min
    scale_E = network.scale_output_max
    scale_E_min = network.scale_output_min
    dr = 0.001

    print('get network')
    network = Network.get_network(molecule, scale_NRF, scale_NRF_min, 
            scale_E, scale_E_min)

    #model = load_model('best_ever_model')

    c_loss = False
    if c_loss:
        def custom_loss1(weights):
            def custom_loss(y_true, y_pred):
                return K.mean(K.abs(y_true - y_pred) * weights)
            return custom_loss

        weights = np.zeros((_NC2+1)) #np.ones((_NC2+1)) #
        weights[-1] = 1 #sumE_weight
        cl = custom_loss1(weights)
        model = load_model('best_ever_model', 
                custom_objects={'custom_loss': custom_loss1(weights)})
    else:
        model = load_model('best_ever_model')
    #model = None
    sys.stdout.flush()

    if get_decompFE:
        print('get forces from finite difference')
        molecule.energies = np.subtract(molecule.energies / prescale[1], 
                prescale[0])
        #for i in range(len(molecule.coords)):
        for i in range(1):
            print('\ni', i)
            print('actual energy', molecule.energies[i])
            print('actual forces', molecule.forces[i])

            forces, curl = Conservation.get_forces_from_energy(
                    molecule.coords[i], 
                    molecule.atoms, scale_NRF, scale_NRF_min, scale_E, 
                    model, dr, bias_type, molecule, prescale)
            print('pred forces', forces)
            sys.stdout.flush()

        #get all Fs from Es with finite diff and check errors
        train_pred_f2 = np.zeros((len(train_input), n_atoms, 3))
        test_pred_f2 = np.zeros((len(test_input), n_atoms, 3))
        for s in range(len(train_input)):
            forces, curl = Conservation.get_forces_from_energy(
                    train_input[s], 
                    molecule.atoms, scale_NRF, scale_NRF_min, scale_E, 
                    model, dr, bias_type, molecule, prescale)
            train_pred_f2[s] = forces[0]
        for s in range(len(test_input)):
            forces, curl = Conservation.get_forces_from_energy(
                    test_input[s], 
                    molecule.atoms, scale_NRF, scale_NRF_min, scale_E, 
                    model, dr, bias_type, molecule, prescale)
            test_pred_f2[s] = forces[0]

        train_mae, train_rms = Binner.get_error(train_f.flatten(), 
                train_pred_f2.flatten())
        test_mae, test_rms = Binner.get_error(test_f.flatten(), 
                test_pred_f2.flatten())

        print(train_pred_f2.shape, test_pred_f2.shape)

        print('\nForces from Es:\nTrain MAE: {} \nTrain RMS: {}'\
                '\nTest MAE: {} \nTest RMS: {}'.format(
                train_mae, train_rms, test_mae, test_rms))

        train_bin_edges, train_hist = Binner.get_scurve(train_f.flatten(), 
                train_pred_f2.flatten(), 'train-hist2.txt')

        test_bin_edges, test_hist = Binner.get_scurve(test_f.flatten(),
                test_pred_f2.flatten(),
                'test-hist2.txt')
       
        Plotter.plot_2d([train_bin_edges, test_bin_edges], 
                [train_hist, test_hist], ['train', 'test'], 
                'Error', '% of points below error', 's-curves-recomp2.png')

    #'''


    print(datetime.now() - startTime)


def main():

    try:
        usage = 'runForcePred.py [-h]'
        parser = argparse.ArgumentParser(description='Program for reading '\
                'in molecule forces, coordinates and energies for '\
                'force prediction.', usage=usage, 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group('Options')
        group = parser.add_argument('-i', '--input_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing forces '\
                'coordinates and energies.')
        group = parser.add_argument('-a', '--atom_file', 
                metavar='file', default=None,
                help='name of file/s containing atom nuclear charges.')
        group = parser.add_argument('-c', '--coord_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing coordinates.')
        group = parser.add_argument('-f', '--force_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing forces.')
        group = parser.add_argument('-e', '--energy_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing energies.')
        group = parser.add_argument('-q', '--charge_files', nargs='+', 
                metavar='file', default=[],
                help='name of file/s containing charges.')
        group = parser.add_argument('-l', '--list_files', action='store', 
                metavar='file', default=False,
                help='file containing list of file paths.')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    run_force_pred(input_files=op.input_files, atom_file=op.atom_file, 
            coord_files=op.coord_files, force_files=op.force_files, 
            energy_files=op.energy_files, charge_files=op.charge_files, 
            list_files=op.list_files)

if __name__ == '__main__':
    main()


