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

openmm = True
if openmm:
    from ForcePred.calculate.OpenMM import OpenMM

mdanal = False
if mdanal:
    from ForcePred.read.AMBLAMMPSParser import AMBLAMMPSParser
    from ForcePred.read.AMBERParser import AMBERParser
                               
from ForcePred import Molecule, OPTParser, NPParser, Converter, Preprocess, \
        Permuter, XYZParser, Binner, Writer, Plotter, MultiPlotter, Conservation
        #Network
from ForcePred.nn.Network_shared_ws import Network
from keras.models import Model, load_model    
from keras import backend as K                                              
import sys
import tensorflow as tf
import os
import math

NUMCORES=int(os.getenv('NSLOTS', 1))
print('Using', NUMCORES, 'core(s)' )


tf.config.threading.set_inter_op_parallelism_threads(NUMCORES) 
tf.config.threading.set_intra_op_parallelism_threads(NUMCORES)
tf.config.set_soft_device_placement(1)


def run_force_pred(input_files='input_files', input_paths='input_paths',
        atom_file='atom_file', coord_files='coord_files',
        force_files='force_files', energy_files='energy_files',
        charge_files='charge_files', list_files='list_files', 
        n_nodes='n_nodes', n_layers='n_layers', n_training='n_training', 
        n_val='n_val', n_test='n_test', grad_loss_w='grad_loss_w', 
        qFE_loss_w='qFE_loss_w', E_loss_w='E_loss_w', bias='bias',
        filtered='filtered', load_model='load_model', dihedrals='dihedrals',
        model_num='model_num'):
    """
    If running this script locally on mac, then use the new-forcepred conda
    environment.
    """

    startTime = datetime.now()
    print(startTime)

    """
    # Energy vs time plot example
    filepath = input_paths[0]
    md17path = os.path.join(filepath, "malonaldehyde/revised_data")
    malon_ml = Molecule() # ML sim, 500K, gas
    mlpath = os.path.join(filepath, "malonaldehyde/load108_gas")
    NPParser(os.path.join(md17path, "nuclear_charges.txt"), 
             [os.path.join(mlpath, "openmm-coords.txt")], 
             [os.path.join(mlpath, "openmm-forces.txt")], 
             [os.path.join(mlpath, "openmm-delta_energies.txt")], malon_ml)
    # get timestep in picoseconds
    timesteps = np.arange(0, len(malon_ml.energies), 4) / 200
    Plotter.xy_scatter([timesteps], [malon_ml.energies[::4]], [''], ['k'], '$t$ / ps',
                    '$\Delta$E / kcal/mol', [10], 'scatter-time-E.pdf')
    sys.exit()
    """

    """
    # salicylic acid scan
    molecule = Molecule()
    filepath2 = input_paths[1]
    OPTParser([os.path.join(filepath2, "salicylic_10_3_scan.out")], molecule, opt=True) #read in FCEZ for SP
    print(molecule)
    measure = Binner()
    measure.get_dih_pop(molecule.coords, [dihedrals[7]]) # [[9, 0, 1, 2]])
    delta_es = molecule.energies - np.min(molecule.energies)
    # print(measure.phis)
    # print(delta_es)
    Plotter.xy_scatter([measure.phis.T[0]], [delta_es], [''], ['k'], "$\phi / ^{\circ}$",
                '$\Delta E$ / kcal/mol', [40], 'salicylic_scan.pdf')
    #sys.exit()
    """

    """
    # plot dihedrals for malonaldehyde: rMD17, ML 500K and ML/MM 300K
    malon_md17 = Molecule() # rMD17, 500K, gas
    filepath = input_paths[0]
    md17path = os.path.join(filepath, "malonaldehyde/revised_data")
    NPParser(os.path.join(md17path, "nuclear_charges.txt"), 
             [os.path.join(md17path, "coords.txt")], 
             [os.path.join(md17path, "forces.txt")], 
             [os.path.join(md17path, "energies.txt")], malon_md17)
    malon_ml = Molecule() # ML sim, 500K, gas
    mlpath = os.path.join(filepath, "malonaldehyde/load108_gas")
    NPParser(os.path.join(md17path, "nuclear_charges.txt"), 
             [os.path.join(mlpath, "openmm-coords.txt")], 
             [os.path.join(mlpath, "openmm-forces.txt")], 
             [os.path.join(mlpath, "openmm-delta_energies.txt")], malon_ml)
    malon_mlmm = Molecule() # ML/MM sim, 300K, aq
    mlmmpath = os.path.join(filepath, "malonaldehyde/load108_300K_soln")
    NPParser(os.path.join(md17path, "nuclear_charges.txt"), 
             [os.path.join(mlmmpath, "openmm-coords.txt")], 
             [os.path.join(mlmmpath, "openmm-forces.txt")], 
             [os.path.join(mlmmpath, "openmm-delta_energies.txt")], malon_mlmm)
    malon_qmmm = Molecule() # QM/MM cp2k sim, gaus calcs, 500K, aq
    filepath2 = input_paths[3]
    qmmmpath = os.path.join(filepath2, "malonaldehyde")
    NPParser(os.path.join(md17path, "nuclear_charges.txt"), 
             [os.path.join(qmmmpath, "coords.txt")], 
             [os.path.join(qmmmpath, "forces.txt")], 
             [os.path.join(qmmmpath, "energies.txt")], malon_qmmm)
    


    pairs = []
    _N = 0
    for i in range(len(malon_md17.atoms)):
        for j in range(i):
            pairs.append([i, j])
            _N += 1
    malon_md17_measures = Binner()
    malon_md17_measures.get_bond_pop(malon_md17.coords[::5], pairs)
    malon_ml_measures = Binner()
    malon_ml_measures.get_bond_pop(malon_ml.coords, pairs)
    malon_mlmm_measures = Binner()
    malon_mlmm_measures.get_bond_pop(malon_mlmm.coords, pairs)
    malon_qmmm_measures = Binner()
    malon_qmmm_measures.get_bond_pop(malon_qmmm.coords, pairs)


    info = [[malon_md17_measures.rs.T.flatten(),
            malon_ml_measures.rs.T.flatten(), 
            malon_qmmm_measures.rs.T.flatten(), 
            malon_mlmm_measures.rs.T.flatten(), 
            1000, '$r_{ij} / \AA$', 'all']]

    for i in info:
        bin_edges, hist = Binner.get_hist(i[0], i[4])
        bin_edges2, hist2 = Binner.get_hist(i[1], i[4])
        bin_edges3, hist3 = Binner.get_hist(i[2], i[4])
        bin_edges4, hist4 = Binner.get_hist(i[3], i[4])
        Plotter.xy_scatter(
                [bin_edges, bin_edges2, bin_edges3, bin_edges4], 
                [hist, hist2, hist3, hist4], 
                ['MD17', 'ML', 'QM/MM', 'ML/MM'], ['k', 'r', 'b', 'dodgerblue'], 
                i[5], 'P($r_{ij}$)', [10, 10, 10, 10],
                'hist-ml-mlmm-md17-r-{}.pdf'.format(i[6]))
        Plotter.xy_scatter(
                [bin_edges3, bin_edges4], 
                [ hist3, hist4], 
                ["QM/MM", "ML/MM"], ['r', 'dodgerblue'], i[5], 
                'P($r_{ij}$)', [10, 10],
                'hist-qmmm-mlmm-r-{}.pdf'.format(i[6]))
    sys.exit()
    """

    mols = input_files
    N = len(mols)
    molecules_md17 = []
    measures_md17 = []
    molecules_qmmm = []
    measures_qmmm = []
    molecules_ml = []
    measures_ml = []
    molecules_mlmm = []
    measures_mlmm = []

    idx = [i for i in range(N)] #[0, 4, 6, 7] #[3,4] # [3,4] #
    for i in range(len(mols)):
        if i in idx:
            print(i, mols[i], dihedrals[i], dihedrals[i+N], model_num[i])
            filepath = input_paths[0]
            md17path = os.path.join(filepath, f"{mols[i]}/revised_data")
            # print(os.listdir(md17path))
            # rMD17 500K gas phase dataset
            molecule_md17 = Molecule()
            atom_file = os.path.join(md17path, "nuclear_charges.txt")
            coord_files = [os.path.join(md17path, "coords.txt")]
            force_files = [os.path.join(md17path, "forces.txt")]
            energy_files = [os.path.join(md17path, "energies.txt")]
            NPParser(atom_file, coord_files, force_files, energy_files, molecule_md17)

            molecule_qmmm = Molecule()
            # test qmmm
            # qmmmpath = os.path.join(filepath, f"{mols[i]}/gaus_files2/cp2k_all.out")
            # # print(os.listdir(qmmmpath))
            # OPTParser([qmmmpath], molecule_qmmm, opt=False) #read in FCEZ for SP

            # Gaussian results for 500K QM/MM sims
            filepath_500K = input_paths[3]
            qmmmpath = os.path.join(filepath_500K, f"{mols[i]}")

            coord_files = [os.path.join(qmmmpath, "coords.txt")]
            force_files = [os.path.join(qmmmpath, "forces.txt")]
            energy_files = [os.path.join(qmmmpath, "energies.txt")]
            NPParser(atom_file, coord_files, force_files, energy_files, molecule_qmmm)

            # ML sim results 500K gas
            molecule_ml = Molecule()
            mlpath = os.path.join(filepath, f"{mols[i]}/load{model_num[i]}_gas")
            coord_files = [os.path.join(mlpath, "openmm-coords.txt")]
            force_files = [os.path.join(mlpath, "openmm-forces.txt")]
            energy_files = [os.path.join(mlpath, "openmm-delta_energies.txt")]
            NPParser(atom_file, coord_files, force_files, energy_files, molecule_ml)


            molecule_mlmm = Molecule()

            # Gaussian calcs for results for 300K Ml/MM sims
            filepath_300K = input_paths[2]
            mlmmpath = os.path.join(filepath_300K, f"{mols[i]}/300K_gaus_processed")
            coord_files = [os.path.join(qmmmpath, "coords.txt")]
            force_files = [os.path.join(qmmmpath, "forces.txt")]
            energy_files = [os.path.join(qmmmpath, "energies.txt")]

            # ML/MM sim results 500K aqueous
            # mlmmpath = os.path.join(filepath, f"{mols[i]}/load{model_num[i]}_soln")
            # if i == 7:
            #     pass
            # else:
            #     coord_files = [os.path.join(mlmmpath, "openmm-coords.txt")]
            #     force_files = [os.path.join(mlmmpath, "openmm-forces.txt")]
            #     energy_files = [os.path.join(mlmmpath, "openmm-delta_energies.txt")]
            NPParser(atom_file, coord_files, force_files, energy_files, molecule_mlmm)

            if len(molecule_md17.coords) >= 99_000:
                molecule_md17.coords = molecule_md17.coords[::5]
                molecule_md17.energies = molecule_md17.energies[::5]
                molecule_md17.forces = molecule_md17.forces[::5]
            pairs = []
            for j in range(len(molecule_md17.atoms)):
                for k in range(j):
                    pairs.append([j, k])
            _NC2 = len(pairs)
            measure_md17 = Binner()
            measure_md17.get_dih_pop(molecule_md17.coords, [dihedrals[i], dihedrals[i+N]])
            measure_md17.get_bond_pop(molecule_md17.coords, pairs)
            molecules_md17.append(molecule_md17)
            measures_md17.append(measure_md17)
            print("MD17", molecule_md17.coords.shape)
            #Preprocess.preprocessFE(molecule_md17, n_training, n_val, n_test, bias)

            if len(molecule_qmmm.coords) >= 99_000:
                molecule_qmmm.coords = molecule_qmmm.coords[::5]
                molecule_qmmm.energies = molecule_qmmm.energies[::5]
                molecule_qmmm.forces = molecule_qmmm.forces[::5]
            measure_qmmm = Binner()
            measure_qmmm.get_dih_pop(molecule_qmmm.coords, [dihedrals[i], dihedrals[i+N]])
            measure_qmmm.get_bond_pop(molecule_qmmm.coords, pairs)
            molecules_qmmm.append(molecule_qmmm)
            measures_qmmm.append(measure_qmmm)
            print("QMMM", molecule_qmmm.coords.shape)
            #Preprocess.preprocessFE(molecule_qmmm, n_training, n_val, n_test, bias)

            measure_ml = Binner()
            measure_ml.get_dih_pop(molecule_ml.coords, [dihedrals[i], dihedrals[i+N]])
            measure_ml.get_bond_pop(molecule_ml.coords, pairs)
            molecules_ml.append(molecule_ml)
            measures_ml.append(measure_ml)
            print("ML", molecule_ml.coords.shape)
            #Preprocess.preprocessFE(molecule_ml, n_training, n_val, n_test, bias)

            measure_mlmm = Binner()
            measure_mlmm.get_dih_pop(molecule_mlmm.coords, [dihedrals[i], dihedrals[i+N]])
            measure_mlmm.get_bond_pop(molecule_mlmm.coords, pairs)
            molecules_mlmm.append(molecule_mlmm)
            measures_mlmm.append(measure_mlmm)
            print("MLMM", molecule_mlmm.coords.shape)
            #Preprocess.preprocessFE(molecule_mlmm, n_training, n_val, n_test, bias)
    
    first_dihs = []
    first_rs = []
    first_qs = []
    second_dihs = []
    second_rs = []
    second_qs = []

    molecules1 = molecules_md17 # molecules_qmmm #
    measures1 = measures_md17 # measures_qmmm #
    molecules2 = molecules_mlmm #molecules_ml
    measures2 = measures_mlmm #measures_ml
    mols = [mols[i] for i in idx]
    for i in range(len(mols)):
        data = measures1[i]
        x = data.phis.T[0]
        dist = data.rs.flatten()
        first_dihs.append(x)
        first_rs.append(dist)
        #first_qs.append(molecules1[i].mat_FE[:,:_NC2,].flatten())

        data2 = measures2[i]
        x2 = data2.phis.T[0]
        dist2 = data2.rs.flatten()
        second_dihs.append(x2)
        second_rs.append(dist2)
        #second_qs.append(molecules2[i].mat_FE[:,:_NC2,].flatten())

        #print(measures1[i].phis.T[0], measures2[i].phis.T[0])
        # Plotter.hist_2d([measures1[i].phis.T[0], measures2[i].phis.T[0]], 
        #         [measures1[i].phis.T[1], measures2[i].phis.T[1]], 
        #         ['Greys', 'Reds'],
        #         '$\phi_1 / ^{\circ}$', '$\phi_2 / ^{\circ}$', 
        #         f'2dhist_dihs_{mols[i]}.pdf')
        # Plotter.hist_2d([measures1[i].phis.T[0]], 
        #         [measures1[i].phis.T[1]], 
        #         ['Greys', 'Reds'],
        #         '$\phi_1 / ^{\circ}$', '$\phi_2 / ^{\circ}$', 
        #         f'2dhist_dih_1_{mols[i]}.pdf')
        # Plotter.hist_2d([measures2[i].phis.T[0]], 
        #         [measures2[i].phis.T[1]], 
        #         ['Greys', 'Reds'],
        #         '$\phi_1 / ^{\circ}$', '$\phi_2 / ^{\circ}$', 
        #         f'2dhist_dih_2_{mols[i]}.pdf')

    colors = ["black", "dodgerblue"] #["grey", "dodgerblue"] #["k", "r"]
    Plotter.plot_violin(first_dihs, second_dihs, mols, colors, "P($\phi$)", "$\phi / ^{\circ}$", "dihs.pdf")
    Plotter.plot_violin(first_rs, second_rs, mols, colors, "P($r$)", "$r_{ij} / \AA$", "rs.pdf")
    #Plotter.plot_violin(first_qs, second_qs, mols, colors, "", "$q_{ij}$ / kcal/mol", "decompFE.pdf")


    """
    # plot the 2D hist for both salicylic conformers
    measure_sal_md17 = Binner()
    measure_sal_md17.get_dih_pop(molecules_md17[7].coords, [[9,0,1,2], [0,1,2,3]])
    measure_sal_qmmm = Binner()
    measure_sal_qmmm.get_dih_pop(molecules_qmmm[7].coords, [[9,0,1,2], [0,1,2,3]])
    print(measure_sal_md17.phis.shape)
    print(measure_sal_qmmm.phis.shape)
    Plotter.hist_2d([measure_sal_md17.phis.T[0], measure_sal_qmmm.phis.T[0]], 
                [measure_sal_md17.phis.T[1], measure_sal_qmmm.phis.T[1]], 
                ['Reds', 'Blues'],
                '$\phi_1 / ^{\circ}$', '$\phi_2 / ^{\circ}$', 
                'a-2dhist_dihs_salicylic.pdf')

    Plotter.hist_2d([measure_sal_qmmm.phis.T[0], measure_sal_md17.phis.T[0]], 
                [measure_sal_qmmm.phis.T[1], measure_sal_md17.phis.T[1]], 
                ['Blues', 'Reds'],
                '$\phi_1 / ^{\circ}$', '$\phi_2 / ^{\circ}$', 
                'b-2dhist_dihs_salicylic.pdf')
    

    Plotter.hist_1d([measure_sal_md17.phis.T[0], measure_sal_qmmm.phis.T[0]], 
                    '$\phi / ^{\circ}$', 'P($\phi$)', 'hist1d_salicylic.pdf',
                    color_list=["r", "tab:blue"])
    Plotter.plot_violin([measure_sal_md17.phis.T[0]], 
                        [measure_sal_qmmm.phis.T[0]], 
                        [""], "salicylic", "$\phi / ^{\circ}$", "violin1d_salicylic.pdf")
    """

    

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
        group = parser.add_argument('-p', '--input_paths', nargs='+', 
                metavar='file', default=[],
                help='list of paths.')
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
        group = parser.add_argument('-n_nodes', '--n_nodes', 
                action='store', default=1000, type=int, 
                help='number of nodes in neural network hidden layer/s')
        group = parser.add_argument('-n_layers', '--n_layers', 
                action='store', default=1, type=int, 
                help='number of dense layers in neural network')
        group = parser.add_argument('-n_training', '--n_training', 
                action='store', default=1000, type=int,
                help='number of data points for training neural network')
        group = parser.add_argument('-n_val', '--n_val', 
                action='store', default=50, type=int,
                help='number of data points for validating neural network')
        group = parser.add_argument('-n_test', '--n_test', 
                action='store', default=-1, type=int,
                help='number of data points for testing neural network')
        group = parser.add_argument('-grad_loss_w', '--grad_loss_w', 
                action='store', default=1000, type=int,
                help='loss weighting for gradients')
        group = parser.add_argument('-qFE_loss_w', '--qFE_loss_w', 
                action='store', default=1, type=int,
                help='loss weighting for pairwise decomposed forces '\
                        'and energies')
        group = parser.add_argument('-E_loss_w', '--E_loss_w', 
                action='store', default=1, type=int,
                help='loss weighting for energies')
        group = parser.add_argument('-bias', '--bias', 
                action='store', default='1/r', type=str,
                help='choose the bias used to describe decomposed/pairwise '\
                        'terms, options are - \n1:   No bias (bias=1)'\
                        '\n1/r:  1/r bias \n1/r2: 1/r^2 bias'\
                        '\nNRF:  bias using nuclear repulsive forces '\
                        '(zA zB/r^2) \nr:    r bias')
        group = parser.add_argument('-filtered', '--filtered', 
                action='store_true', 
                help='filter structures by removing high '\
                        'magnitude q structures')
        group = parser.add_argument('-load_model', '--load_model', 
                action='store', default=None,
                help='load an existing network to perform MD, provide '\
                        'the path+folder to model here')
        group = parser.add_argument('-d', '--dihedrals', nargs='+', 
                action='append', type=int, default=[],
                help='list of dihedrals')
        group = parser.add_argument('-mn', '--model_num', nargs='+', 
                #action='append', 
                type=int, default=[],
                help='list of model numbers for each molecule')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error('Command line arguments are ill-defined, '
        'please check the arguments.')
        raise
        sys.exit(1)

    run_force_pred(input_files=op.input_files, input_paths=op.input_paths, 
            atom_file=op.atom_file, 
            coord_files=op.coord_files, force_files=op.force_files, 
            energy_files=op.energy_files, charge_files=op.charge_files, 
            list_files=op.list_files, n_nodes=op.n_nodes, 
            n_layers=op.n_layers, n_training=op.n_training, n_val=op.n_val, 
            n_test=op.n_test, grad_loss_w=op.grad_loss_w, 
            qFE_loss_w=op.qFE_loss_w, E_loss_w=op.E_loss_w, bias=op.bias,
            filtered=op.filtered, load_model=op.load_model, 
            dihedrals=op.dihedrals, model_num=op.model_num)

if __name__ == '__main__':
    main()


