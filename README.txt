
!!!! PLEASE NOTE, THIS FILE IS IN PROGRESS !!!! 
ANYTHING BELOW THE LINE MIGHT NOT BE RELEVANT


1.  Before running a job, install the conda environment by running 
    ./create_conda_env.sh on the command line in this directory.
    This will install a conda environment in your $HOME directory called 
    updated_ml_env from the updated_ml_env.yml file which contains all the 
    Python libraries needed to run ForcePred. The installation could take 
    about 10 minutes.

2.  To train an artificial neural network (ANN), you will need the following
    training data+dimensions+units for a given molecule:
    - nuclear charges (1,N_atoms) Z
    - Cartesian coordinates (3,N_atoms*N_structures) Angstrom
    - Cartesian forces (3,N_atoms*N_structures) kcal/mol/Angstrom
    - Molecule energies (1,N_structures) kcal/mol

    ASIDE: The program does have a class to read in Gaussian files directly 
    and convert to the units above if needed.

3.  To train/test an ANN, the following python script can be used:
    > runloadForcePred.py
    This will output a folder called model, which contains all the information
    needed to reload the model weights and scaling factors.
    The corresponding csf jobscript for this python script is:
    > submit_runloadForcePred.sh

4.  To load a previously trained ANN and perform MD simulations with predicted
    forces, the following python script can be used:
    > runloadForcePred.py
    By including the --load_model flag, and removing flags for input files, 
    This will recreate the ANN archtecture and load weights from a trained
    model.
    The corresponding csf jobscript for this python script is:
    > submit_runloadForcePred.py

5.  Use the following command in your loaded conda environment to print out 
    the flag options for a python script:
    > <python_script_name> -h

    usage: runloadForcePred.py [-h]

    Program for reading in molecule forces, coordinates and energies for force
    prediction.

    optional arguments:
      -h, --help            show this help message and exit
      -i file [file ...], --input_files file [file ...]
                            name of file/s containing forces coordinates and
                            energies. (default: [])
      -a file, --atom_file file
                            name of file/s containing atom nuclear charges.
                            (default: None)
      -c file [file ...], --coord_files file [file ...]
                            name of file/s containing coordinates. (default: [])
      -f file [file ...], --force_files file [file ...]
                            name of file/s containing forces. (default: [])
      -e file [file ...], --energy_files file [file ...]
                            name of file/s containing energies. (default: [])
      -q file [file ...], --charge_files file [file ...]
                            name of file/s containing charges. (default: [])
      -l file, --list_files file
                            file containing list of file paths. (default: False)
      -n_nodes N_NODES, --n_nodes N_NODES
                            number of nodes in neural network hidden layer/s
                            (default: 1000)
      -n_layers N_LAYERS, --n_layers N_LAYERS
                            number of dense layers in neural network (default: 1)
      -n_training N_TRAINING, --n_training N_TRAINING
                            number of data points for training neural network
                            (default: 1000)
      -n_val N_VAL, --n_val N_VAL
                            number of data points for validating neural network
                            (default: 50)
      -n_test N_TEST, --n_test N_TEST
                            number of data points for testing neural network
                            (default: -1)
      -grad_loss_w GRAD_LOSS_W, --grad_loss_w GRAD_LOSS_W
                            loss weighting for gradients (default: 1000)
      -qFE_loss_w QFE_LOSS_W, --qFE_loss_w QFE_LOSS_W
                            loss weighting for pairwise decomposed forces and
                            energies (default: 1)
      -E_loss_w E_LOSS_W, --E_loss_w E_LOSS_W
                            loss weighting for energies (default: 1)
      -epochs EPOCHS, --epochs EPOCHS
                            number of epochs for training (default: 1000000)
      -bias BIAS, --bias BIAS
                            choose the bias used to describe decomposed/pairwise
                            terms, options are - 1: No bias (bias=1) 1/r: 1/r bias
                            1/r2: 1/r^2 bias NRF: bias using nuclear repulsive
                            forces (zA zB/r^2) r: r bias (default: 1/r)
      -filtered, --filtered
                            filter structures by removing high magnitude q
                            structures (default: False)
      -load_model LOAD_MODEL, --load_model LOAD_MODEL
                            load an existing network to perform MD, provide the
                            path+folder to model here (default: None)
      -temp TEMP, --temp TEMP
                            set temperature (K) of MD simulation (default: 500)
      -nsteps NSTEPS, --nsteps NSTEPS
                            set number of steps to take for MD simulation, each
                            time step is 0.5 fs (default: 200)
      -dt DT, --dt DT       interval between number of steps saved to file
                            (default: 10)

6.	Output files:
	For training -  
		the training.log file contains info about how energies have been
	scaled and then the loss values for each epoch during training. At the end of
	the file, the errors for energy, force and decompFE are printed. 
		In the plots/ folder, there are several plots of s-curves, loss curve, 
	histogram of qs for the inputted structures and a scatter plot of r vs decompFE.
		In the model/ folder, the best_ever_model* files contain the ANN model
	weights, etc., needed for tensorflow to reload the model. 
	> molecule.pdb, atoms.txt - initial starting structure and nuclear charges 
		used to perform MD simulations later with OpenMM.
	> prescale.txt - scaling information needed to scale molecule energies in
		ANN model predictions, list of six values for training structures are: 
		[min_E/kcal/mol, max_E/kcal/mol, 
		min(abs(F/kcal/mol/Ang)), max(abs(F/kcal/mol/Ang)), 
		max(abs(NRF/kcal/mol/Ang)), max(abs(q/kcal/mol/Ang))]

	For loading - 
		the loading.log file contains info about what ANN model is loaded
	and general prints of simulation conditions. In the simulation/ folder, there
	are files for the following;
	> openmm.csv - log of the step, PE, KE and temp in each simulation frame saved
	> openmm-coords.txt/.xyz / o-trajectory.dcd - coordinates of simulated system 
		in three different formats.
	> openmm-forces.txt - predicted forces for each saved frame
	> openmm-delta-energies.txt - change in energy of molecule compared to first
		frame of simulation. 
	

