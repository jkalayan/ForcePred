#!/bin/bash --login
#$ -cwd
#$ -S /bin/bash
#$ -N ANN #job name, keep it to 10 characters, can't start with a number
#$ -pe smp.pe 4   # Each task will use X cores
####$ -l short

# load the module used for python/tensorflow
module load apps/anaconda3/5.2.0
#module load apps/binapps/tensorflow/2.8.0-39-cpu

# current working directory
BASE=$(pwd)

# create variable for executable python script
EXE=$HOME/bin/ForcePred/runloadForcePred.py
chmod +x ${EXE} #make the script executable

echo "NSLOTS" $NSLOTS
export OMP_NUM_THREADS=$NSLOTS

# load conda environment for relevant python libraries
source activate updated_ml_env

##################################################################################
##################################################################################

# define variables here

# variables for input files
DIR="data" # optional directory path for input files
GAUS_FILES_PATH="" # if training with files from Gaussian calculations provide the path to all gaussian .out files here
ATOM_FILE_PATH="${DIR}/nuclear_charges.txt" # if not using Gaussian files, path to atom nuclear charge file
COORDS_FILE_PATH="${DIR}/coords200.txt" # path to coordinate files in Angstrom
FORCES_FILE_PATH="${DIR}/forces200.txt" # path to forces files in kcal/mol/Angstrom
ENERGY_FILE_PATH="${DIR}/energies200.txt" # pathe to energy files in kcal/mol

# variables for ANN architecture
BIAS="1/r" # bias used for decomposed energies, this is fixed for now, don't change!
N_TRAINING=100 # how many training points, make sure these are read first
N_VAL=50 # how many validation points from training, usually 5% of N_TRAINING
N_NODES=1000 # number of hidden nodes used in ANN
N_LAYERS=1 # number of hidden dense layers used in ANN
GRAD_LOSS_W=1000 # how much to weight the gradient in the loss function
QFE_LOSS_W=1 # how much to weight the decompFE in the loss function
E_LOSS_W=1 # how much to weight the energy in the loss function
EPOCHS=1_00 #0_000 # maximum number of epochs for training the ANN, default=1_000_000

# once ANN is trained, the following variables are for loading ANN and running MD
ANN_MODEL_PATH="${BASE}/model"
TEMP=500
NSTEPS=200
DT=10

##################################################################################
##################################################################################

# include command-line arguments used by the python script

# executable for training ANN with Gaussian files as input, uncomment below if needed
#EXE2="${EXE} -i ${GAUS_FILES_PATH} --bias ${BIAS} --n_training ${N_TRAINING} --n_val ${N_VAL} --n_nodes ${N_NODES} --n_layers ${N_LAYERS} --grad_loss_w ${GRAD_LOSS_W} --qFE_loss_w ${QFE_LOSS_W} --E_loss_w ${E_LOSS_W} --epochs ${EPOCHS} > training.log"

# executable for training ANN with array files as input 
EXE2="${EXE} -a ${ATOM_FILE_PATH} -c ${COORDS_FILE_PATH} -f ${FORCES_FILE_PATH} -e ${ENERGY_FILE_PATH} --bias ${BIAS} --n_training ${N_TRAINING} --n_val ${N_VAL} --n_nodes ${N_NODES} --n_layers ${N_LAYERS} --grad_loss_w ${GRAD_LOSS_W} --qFE_loss_w ${QFE_LOSS_W} --E_loss_w ${E_LOSS_W} --epochs ${EPOCHS} > training.log"

# executable for loading ANN and performing MD sim
EXE3="${EXE} --load_model ${ANN_MODEL_PATH} --temp ${TEMP} --nsteps ${NSTEPS} --dt ${DT} > loading.log"

# print out command above
echo ${EXE2}
echo ${EXE3}

# run script
$(eval ${EXE2} ) # run training
$(eval ${EXE3} ) # run MD sims with loaded ANN, comment out if not needed

# close conda environment
source deactivate
