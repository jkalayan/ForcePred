
!!!! PLEASE NOTE, THIS FILE IS IN PROGRESS !!!! 
ANYTHING BELOW THE LINE MIGHT NOT BE RELEVANT


1.  Before running a job, install the conda environment by running 
    ./create_conda_env.sh on the command line in this directory.
    This will install a conda environment in your $HOME directory called 
    updated_ml_env from the updated_ml_env.yml file which contains all the 
    Python libraries needed to run ForcePred. The installation could take 
    about 10 minutes.






_______________________________________________________________________________

RUNNING AN ANN WITH YOUR GENERATED DATA
 
I’ve written a script for you to run an ANN with your generated data. You have two sets of data, one is the training data, and from this we pick a few structures (around 5% of the training structures) to validate the ANN during training. Your test data (which you’ve labelled as validation) is the data that is not used to train the network and is instead used to test how well the trained ANN performs with unseen data.
 
The ForcePred folder contains the python program used to train an ANN. Please put this in a folder called bin in your home directory on the CSF.
 
The submit_NN.sh file is the job script used to submit the job to the CSF. There are a few variables in this job script that you can change.
 
FILES_PATH="displaced_*/glycol*/glycol*.out"
This is the path to all your .out files. This assumes that your .out files containing training data will be read before your .out files containing test data. So make sure the folder name containing your training data comes alphanumerically before the folder name containing you test data!
N_TRAINING=291
How many training structures you have.
N_VAL=15
How many validation structures from the training data, usually 5% of N_TRAINING
N_NODES=1000
The number of hidden nodes used in ANN, keep this at 1000 for now
N_LAYERS=1
The number of hidden dense layers used in ANN, keep this at 1 for now
GRAD_LOSS_W=1000
How much to weight the gradient loss function, the loss function is how we evaluate the error in our ML predictions. Keep this to 1000 for now
QFE_LOSS_W=1
How much to weight the decomposed forces and energies in the loss function, keep this at 1 for now
E_LOSS_W=1
How much to weight the energy in the loss function, keep this at 1 for now
 
You can then submit the job script with;
 
qsub submit_NN.sh
 
If you successfully run the job, then there will be quite few output files.
idx_*.dat – These files contain the indices of the structures used for training, validation and testing.
molecule_*.dat – These files contain the coordinates, forces, energies and nuclear charges of all the structures. Neil might want to use these files.
testset_hist_*.dat – These files contain the histogram data for s-curves, which tells us what percentage of structures (second column) have errors below a given value (first column). You could try to plot these using matplotlib if you like, just make sure your error axis is a log scale.
all_energies.csv – These files contain energies of all the structures (first column) and the predicted energies from the ANN (second column)
file.log – this is the log file from the script and contains information about reading in structures, scaling energies, training the network and the end of the file contains the test errors for gradients, pairwise decomposed forces and energies and the molecule energy. The MAE is probably what you’ll be using to assess how accurately you trained ANN is predicted molecular properties.
model – in this folder, you’ll find the outputted model weights of the best_ever_model generated from training. These files are not human readable but will be used if we decide to run some MD simulations with your ANN models.
 
 
 

