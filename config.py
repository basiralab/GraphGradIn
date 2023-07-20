#Train on simulated data (S, normal random dist.) or external data (E)
Dataset = "S"

#Path for the dataset (binary file in NumPy .npy format) with shape
#[N_Subjects, N_Nodes, N_Nodes, N_Views]
#Path = "./simulated dataset/example.npy"
Path = ""

#Number of simulated subjects (overwriten if Dataset = "E") 
N_Subjects = 20

#Number of nodes for simulated brain networks (overwriten if Dataset = "E") 
N_Nodes = 35

#Number of brain views (overwriten if Dataset = "E")
N_views = 6

#Number of training epochs
N_max_epochs = 800

#List of epochs to take gradients from, in our case we take the final epoch
saved_epoch_score_list = [340]

#Apply early stopping True/False
early_stop = True

#Random subset size for SNL function  
random_sample_size = 10

#Number of cross validation folds
n_folds = 5

#Learning Rate for Adam optimizer
lr = 0.001

#dimension of embeddings output by the first GDL layer (for each ROI)
CONV1 = 32
#dimension of embeddings output by the second GDL layer (for each ROI)
CONV2 = 64
#dimension of embeddings output by the third GDL layer (for each ROI)
CONV3 = 64

#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#                 Below is not to be modified manually                       #
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#

import numpy as np
import helper 

if Dataset.lower() not in ["e", "E", "s", "S"]:
    raise ValueError("Dataset options are E or S.")
    
if (Dataset.lower() == "e"):
    X = np.load(Path)
    N_Subjects = X.shape[0]
    N_Nodes = X.shape[1]
    N_views = X.shape[3]
else:
    X = helper.create_better_simulated(N_Subjects, N_Nodes)

CONFIG = {
        "X": X,
        "N_ROIs":  X.shape[1],
        "N_views":  X.shape[3],
        "N_max_epochs": N_max_epochs,
        "saved_epoch_score_list": saved_epoch_score_list,
        "n_folds": n_folds,
        "random_sample_size": random_sample_size,
        "early_stop": early_stop
    }

MODEL_PARAMS = {
        "N_ROIs": N_Nodes,
        "learning_rate" : lr,
        "n_attr": X.shape[3],
        "Linear1" : {"in": N_views, "out": CONV1},
        "conv1": {"in" : 1, "out": CONV1},
        
        "Linear2" : {"in": N_views, "out": CONV1*CONV2},
        "conv2": {"in" : CONV1, "out": CONV2},
        
        "Linear3" : {"in": N_views, "out": CONV2*CONV3},
        "conv3": {"in" : CONV2, "out": CONV3},
        "num_classes": 2
    }