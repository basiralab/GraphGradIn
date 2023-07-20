import os
import argparse
from model import DGN
from config import MODEL_PARAMS, CONFIG
if not os.path.exists('temp'):
    os.makedirs('temp')
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('saved_weights'):
    os.makedirs('saved_weights')
if not os.path.exists('./output/cbts'):
    os.makedirs('./output/cbts')
import numpy as np
import random
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--GraphGradIn", help="Use GraphGradIn influence score calculation", action="store_true")
parser.add_argument("--GraphTestIn", help="Use GraphTestIn influence score calculation", action="store_true")
args = parser.parse_args()

simulated_dataset = CONFIG["X"]

seed = 35813
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DGN.train_model(
                X=simulated_dataset,
                model_params=MODEL_PARAMS,
                n_max_epochs=CONFIG["N_max_epochs"],
                random_sample_size=CONFIG["random_sample_size"],
                early_stop=CONFIG["early_stop"],
                args=args
)
