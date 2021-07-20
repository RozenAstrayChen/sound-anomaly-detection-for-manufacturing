import argparse, os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

def autoencoder(input_dims, type=0):
    """
    Define a torch model for anoamly detection

    PARAMS
    ===
        input_dims
    
    RETURN
    ===
        Model return
    """
    if type == 0:
        # origin autoencoder
        model = nn.Sequential(
            # Encoder
            torch.nn.Linear(input_dims, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 8), torch.nn.ReLU(),

            # Decoder
            torch.nn.Linear(8, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, input_dims)
        )

    elif type == 1:
        # smooth autoencoder
        model = nn.Sequential(
            # Encoder
            torch.nn.Linear(input_dims, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),

            # Decoder
            torch.nn.Linear(32, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, input_dims)
        )
        
    elif type == 2:
        # CNN
        pass

    return model

