# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup GPU device if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup arameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15
IMG_SIZE = 32
N_CLASSES = 10
