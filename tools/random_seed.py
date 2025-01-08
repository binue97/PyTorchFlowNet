import torch
import random
import numpy as np

SEED_VALUE = 42

def set_seed(seed=SEED_VALUE):
    # Set the random seed for all libraries.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
