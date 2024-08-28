import torch
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('main.log')])

logger = logging.getLogger(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()  #Cuda available
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

SEED_NUMBER = 2002