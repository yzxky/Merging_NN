import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import random
import matplotlib.pyplot as plt
import scipy.io
import time
import math
import numpy as np
from UnconstrainedOptimal import UnconstrainedOptimal
from config import config
from CAV import CAV
import pickle

pickle_file = open("simu_06_len10_nn_ocbf_fb.pickle", "rb")
objects = []
while True:
    try:
        objects.append(pickle.load(pickle_file))
    except EOFError:
        break
pickle_file.close()



print('ok')