import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

if config.alpha == 0.01:
    state_dict = torch.load('F:/xky/NNtrain_pycharm/saved_models/alpha_001_len10/RNN_alpha001_epoch_491.pt')
elif config.alpha == 0.1:
    state_dict = torch.load('F:/xky/NNtrain_pycharm/saved_models/alpha_01_len10/RNN_alpha01_epoch_491.pt')
elif config.alpha == 0.25:
    state_dict = torch.load('F:/xky/NNtrain_pycharm/saved_models/alpha_025_len10/RNN_alpha025_epoch_491.pt')
elif config.alpha == 0.4:
    state_dict = torch.load('F:/xky/NNtrain_pycharm/saved_models/alpha_04_len10/RNN_alpha04_epoch_491.pt')
elif config.alpha == 0.6:
    state_dict = torch.load('F:/xky/NNtrain_pycharm/saved_models/alpha_06_len10/RNN_alpha06_epoch_491.pt')

if config.alpha == 0.01:
    x_mean = [207.4746553525554, 16.31337511671317, 0, 246.79477447531002, 15.68320258580891, 0]
    x_std  = [115.9349791973303, 2.731666959593572, 1, 121.11882860588405, 3.487326379276549, 1]
elif config.alpha == 0.1:
    x_mean = [190.38560436083668, 21.447527876892654, 0, 257.8563903430447, 20.230362839237575, 0]
    x_std  = [117.72334932178626, 2.180779934453148,  1, 147.66537797756982, 7.216658099384231, 1]
elif config.alpha == 0.25:
    x_mean = [188.00121574629037, 23.663921676510885, 0, 236.21750235176745, 23.93718141894336, 0]
    x_std  = [118.36699990805228, 3.4318960368832783, 1, 131.62106494798954, 5.350234352504258, 1]
elif config.alpha == 0.6:
    x_mean = [180.49882844332456, 28.681441399878462, 0, 239.81383557413233, 29.257882520148684, 0]
    x_std  = [119.70432487088446,  5.716734511153706, 0, 141.62596064702674, 8.697516369629076, 1]











