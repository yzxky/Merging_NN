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

## Import Data
data = scipy.io.loadmat('init_data\init_queue_fg.mat')
init_queue = data['init_queue']
# init_queue = init_queue[1:100, :]
data = scipy.io.loadmat('init_data\init_queue_fg_ocpar_025.mat')
config.oc_para = data['para']
for i in range(init_queue.shape[0]):
    init_queue[i][0] = i
# for i in range(init_queue.shape[0]):
#     if init_queue[i][1] == 2:
#         init_queue[i][3] += 4

queue = []
cav_log = []

L = 400
u_min = -4
beta = config.alpha * (u_min ** 2) / (2 * (1 - config.alpha))
UnconstrainedOptimal.para = [L, beta]

t = 0
config.t_step = 0.05
simulation_time = 60000

config.mode = 0  # 0: nn+cbf, 1: ocbf, 2: nn+ocbf

for i in range(simulation_time):
    t += config.t_step
    if len(init_queue) > 0 and t >= init_queue[0][2]:
        queue.append(CAV(init_queue[0]))
        init_queue = np.delete(init_queue, 0, 0)

    for idx, cav in enumerate(queue):
        if idx == 0:
            cav.update_u()
        else:
            cav.update_u(queue[idx - 1])

    for idx, cav in enumerate(queue):
        cav.update(config.t_step)

    while len(queue) > 0 and queue[0].x >= L + 100:
        cav_log.append(queue.pop(0))
        t_sum = 0
        energy_sum = 0
        for idx, cav in enumerate(cav_log):
            t_sum += cav.tf
            energy_sum += cav.energy
        print('t: %.4f, energy: %.4f' % (t_sum / len(cav_log), energy_sum / len(cav_log)))
        print('infeasible_cnt: %d' % config.infeasible_cnt)

t_sum = 0
energy_sum = 0
for idx, cav in enumerate(cav_log):
    t_sum += cav.tf
    energy_sum += cav.energy

print(t_sum / len(cav_log))
print(energy_sum / len(cav_log))
print('simulation end, alpha:%f' % config.alpha)

import pickle
if config.alpha == 0.01:
    if config.mode == 1:
        pickle.dump(cav_log, open('simu_001_len10_ocbf_fb.pickle', 'wb'))
    else:
        pickle.dump(cav_log, open('simu_001_len10_nn_ocbf_fb.pickle', 'wb'))
elif config.alpha == 0.1:
    if config.mode == 1:
        pickle.dump(cav_log, open('simu_01_len10_ocbf_fb.pickle', 'wb'))
    else:
        pickle.dump(cav_log, open('simu_01_len10_nn_ocbf_fb.pickle', 'wb'))
elif config.alpha == 0.25:
    if config.mode == 1:
        pickle.dump(cav_log, open('simu_025_len10_ocbf_fb.pickle', 'wb'))
    else:
        pickle.dump(cav_log, open('simu_025_len10_nn_ocbf_fb.pickle', 'wb'))
elif config.alpha == 0.4:
    pickle.dump(cav_log, open('simu_04_len10_nn_ocbf2_fb.pickle', 'wb'))
elif config.alpha == 0.6:
    if config.mode == 1:
        pickle.dump(cav_log, open('simu_06_len10_ocbf_fb.pickle', 'wb'))
    else:
        pickle.dump(cav_log, open('simu_06_len10_nn_ocbf_fb.pickle', 'wb'))