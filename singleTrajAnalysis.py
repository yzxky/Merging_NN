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
data = scipy.io.loadmat('init_data\\init_queue_train36.mat')
init_queue = data['init_queue']
data = scipy.io.loadmat('init_data\\init_queue_train36_ocpar_025.mat')
config.oc_para = data['para']
data = scipy.io.loadmat('init_data\\train_alpha025_raw_36.mat')
oc = data['data_log']
oc_traj = {}
for i in range(init_queue.shape[0]):
    init_queue[i][0] = i
for i in range(int(oc[:, 5].max() - 50)):
    oc_traj[i] = oc[oc[:, 5] == i + 51, :]

queue = []
cav_log = []

L = 400
u_min = -4
beta = config.alpha * (u_min ** 2) / (2 * (1 - config.alpha))
UnconstrainedOptimal.para = [L, beta]

t = 0
config.t_step = 0.1
simulation_time = 6000

config.mode = 0  # 0: nn+cbf, 1: ocbf, 2: nn+ocbf

pre_id = 17
t = init_queue[pre_id][2]
queue.append(CAV(init_queue[pre_id]))
flag = True
pre_t = 0
pre_t2 = 0

for i in range(simulation_time):
    if flag and t >= init_queue[pre_id + 1][2]:
        queue.append(CAV(init_queue[pre_id + 1]))
        flag = False

    for idx, cav in enumerate(queue):
        if idx == 0:
            if cav.id == pre_id:
                if pre_t < oc_traj[pre_id].shape[0]:
                    cav.u = oc_traj[pre_id][pre_t, 11]
                else:
                    cav.u = 0
                pre_t += 1
            else:
                if config.mode == 3:
                    if pre_t2 < oc_traj[pre_id + 1].shape[0]:
                        cav.u = oc_traj[pre_id + 1][pre_t2, 11]
                    else:
                        cav.u = 0
                    pre_t2 += 1
                else:
                    cav.update_u()
        else:
            if config.mode == 3:
                if pre_t2 < oc_traj[pre_id + 1].shape[0]:
                    cav.u = oc_traj[pre_id + 1][pre_t2, 11]
                else:
                    cav.u = 0
                pre_t2 += 1
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

    t += config.t_step

# plt.plot(cav_log[1].t_log[1:], cav_log[1].u_log[1:])
# plt.show()

if config.mode == 0:
    np.savetxt('nn_data2.csv', np.array([cav_log[1].t_log, cav_log[1].u_log, cav_log[1].v_log, cav_log[1].x_log]))
elif config.mode == 1:
    np.savetxt('ocbf_data.csv', np.array([cav_log[1].t_log, cav_log[1].u_log, cav_log[1].v_log, cav_log[1].x_log]))
elif config.mode == 2:
    np.savetxt('nn_ocbf_data.csv', np.array([cav_log[1].t_log, cav_log[1].u_log, cav_log[1].v_log, cav_log[1].x_log]))
elif config.mode == 3:
    np.savetxt('oc_data.csv', np.array([cav_log[1].t_log, cav_log[1].u_log, cav_log[1].v_log, cav_log[1].x_log]))

print('t: %.4f, energy: %.4f, obj: %.4f' % (cav_log[1].tf, cav_log[1].energy, beta * cav_log[1].tf + cav_log[1].energy))

print('simulation successfully ended')
