import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import random
import matplotlib.pyplot as plt
import numpy as np
from NNmodel import MergingGRU, MergingMLP, MergingDecoder, device
from CBF import CBF
from UnconstrainedOptimal import UnconstrainedOptimal
from config import config
from CAV_IO import x_mean, x_std, state_dict

class CAV:
    def __init__(self, init):
        self.id = init[0]
        self.lane = init[1]
        self.t0 = init[2]
        self.v0 = init[3]
        self.t = self.t0
        self.tf = self.t0
        self.x = 0
        self.v = init[3]
        self.u = 0
        self.u_ = 0
        self.energy = 0
        self.x_log = [self.x]
        self.v_log = [self.v]
        self.u_log = [self.u]
        self.t_log = [self.t]
        self.mode = 0
        ## OCBF init
        if config.mode == 0:
            self.para = self.calculate_unconstrained_opt()
        elif config.mode == 1 or config.mode == 2:
            self.para = config.oc_para[int(self.id)]
        ## NN init
        self.input_size = 6
        self.hidden_size = 256
        self.rnn = MergingGRU(self.input_size, self.hidden_size).to(device)
        self.rnn.load_state_dict(state_dict['rnn_state_dict'])
        # self.mlp = MergingMLP(self.hidden_size, 1).to(device)
        # self.mlp.load_state_dict(state_dict['mlp_state_dict'])
        self.mlp = MergingDecoder(self.hidden_size, 1).to(device)
        self.mlp.load_state_dict(state_dict['mlp_state_dict'])
        ## NN input
        self.nn_input = np.empty((0, 0))
        self.rnn_hidden = torch.zeros(1, 1, self.hidden_size, device=device)

    def update_u(self, cav=0):
        if self.x <= 400:
            if config.mode == 0:
                ref_u = self.calculate_ref_traj(cav)
                if abs(ref_u) > 4:
                    self.u = 4 * ref_u / abs(ref_u)
                else:
                    self.u = ref_u
                #self.u = self.cbf_nn(ref_u, cav)
            elif config.mode == 1:
                x_ref, v_ref, u_ref = self.calculate_oc_traj()
                self.u = self.ocbf([u_ref, v_ref], cav)
            elif config.mode == 2:
                ref_u = self.calculate_ref_traj(cav)
                if cav != 0 and self.t - self.t0 > 10 * config.t_step: #cav.x - self.x <= 1.8 * self.v + 10:
                    self.mode = 1
                if self.mode == 0:
                    x_ref, v_ref, u_ref = self.calculate_oc_traj()
                    self.u = self.ocbf([u_ref, v_ref], cav)
                else:
                    self.u = self.cbf_nn(ref_u, cav)
        else:
            self.u = 0

    def update(self, t_step):
        self.x += self.v * t_step + 0.5 * self.u * (t_step ** 2)
        self.v += self.u * t_step
        self.u_ = self.u
        if self.x <= 400:
            self.energy += 0.5 * (self.u ** 2) * t_step
            self.x_log.append(self.x)
            self.v_log.append(self.v)
            self.u_log.append(self.u)
            self.t_log.append(self.t)
            self.t += t_step
            self.tf = self.t - self.t0

    def calculate_ref_traj(self, cav=0):
        self.update_nn_input(cav)
        tensor_tmp = torch.from_numpy(self.nn_input).float()
        self.rnn_hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        rnn_output, rnn_hidden = self.rnn(tensor_tmp.view(1, -1, 6), self.rnn_hidden)
        mlp_output = self.mlp(rnn_output, rnn_hidden)
        return mlp_output[:, -1].item()

    def calculate_oc_traj(self):
        x_ref = 1/6 * self.para[0] * (self.t ** 3) + 1/2 * self.para[1] * (self.t ** 2) + self.para[2] * self.t + self.para[3]
        v_ref = 1/2 * self.para[0] * (self.t ** 2) + self.para[1] * self.t + self.para[2]
        u_ref = self.para[0] * self.t + self.para[1]
        return x_ref, v_ref, u_ref

    def calculate_unconstrained_opt(self):
        return UnconstrainedOptimal.calculate_unconstrained_opt(self.t0, self.v0, UnconstrainedOptimal.para[0], UnconstrainedOptimal.para[1])

    def cbf_nn(self, ref_u, cav=0):
        if cav == 0:
            return CBF.cbf_nn([self.x, self.v], ref_u, 0, 0)
        elif self.lane == cav.lane:
            return CBF.cbf_nn([self.x, self.v], ref_u, 1, [cav.x, cav.v, cav.u_])
        elif self.lane != cav.lane:
            return CBF.cbf_nn([self.x, self.v], ref_u, 2, [cav.x, cav.v, cav.u_])

    def ocbf(self, ref_traj, cav=0):
        if cav == 0:
            return CBF.cbf_oc([self.x, self.v], ref_traj, 0, 0)
        elif self.lane == cav.lane:
            return CBF.cbf_oc([self.x, self.v], ref_traj, 1, [cav.x, cav.v, cav.u_])
        elif self.lane != cav.lane:
            return CBF.cbf_oc([self.x, self.v], ref_traj, 2, [cav.x, cav.v, cav.u_])

    ## NN related class method
    def update_nn_input(self, cav=0):
        curr_input = self.generate_single_input(cav)
        if self.nn_input.shape[0] == 0:
            self.nn_input = curr_input
        elif self.nn_input.shape[0] < 10:
            self.nn_input = np.append(self.nn_input, curr_input, axis=0)
        else:
            self.nn_input = np.append(np.delete(self.nn_input, 0, axis=0),
                                      curr_input, axis=0)

    def generate_single_input(self, cav=0):
        if cav == 0:
            return self.normalize_input(np.array([[self.x, self.v, self.lane, 0, 0, 0]]))
        else:
            return self.normalize_input(np.array([[self.x, self.v, self.lane, cav.x, cav.v, cav.lane]]))

    def normalize_input(self, input):
        for i in [0, 1, 3, 4]:
            input[:, i] = (input[:, i] - x_mean[i]) / x_std[i]
        return input
