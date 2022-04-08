import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
from mosek import iparam
from config import config

class CBF:

    u_min = -4
    u_max = 4
    v_min = 5
    v_max = 30
    phi = 1.8
    delta = 0
    L = 400
    # OCBF para
    psi = 1
    eps = 10

    @staticmethod
    def cbf_nn(state, ref_traj, mode = 0, state_p = 0):
        x = state[0]
        v = state[1]
        u_ref = ref_traj

        P = matrix(np.array([1]), tc='d')
        q = matrix(np.array([-u_ref]), tc='d')
        # u_max, u_min
        G1 = np.array([1, -1])
        h1 = np.array([CBF.u_max, -CBF.u_min])
        # v_max, v_min
        G2 = np.array([1, -1])
        h2 = np.array([CBF.v_max - v, v - CBF.v_min])
        if mode == 0:
            G = matrix(np.concatenate([G1, G2], axis=0), tc='d')
            h = matrix(np.concatenate([h1, h2], axis=0), tc='d')
        elif mode == 1:
            x_ip, v_ip, u_ip = state_p[0], state_p[1], state_p[2]
            G3 = np.array([CBF.phi])
            h3 = np.array([v_ip - v + x_ip - x - CBF.phi * v - CBF.delta])
            if config.fg:
                G5 = np.array([1])
                h5 = np.array([u_ip + v_ip - v - CBF.phi * CBF.u_min])
                G = matrix(np.concatenate([G1, G2, G3, G5], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3, h5], axis=0), tc='d')
            else:
                G = matrix(np.concatenate([G1, G2, G3], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3], axis=0), tc='d')
        elif mode == 2:
            x_ip, v_ip, u_ip = state_p[0], state_p[1], state_p[2]
            G3 = np.array([CBF.phi / CBF.L * x])
            h3 = np.array([v_ip - v - CBF.phi / CBF.L * (v ** 2) +
                           x_ip - x - CBF.phi / CBF.L * x * v - CBF.delta])
            if config.fg:
                G5 = np.array([1 + 2 * CBF.phi / CBF.L * v])
                h5 = np.array([u_ip - CBF.phi / CBF.L * v * CBF.u_min + v_ip - v - CBF.phi / CBF.L * (v ** 2) -
                               CBF.phi / CBF.L * x * CBF.u_min])
                G = matrix(np.concatenate([G1, G2, G3, G5], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3, h5], axis=0), tc='d')
            else:
                G = matrix(np.concatenate([G1, G2, G3], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3], axis=0), tc='d')
        else:
            print('error')

        solvers.options['show_progress'] = False
        solvers.options['mosek'] = {iparam.log: 0, iparam.max_num_warnings: 0}
        sol = solvers.qp(P, q, G, h, solver='mosek')
        if sol['status'] == 'optimal':
            u = np.array(sol['x'])[0][0]
        else:
            config.infeasible_cnt += 1
            u = -4
        if abs(u) > 4:
            print(u)
        return u

    @staticmethod
    def cbf_oc(state, ref_traj, mode = 0, state_p = 0): # ref_traj: [u, v]
        x = state[0]
        v = state[1]
        u_ref = ref_traj[0]
        v_ref = ref_traj[1]

        P = matrix(np.array([[1, 0], [0, CBF.psi]]), tc='d')
        q = matrix(np.array([-u_ref, 0]), tc='d')
        # u_max, u_min
        G1 = np.array([[1, 0], [-1, 0]])
        h1 = np.array([CBF.u_max, -CBF.u_min])
        # v_max, v_min
        G2 = np.array([[1, 0], [-1, 0]])
        h2 = np.array([CBF.v_max - v, v - CBF.v_min])
        # CLF
        G3 = np.array([[2 * (v - v_ref), -1]])
        h3 = np.array([-CBF.eps * ((v - v_ref) ** 2)])
        if mode == 0:
            G = matrix(np.concatenate([G1, G2, G3], axis=0), tc='d')
            h = matrix(np.concatenate([h1, h2, h3], axis=0), tc='d')
        elif mode == 1:
            x_ip, v_ip, u_ip = state_p[0], state_p[1], state_p[2]
            G4 = np.array([[CBF.phi, 0]])
            h4 = np.array([v_ip - v + x_ip - x - CBF.phi * v - CBF.delta])
            if config.fg:
                G5 = np.array([[1, 0]])
                h5 = np.array([u_ip + v_ip - v - CBF.phi * CBF.u_min])
                G = matrix(np.concatenate([G1, G2, G3, G4, G5], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3, h4, h5], axis=0), tc='d')
            else:
                G = matrix(np.concatenate([G1, G2, G3, G4], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3, h4], axis=0), tc='d')
        elif mode == 2:
            x_ip, v_ip, u_ip = state_p[0], state_p[1], state_p[2]
            G4 = np.array([[CBF.phi / CBF.L * x, 0]])
            h4 = np.array([v_ip - v - CBF.phi / CBF.L * (v ** 2) +
                           x_ip - x - CBF.phi / CBF.L * x * v - CBF.delta])
            if config.fg:
                G5 = np.array([[1 + 2 * CBF.phi / CBF.L * v, 0]])
                h5 = np.array([u_ip - CBF.phi / CBF.L * v * CBF.u_min + v_ip - v - CBF.phi / CBF.L * (v ** 2) -
                               CBF.phi / CBF.L * x * CBF.u_min])
                G = matrix(np.concatenate([G1, G2, G3, G4, G5], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3, h4, h5], axis=0), tc='d')
            else:
                G = matrix(np.concatenate([G1, G2, G3, G4], axis=0), tc='d')
                h = matrix(np.concatenate([h1, h2, h3, h4], axis=0), tc='d')
        else:
            print('error')

        solvers.options['show_progress'] = False
        solvers.options['mosek'] = {iparam.log: 0, iparam.max_num_warnings: 0}
        sol = solvers.qp(P, q, G, h, solver='mosek')
        if sol['status'] == 'optimal':
            u = np.array(sol['x'])[0][0]
        else:
            config.infeasible_cnt += 1
            u = -4
        if abs(u) > 4:
            print(u)

        return u