from typing import Tuple

import cmath
import random
from scipy.linalg import expm

from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
from qutip.qip.operations import cnot, cphase

from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from relaqs.api.utils import normalize

import gymnasium as gym
import numpy as np
import scipy.linalg as la

sig_p = np.array([[0, 1.], [0, 0]])
sig_m = np.array([[0, 0], [1., 0]])
X = np.array([[0, 1.], [1., 0]])
Z = np.array([[1., 0], [0, -1.]])
I = np.array([[1., 0], [0, 1.]])
Y = np.array([[0, -1.j], [1.j, 0]])

H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
S = np.array([[1., 0], [0, 1.j]])
Sdagger = np.array([[1., 0], [0, -1.j]])

# two-qubit single qubit gates
II = tensor(Qobj(I), Qobj(I)).data.toarray()
X1 = tensor(Qobj(X), Qobj(I)).data.toarray()
X2 = tensor(Qobj(I), Qobj(X)).data.toarray()
Y1 = tensor(Qobj(Y), Qobj(I)).data.toarray()
Y2 = tensor(Qobj(I), Qobj(Y)).data.toarray()
Z1 = tensor(Qobj(Z), Qobj(I)).data.toarray()
Z2 = tensor(Qobj(I), Qobj(Z)).data.toarray()

sig_p1 = tensor(Qobj(sig_p), Qobj(I)).data.toarray()
sig_p2 = tensor(Qobj(I), Qobj(sig_p)).data.toarray()
sig_m1 = tensor(Qobj(sig_m), Qobj(I)).data.toarray()
sig_m2 = tensor(Qobj(I), Qobj(sig_m)).data.toarray()
sigmap1 = Qobj(sig_p1)
sigmap2 = Qobj(sig_p2)
sigmam1 = Qobj(sig_m1)
sigmam2 = Qobj(sig_m2)

# two-qubit gate basis
XX = tensor(Qobj(X), Qobj(X)).data.toarray()
YY = tensor(Qobj(Y), Qobj(Y)).data.toarray()
ZZ = tensor(Qobj(Z), Qobj(Z)).data.toarray()
exchangeOperator1 = XX + YY
exchangeOperator2 = YY + ZZ
exchangeOperator3 = XX + ZZ

CNOT = cnot().data.toarray()
CZ = cphase(np.pi).data.toarray()

# C_MATRIX = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1]])
# C_MATRIX_INV = np.linalg.inv(C_MATRIX)
# HSH = H @ S @ H
# SdaggerH = Sdagger @ H
#
# # magic basis
# B = 1 / np.sqrt(2) * np.array([[1., 0, 0, 1.j], [0, 1.j, 1., 0],
#                                [0, 1.j, -1., 0], [1., 0, 0, -1.j]])  # Magic Basis
#
# # magic basis dagger
# B_dagger = np.conj(B).T


class ScratchNoisyTwoQubitEnv(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "action_space_size": 27,
            "U_initial": II,  # staring with I
            "U_target": CZ,  # target for CZ
            "final_time": 30E-9,
            # in seconds, total time is final_time * 5 because of single qubit + two_qubit + single_qubit + two_qubit + single_qubit
            "num_Haar_basis": 3,  # number of Haar basis (need to update for odd combinations)
            "steps_per_Haar": 1,  # steps per Haar basis per episode
            "detuning_list": np.random.normal(0, np.pi / 100 / 30E-9, size=(2, 100)).tolist(),  # qubit detuning
            "save_data_every_step": 1,
            "verbose": True,
            # "relaxation_rates_list": [[1/60E-6/2/np.pi],[1/30E-6/2/np.pi],[1/66E-6/2/np.pi],[1/5E-6/2/np.pi]], # relaxation lists of list of floats to be sampled from when resetting environment.
            "relaxation_rates_list": [[0], [0], [0], [0]],  # for now
            "relaxation_ops": [sigmam1, sigmam2, Qobj(Z1), Qobj(Z2)],
            # relaxation operator lists for T1 and T2, respectively
            # "observation_space_size": 2*256 + 1 + 4 + 2 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
            # "observation_space_size": 2*16 + 1 + 4 + 2 # 2*16 = (complex number)*(target unitary matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
            # "observation_space_size": 2 * 256 + 2,
            "observation_space_size": 2 * 16 + 2
        }

    # physics: https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.10.054062, eq(2)
    # parameters: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.021058
    # 30 ns duration, g1 = 72.5 MHz, g2 = 71.5 MHz, g12 = 5 MHz
    # T1 = 60 us, 30 us
    # T2* = 66 us, 5 us

    def hamiltonian(self, detuning1, detuning2, alpha1, alpha2, g_eff, gamma_magnitude1, gamma_phase1, gamma_magnitude2,
                    gamma_phase2, index=1):
        selfEnergyTerms = (detuning1 + alpha1) * Z1 + (detuning2 + alpha2) * Z2
        Qubit1ControlTerms = gamma_magnitude1 * (np.cos(gamma_phase1) * X1 + np.sin(gamma_phase1) * Y1)
        Qubit2ControlTerms = gamma_magnitude2 * (np.cos(gamma_phase2) * X2 + np.sin(gamma_phase2) * Y2)

        if index == 1:
            interactionEnergy = g_eff * exchangeOperator1
        elif index == 2:
            interactionEnergy = g_eff * exchangeOperator2
        elif index == 3:
            interactionEnergy = g_eff * exchangeOperator3
        else:
            interactionEnergy = 0
            print("interaction kind not specified")

        energyTotal = selfEnergyTerms + interactionEnergy + Qubit1ControlTerms + Qubit2ControlTerms

        return energyTotal

    def __init__(self, env_config):
        self.final_time = env_config["final_time"]  # Final time for the gates
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(
        env_config["observation_space_size"],))  # propagation operator elements + fidelity + relaxation + detuning
        self.action_space = gym.spaces.Box(low=-0.1 * np.ones(27), high=0.1 * np.ones(
            27))  # alpha1, alpha2, alphaC, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2
        self.detuning_list = env_config["detuning_list"]
        self.detuning_update()
        self._U_target = self.unitary_to_superoperator(env_config["U_target"])
        self.unitary_U_target = env_config["U_target"]
        self.U_initial = self.unitary_to_superoperator(env_config["U_initial"])
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.steps_per_Haar = env_config["steps_per_Haar"]
        self.verbose = env_config["verbose"]
        self.relaxation_rates_list = env_config["relaxation_rates_list"]
        self.relaxation_ops = env_config["relaxation_ops"]
        self.relaxation_rate = self.get_relaxation_rate()
        self.current_Haar_num = 1  # starting with 1
        self.current_step_per_Haar = 1
        self.H_array = []  # saving all H's with Haar wavelet to be multiplied
        self.H_tot = []  # Haar wavelet multipied H summed up for each time bin
        self.L_array = []  # Liouvillian for each time bin
        self.U_array = []  # propagation operators for each time bin
        self.U = self.U_initial.copy()  # multiplied propagtion operators
        self.state = self.unitary_to_observation(self.U_initial)  # starting observation space
        # self.state = self.unitary_to_observation(self.unitary_U_target)  # starting observation space
        self.prev_fidelity = 0  # previous step' fidelity for rewarding
        # self.alpha_max = self.PiFreq / 2
        self.alpha_max = 0.05E9  # detuning of the control pulse in Hz
        self.PiFreq = np.pi / self.final_time / self.steps_per_Haar
        self.g_eff_max = self.PiFreq / 2
        # self.gamma_phase_max = np.pi
        # self.gamma_magnitude_max = self.PiFreq / 2
        self.gamma_phase_max = 1.1675 * np.pi
        self.gamma_magnitude_max = 1.8 * np.pi / self.final_time / self.steps_per_Haar

        self.transition_history = []
        self.env_config = env_config

    def detuning_update(self):
        qubit_1_detuning = random.sample(self.detuning_list[0], k=1)[0]
        qubit_2_detuning = random.sample(self.detuning_list[1], k=1)[0]

        self.detuning = [qubit_1_detuning, qubit_2_detuning]

    def update_target_unitary(self, U):
        self._U_target = self.unitary_to_superoperator(U)
        self.unitary_U_target = U

    def unitary_to_superoperator(self, U):
        return (spre(Qobj(U)) * spost(Qobj(U.conjugate().transpose()))).data.toarray()

    def get_relaxation_rate(self):
        relaxation_size = len(self.relaxation_ops)  # get number of relaxation ops

        sampled_rate_list = []
        for ii in range(relaxation_size):
            sampled_rate_list.append(random.sample(self.relaxation_rates_list[ii], k=1)[0])

        return sampled_rate_list

    def get_observation(self):
        normalized_detuning = [normalize(self.detuning[0], self.detuning_list[0]),
                               normalize(self.detuning[1], self.detuning_list[1])]
        # normalized_relaxation_rates = [normalize(self.relaxation_rate[0], self.relaxation_rates_list[0]),
        #                                normalize(self.relaxation_rate[1], self.relaxation_rates_list[1]),
        #                                normalize(self.relaxation_rate[2], self.relaxation_rates_list[2]),
        #                                normalize(self.relaxation_rate[3], self.relaxation_rates_list[3])]
        # return np.append([self.compute_fidelity()] +
        #                   normalized_relaxation_rates +
        #                   normalized_detuning,
        #                   self.unitary_to_observation(self.unitary_U_target))
        # U_diff = self._U_target @ self.U_initial.conj().T
        # return np.append(normalized_detuning,
        #                 self.unitary_to_observation(self._U_target))
        return np.append(normalized_detuning,
                         self.unitary_to_observation(self.unitary_U_target))

    def compute_fidelity(self):
        # U_target_dagger = self.unitary_to_superoperator(self.unitary_U_target.conjugate().transpose())
        # self._U_target.conjugate().transpose()
        return float(np.abs(np.trace(self._U_target.conjugate().transpose() @ self.U))) / (self.U.shape[0])
        # return F

    def unitary_to_observation(self, U):
        return (
            np.array(
                [(abs(x), (cmath.phase(x) / 2 / np.pi + 1) / 2) for x in U.flatten()],
                dtype=np.float64,
            )
            .squeeze()
            .reshape(-1)  # cmath phase gives -2pi to 2pi (?)
        )

    def reset(self, *, seed=None, options=None):
        # self.U = self.U_initial.copy()
        self.state = self.get_observation()
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H1_1_array = []
        self.H2_1_array = []
        self.H1_2_array = []
        self.H2_2_array = []
        self.H1_3_array = []
        self.H2_3_array = []
        self.H1_4_array = []

        self.H_tot1_1 = []
        self.H_tot2_1 = []
        self.H_tot1_2 = []
        self.H_tot2_2 = []
        self.H_tot1_3 = []
        self.H_tot2_3 = []
        self.H_tot1_4 = []

        self.L_array = []
        self.U_array = []
        self.prev_fidelity = 0
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning = 0
        self.detuning_update()
        starting_observeration = self.get_observation()
        info = {}
        return starting_observeration, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1)

        ### First single qubit gate
        alpha1_1 = self.alpha_max * action[0]
        alpha2_1 = self.alpha_max * action[1]

        gamma_magnitude1_1 = self.gamma_magnitude_max/2 * (action[2] +1)
        gamma_magnitude2_1 = self.gamma_magnitude_max/2 * (action[3]+1)

        gamma_phase1_1 = self.gamma_phase_max * action[4]
        gamma_phase2_1 = self.gamma_phase_max * action[5]

        ### First two qubit gate
        g_eff1 = self.g_eff_max * action[6]

        ### Second Single qubit gate
        alpha1_2 = self.alpha_max * action[7]
        alpha2_2 = self.alpha_max * action[8]

        gamma_magnitude1_2 = self.gamma_magnitude_max/2 * (action[9]+1)
        gamma_magnitude2_2 = self.gamma_magnitude_max/2 * (action[10] + 1)

        gamma_phase1_2 = self.gamma_phase_max * action[11]
        gamma_phase2_2 = self.gamma_phase_max * action[12]

        ### second two qubit gate
        g_eff2 = self.g_eff_max * action[13]

        ### Third Single qubit gate
        alpha1_3 = self.alpha_max * action[14]
        alpha2_3 = self.alpha_max * action[15]

        gamma_magnitude1_3 = self.gamma_magnitude_max/2 * (action[16]+1)
        gamma_magnitude2_3 = self.gamma_magnitude_max/2 * (action[17]+1)

        gamma_phase1_3 = self.gamma_phase_max * action[18]
        gamma_phase2_3 = self.gamma_phase_max * action[19]

        ### third two qubit gate
        g_eff3 = self.g_eff_max * action[20]

        ### Fourth Single qubit gate
        alpha1_4 = self.alpha_max * action[21]
        alpha2_4 = self.alpha_max * action[22]

        gamma_magnitude1_4 = self.gamma_magnitude_max/2 * (action[23]+1)
        gamma_magnitude2_4 = self.gamma_magnitude_max/2 * (action[24]+1)

        gamma_phase1_4 = self.gamma_phase_max * action[25]
        gamma_phase2_4 = self.gamma_phase_max * action[26]

        # Set noise opertors
        jump_ops = []
        for ii in range(len(self.relaxation_ops)):
            jump_ops.append(np.sqrt(self.relaxation_rate[ii]) * self.relaxation_ops[ii])

        # Hamiltonian with controls
        H2_1 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff1, 0, 0, 0, 0, index=1)
        H2_2 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff2, 0, 0, 0, 0, index=1)
        H2_3 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff3, 0, 0, 0, 0, index=1)

        H1_1 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_1, alpha2_1, 0, gamma_magnitude1_1,
                                gamma_phase1_1, gamma_magnitude2_1, gamma_phase2_1)
        H1_2 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_2, alpha2_2, 0, gamma_magnitude1_2,
                                gamma_phase1_2, gamma_magnitude2_2, gamma_phase2_2)
        H1_3 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_3, alpha2_3, 0, gamma_magnitude1_3,
                                gamma_phase1_3, gamma_magnitude2_3, gamma_phase2_3)
        H1_4 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_4, alpha2_4, 0, gamma_magnitude1_4,
                                gamma_phase1_4, gamma_magnitude2_4, gamma_phase2_4)

        self.H2_1_array.append(H2_1)  # Array of Hs at each Haar wavelet
        self.H2_2_array.append(H2_2)  # Array of Hs at each Haar wavelet
        self.H2_3_array.append(H2_3)  # Array of Hs at each Haar wavelet

        self.H1_1_array.append(H1_1)  # Array of Hs at each Haar wavelet
        self.H1_2_array.append(H1_2)  # Array of Hs at each Haar wavelet
        self.H1_3_array.append(H1_3)  # Array of Hs at each Haar wavelet
        self.H1_4_array.append(H1_4)  # Array of Hs at each Haar wavelet

        # H_tot for adding Hs at each time bins
        self.H_tot2_1 = []
        self.H_tot2_2 = []
        self.H_tot2_3 = []

        self.H_tot1_1 = []
        self.H_tot1_2 = []
        self.H_tot1_3 = []
        self.H_tot1_4 = []

        for ii, H_elem in enumerate(self.H1_1_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_1[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_1.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H2_1_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot2_1[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot2_1.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H1_2_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_2[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_2.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H2_2_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot2_2[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot2_2.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H1_3_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_3[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_3.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H2_3_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot2_3[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot2_3.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H1_4_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_4[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_4.append(factor * H_elem)

        # self.L = ([])  # at every step we calculate L again because minimal time bin changes
        self.U = np.eye(16)  # identity
        # self.U = self.U_initial.copy()
        # self.unitary_U = np.eye(4)

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_1[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            # self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            # unitary_Ut = la.expm(-1j * self.H_tot1_1[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            # self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot2_1[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            # self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            # unitary_Ut = la.expm(-1j * self.H_tot2_1[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            # self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_2[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            # self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            # unitary_Ut = la.expm(-1j * self.H_tot1_2[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            # self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot2_2[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            # self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            # unitary_Ut = la.expm(-1j * self.H_tot2_2[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            # self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_3[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            # self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            # unitary_Ut = la.expm(-1j * self.H_tot1_3[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            # self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot2_3[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            # self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            # unitary_Ut = la.expm(-1j * self.H_tot2_3[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            # self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_4[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            # self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            # unitary_Ut = la.expm(-1j * self.H_tot1_4[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            # self.unitary_U = unitary_Ut @ self.unitary_U

        # Reward and fidelity calculation
        fidelity = self.compute_fidelity()
        # if fidelity>0.49:
        # plot_complex_matrix(self.U, action, fidelity, "Matrix U")

        reward = (-3 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        # reward = (-7 * np.log10(1.0000001 - fidelity) + np.log10(1.0000001 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        # reward = (-1 * np.log10(1.0000001 - fidelity) + np.log10(1.0000001 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        if self.current_Haar_num == self.num_Haar_basis:
            self.transition_history.append([fidelity, reward, *action, *self.U.flatten()])

        # Determine if episode is over
        truncated = False
        terminated = False
        if fidelity >= 1:
            truncated = True  # truncated when target fidelity reached
        elif (self.current_Haar_num >= self.num_Haar_basis) and (
                self.current_step_per_Haar >= self.steps_per_Haar):  # terminate when all Haar is tested
            terminated = True
        else:
            terminated = False

        if (
                self.current_step_per_Haar == self.steps_per_Haar):  # For each Haar basis, if all trial steps ends, them move to next haar wavelet
            self.current_Haar_num += 1
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

        info = {}
        return (self.state, reward, terminated, truncated, info)
