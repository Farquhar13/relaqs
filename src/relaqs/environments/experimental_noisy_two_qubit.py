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

C_MATRIX = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1]])
C_MATRIX_INV = np.linalg.inv(C_MATRIX)
HSH = H @ S @ H
SdaggerH = Sdagger @ H

# magic basis
B = 1 / np.sqrt(2) * np.array([[1., 0, 0, 1.j], [0, 1.j, 1., 0],
                               [0, 1.j, -1., 0], [1., 0, 0, -1.j]])  # Magic Basis

# magic basis dagger
B_dagger = np.conj(B).T


class ExperimentalNoisyTwoQubitEnv(gym.Env):
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
        self.PiFreq = np.pi / self.final_time
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
        self.alpha_max = self.PiFreq / 2
        self.g_eff_max = self.PiFreq / 2
        self.gamma_phase_max = np.pi
        self.gamma_magnitude_max = self.PiFreq / 2
        self.transition_history = []
        self.env_config = env_config
        self.initialActions = self.KakActionCalculation()

    def detuning_update(self):
        qubit_1_detuning = random.sample(self.detuning_list[0], k=1)[0]
        qubit_2_detuning = random.sample(self.detuning_list[1], k=1)[0]

        self.detuning = [qubit_1_detuning, qubit_2_detuning]

    def update_target_unitary(self, U):
        self._U_target = self.unitary_to_superoperator(U)
        self.unitary_U_target = U
        self.initialActions = self.KakActionCalculation()

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
        return float(np.abs(np.trace(self._U_target.conjugate().transpose() @ self.U))) / (self.U.shape[0])

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
        self.initialActions = self.KakActionCalculation()
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

    def parse_actions(self, action):
        # Determine modifiers based on current_Haar_num
        if self.current_Haar_num == 1:
            # Use initialActions for everything and positive alpha
            alpha_sign = 1
            use_initial = True
        elif self.current_Haar_num == 2:
            # Use initialActions for single qubit gates, negative alpha
            alpha_sign = -1
            use_initial = True
        else:
            # Don't use initialActions, positive alpha
            alpha_sign = 1
            use_initial = False

        # Calculate all parameters at once with minimal branching
        # Single qubit gate 1
        alpha1_1 = alpha_sign * self.alpha_max * (action[0] + (self.initialActions[0] if use_initial else 0))
        alpha2_1 = alpha_sign * self.alpha_max * (action[1] + (self.initialActions[1] if use_initial else 0))
        gamma_magnitude1_1 = self.gamma_magnitude_max * (action[2] + (self.initialActions[2] if use_initial else 0))
        gamma_magnitude2_1 = self.gamma_magnitude_max * (action[3] + (self.initialActions[3] if use_initial else 0))
        gamma_phase1_1 = self.gamma_phase_max * (action[4] + (self.initialActions[4] if use_initial else 0))
        gamma_phase2_1 = self.gamma_phase_max * (action[5] + (self.initialActions[5] if use_initial else 0))

        # Single qubit gate 2
        alpha1_2 = alpha_sign * self.alpha_max * (action[7] + (self.initialActions[7] if use_initial else 0))
        alpha2_2 = alpha_sign * self.alpha_max * (action[8] + (self.initialActions[8] if use_initial else 0))
        gamma_magnitude1_2 = self.gamma_magnitude_max * (action[9] + (self.initialActions[9] if use_initial else 0))
        gamma_magnitude2_2 = self.gamma_magnitude_max * (action[10] + (self.initialActions[10] if use_initial else 0))
        gamma_phase1_2 = self.gamma_phase_max * (action[11] + (self.initialActions[11] if use_initial else 0))
        gamma_phase2_2 = self.gamma_phase_max * (action[12] + (self.initialActions[12] if use_initial else 0))

        # Single qubit gate 3
        alpha1_3 = alpha_sign * self.alpha_max * (action[14] + (self.initialActions[14] if use_initial else 0))
        alpha2_3 = alpha_sign * self.alpha_max * (action[15] + (self.initialActions[15] if use_initial else 0))
        gamma_magnitude1_3 = self.gamma_magnitude_max * (action[16] + (self.initialActions[16] if use_initial else 0))
        gamma_magnitude2_3 = self.gamma_magnitude_max * (action[17] + (self.initialActions[17] if use_initial else 0))
        gamma_phase1_3 = self.gamma_phase_max * (action[18] + (self.initialActions[18] if use_initial else 0))
        gamma_phase2_3 = self.gamma_phase_max * (action[19] + (self.initialActions[19] if use_initial else 0))

        # Single qubit gate 4
        alpha1_4 = alpha_sign * self.alpha_max * (action[21] + (self.initialActions[21] if use_initial else 0))
        alpha2_4 = alpha_sign * self.alpha_max * (action[22] + (self.initialActions[22] if use_initial else 0))
        gamma_magnitude1_4 = self.gamma_magnitude_max * (action[23] + (self.initialActions[23] if use_initial else 0))
        gamma_magnitude2_4 = self.gamma_magnitude_max * (action[24] + (self.initialActions[24] if use_initial else 0))
        gamma_phase1_4 = self.gamma_phase_max * (action[25] + (self.initialActions[25] if use_initial else 0))
        gamma_phase2_4 = self.gamma_phase_max * (action[26] + (self.initialActions[26] if use_initial else 0))

        # Two qubit gates - handle special case for Haar_num == 1
        if self.current_Haar_num == 1:
            g_eff1 = self.g_eff_max * (action[6] + self.initialActions[6])
            g_eff2 = self.g_eff_max * (action[13] + self.initialActions[13])
            g_eff3 = self.g_eff_max * (action[20] + self.initialActions[20])
        else:
            g_eff1 = self.g_eff_max * action[6]
            g_eff2 = self.g_eff_max * action[13]
            g_eff3 = self.g_eff_max * action[20]

        # Return all calculated values
        return (
            alpha1_1, alpha2_1, gamma_magnitude1_1, gamma_magnitude2_1, gamma_phase1_1, gamma_phase2_1,
            alpha1_2, alpha2_2, gamma_magnitude1_2, gamma_magnitude2_2, gamma_phase1_2, gamma_phase2_2,
            alpha1_3, alpha2_3, gamma_magnitude1_3, gamma_magnitude2_3, gamma_phase1_3, gamma_phase2_3,
            alpha1_4, alpha2_4, gamma_magnitude1_4, gamma_magnitude2_4, gamma_phase1_4, gamma_phase2_4,
            g_eff1, g_eff2, g_eff3
        )

    def hamiltonian_update(self, num_time_bins, H_tot, H_array, *hamiltonian_args):
        H = self.hamiltonian(*hamiltonian_args)
        H_array.append(H)
        for ii, H_elem in enumerate(H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))  # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    H_tot[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    H_tot.append(factor * H_elem)

    def operator_update(self, H_tot, num_time_bins, jump_ops):
        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(H_tot[jj]), jump_ops, data_only=False,
                             chi=None)).data.toarray()  # Liouvillian calc
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at

    def compute_reward(self, fidelity):
        return (-3 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)

    def is_episode_over(self, fidelity):
        truncated = False
        terminated = False
        if fidelity >= 1:
            truncated = True  # truncated when target fidelity reached
        elif (self.current_Haar_num >= self.num_Haar_basis) and (self.current_step_per_Haar >= self.steps_per_Haar):  # terminate when all Haar is tested
            terminated = True
        return truncated, terminated

    def Haar_update(self):
        if (self.current_step_per_Haar == self.steps_per_Haar):  # For each Haar basis, if all trial steps ends, them move to next haar wavelet
            self.current_Haar_num += 1
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1)
        self.initialActions = self.KakActionCalculation()
        (
            alpha1_1, alpha2_1, gamma_magnitude1_1, gamma_magnitude2_1,
            gamma_phase1_1, gamma_phase2_1,
            alpha1_2, alpha2_2, gamma_magnitude1_2, gamma_magnitude2_2,
            gamma_phase1_2, gamma_phase2_2,
            alpha1_3, alpha2_3, gamma_magnitude1_3, gamma_magnitude2_3,
            gamma_phase1_3, gamma_phase2_3,
            alpha1_4, alpha2_4, gamma_magnitude1_4, gamma_magnitude2_4,
            gamma_phase1_4, gamma_phase2_4,
            g_eff1, g_eff2, g_eff3
        ) = self.parse_actions(action)

        self.H2_1_array = []  # Array of Hs at each Haar wavelet
        self.H2_2_array = []  # Array of Hs at each Haar wavelet
        self.H2_3_array = []  # Array of Hs at each Haar wavelet

        self.H1_1_array = []  # Array of Hs at each Haar wavelet
        self.H1_2_array = []  # Array of Hs at each Haar wavelet
        self.H1_3_array = []  # Array of Hs at each Haar wavelet
        self.H1_4_array = []  # Array of Hs at each Haar wavelet

        # H_tot for adding Hs at each time bins
        self.H_tot2_1 = []
        self.H_tot2_2 = []
        self.H_tot2_3 = []

        self.H_tot1_1 = []
        self.H_tot1_2 = []
        self.H_tot1_3 = []
        self.H_tot1_4 = []

        self.hamiltonian_update(num_time_bins, self.H_tot2_1, self.H2_1_array,self.detuning[0], self.detuning[1], 0, 0, g_eff1, 0, 0, 0, 0)
        self.hamiltonian_update(num_time_bins, self.H_tot2_2, self.H2_2_array, self.detuning[0], self.detuning[1], 0, 0, g_eff2, 0, 0, 0, 0)
        self.hamiltonian_update(num_time_bins, self.H_tot2_3, self.H2_3_array, self.detuning[0], self.detuning[1], 0, 0, g_eff3, 0, 0, 0, 0)

        self.hamiltonian_update(num_time_bins, self.H_tot1_1, self.H1_1_array, self.detuning[0], self.detuning[1], alpha1_1, alpha2_1, 0, gamma_magnitude1_1,
                                gamma_phase1_1, gamma_magnitude2_1, gamma_phase2_1)
        self.hamiltonian_update(num_time_bins, self.H_tot1_2, self.H1_2_array, self.detuning[0], self.detuning[1], alpha1_2, alpha2_2, 0, gamma_magnitude1_2,
                                gamma_phase1_2, gamma_magnitude2_2, gamma_phase2_2)
        self.hamiltonian_update(num_time_bins, self.H_tot1_3, self.H1_3_array, self.detuning[0], self.detuning[1], alpha1_3, alpha2_3, 0, gamma_magnitude1_3,
                                gamma_phase1_3, gamma_magnitude2_3, gamma_phase2_3)
        self.hamiltonian_update(num_time_bins, self.H_tot1_4, self.H1_4_array, self.detuning[0], self.detuning[1], alpha1_4, alpha2_4, 0, gamma_magnitude1_4,
                                gamma_phase1_4, gamma_magnitude2_4, gamma_phase2_4)

        self.U = np.eye(16)  # identity

        # Set noise opertors
        jump_ops = []
        for ii in range(len(self.relaxation_ops)):
            jump_ops.append(np.sqrt(self.relaxation_rate[ii]) * self.relaxation_ops[ii])

        H_tot_list = [self.H_tot1_1, self.H_tot2_1, self.H_tot1_2, self.H_tot2_2, self.H_tot1_3, self.H_tot2_3, self.H_tot1_4]
        for H_tot in H_tot_list:
            self.operator_update(H_tot, num_time_bins, jump_ops)


        # Reward and fidelity calculation
        fidelity = self.compute_fidelity()
        reward = self.compute_reward(fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        if self.current_Haar_num == self.num_Haar_basis:
            self.transition_history.append([fidelity, reward, *action, *self.U.flatten()])

        truncated, terminated = self.is_episode_over(fidelity)

        self.Haar_update()

        info = {}
        return (self.state, reward, terminated, truncated, info)

    def canonicalDecomposition(self):

        ## This part of the code is from https://github.com/mpham26uchicago/laughing-umbrella/

        def decompose_one_qubit_product(
                U: np.ndarray, validate_input: bool = True, atol: float = 1e-8, rtol: float = 1e-5
        ):
            i, j = np.unravel_index(np.argmax(U, axis=None), U.shape)

            def u1_set(i):
                return (1, 3) if i % 2 else (0, 2)

            def u2_set(i):
                return (0, 1) if i < 2 else (2, 3)

            u1 = U[np.ix_(u1_set(i), u1_set(j))]
            u2 = U[np.ix_(u2_set(i), u2_set(j))]

            u1 = to_su(u1)
            u2 = to_su(u2)

            phase = U[i, j] / (u1[i // 2, j // 2] * u2[i % 2, j % 2])

            return phase, u1, u2

        def to_su(u: np.ndarray) -> np.ndarray:

            return u * complex(np.linalg.det(u)) ** (-1 / np.shape(u)[0])

        def KAK_2q(
                U: np.ndarray,
                rounding: int = 19
        ) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float,
        float, float]:

            # 0. Map U(4) to SU(4) (and phase)
            U = U / np.linalg.det(U) ** 0.25

            assert np.isclose(np.linalg.det(U), 1), "Determinant of U is not 1"

            # 1. Unconjugate U into the magic basis
            U_prime = B_dagger @ U @ B

            # Isolating the maximal torus
            Theta = lambda U: np.conj(U)
            M_squared = Theta(np.conj(U_prime).T) @ U_prime

            if rounding is not None:
                M_squared = np.round(M_squared, rounding)  # For numerical stability

            ## 2. Diagonalizing M^2
            D, P = np.linalg.eig(M_squared)

            ## Check and correct for det(P) = -1
            if np.isclose(np.linalg.det(P), -1):
                P[:, 0] *= -1  # Multiply the first eigenvector by -1

            # 3. Extracting K2
            K2 = np.conj(P).T

            assert np.allclose(K2 @ K2.T, np.identity(4)), "K2 is not orthogonal"
            assert np.isclose(np.linalg.det(K2), 1), "Determinant of K2 is not 1"

            # 4. Extracting A
            A = np.sqrt(D)

            ## Check and correct for det(A) = -1
            if np.isclose(np.prod(A), -1):
                A[0] *= -1  # Multiply the first eigenvalue by -1

            A = np.diag(A)  # Turn the list of eigenvalues into a diagonal matrix

            assert np.isclose(np.linalg.det(A), 1), "Determinant of A is not 1"

            # 5. Extracting K1
            K1 = U_prime @ np.conj(K2).T @ np.conj(A).T

            assert np.allclose(K1 @ K1.T, np.identity(4)), "K1 is not orthogonal"
            assert np.isclose(np.linalg.det(K1), 1), "Determinant of K1 is not 1"

            # 6. Extracting Local Gates
            L = B @ K1 @ B_dagger  # Left Local Product
            R = B @ K2 @ B_dagger  # Right Local Product

            phase1, L1, L2 = decompose_one_qubit_product(L)  # L1 (top), L2(bottom)
            phase2, R1, R2 = decompose_one_qubit_product(R)  # R1 (top), R2(bottom)

            # 7. Extracting the Canonical Parameters

            theta_vec = np.angle(np.diag(A))[:3]  # theta vector
            a0, a1, a2 = C_MATRIX_INV @ theta_vec  # Computing the "a"-vector

            # 8. Unpack Parameters and Put into Weyl chamber
            c0, c1, c2 = 2 * a1, -2 * a0, 2 * a2  # Unpack parameters

            CAN = lambda c0, c1, c2: expm(1j / 2 * (c0 * np.kron(X, X) + c1 * np.kron(Y, Y) + c2 * np.kron(Z, Z)))

            assert np.allclose(U, (phase1 * np.kron(L1, L2)) @ CAN(c0, c1, c2)
                               @ (phase2 * np.kron(R1, R2)), atol=1e-03), "U does not equal KAK"

            return phase1, L1, L2, phase2, R1, R2, c0, c1, c2

        return KAK_2q(self.unitary_U_target)

    def KakActionCalculation(self):

        phase1, L1, L2, phase2, R1, R2, c0, c1, c2 = self.canonicalDecomposition()

        initialActions = np.zeros(27)

        initialActions[0] = self.singleQubitActionCalculation(R1)[0]
        initialActions[1] = self.singleQubitActionCalculation(R2)[0]
        initialActions[2] = self.singleQubitActionCalculation(R1)[1]
        initialActions[3] = self.singleQubitActionCalculation(R2)[1]
        initialActions[4] = self.singleQubitActionCalculation(R1)[2]
        initialActions[5] = self.singleQubitActionCalculation(R2)[2]

        initialActions[6] = self.canonicalActionCalculation(c0, c1, c2, 1)

        initialActions[7] = self.singleQubitActionCalculation(H)[0]
        initialActions[8] = self.singleQubitActionCalculation(H)[0]
        initialActions[9] = self.singleQubitActionCalculation(H)[1]
        initialActions[10] = self.singleQubitActionCalculation(H)[1]
        initialActions[11] = self.singleQubitActionCalculation(H)[2]
        initialActions[12] = self.singleQubitActionCalculation(H)[2]

        initialActions[13] = self.canonicalActionCalculation(c0, c1, c2, 2)

        initialActions[14] = self.singleQubitActionCalculation(HSH)[0]
        initialActions[15] = self.singleQubitActionCalculation(HSH)[0]
        initialActions[16] = self.singleQubitActionCalculation(HSH)[1]
        initialActions[17] = self.singleQubitActionCalculation(HSH)[1]
        initialActions[18] = self.singleQubitActionCalculation(HSH)[2]
        initialActions[19] = self.singleQubitActionCalculation(HSH)[2]

        initialActions[20] = self.canonicalActionCalculation(c0, c1, c2, 3)

        L1SdaggerH = L1 @ SdaggerH
        L2SdaggerH = L2 @ SdaggerH
        initialActions[21] = self.singleQubitActionCalculation(L1SdaggerH)[0]
        initialActions[22] = self.singleQubitActionCalculation(L2SdaggerH)[0]
        initialActions[23] = self.singleQubitActionCalculation(L1SdaggerH)[1]
        initialActions[24] = self.singleQubitActionCalculation(L2SdaggerH)[1]
        initialActions[25] = self.singleQubitActionCalculation(L1SdaggerH)[2]
        initialActions[26] = self.singleQubitActionCalculation(L2SdaggerH)[2]

        return initialActions

    def singleQubitActionCalculation(self, U):

        singleQubitActions = np.zeros(3)

        x_angle, z_angle_after, z_angle_before = OneQubitEulerDecomposer(basis='ZXZ').angles(U)

        singleQubitActions[0] = np.mod((z_angle_after + z_angle_before) / np.pi, 2)  ## alpha
        singleQubitActions[1] = np.mod(x_angle / np.pi, 2)  ## gamma_magnitude
        singleQubitActions[2] = np.mod(- z_angle_before / np.pi, 2)  ## gamma_phase

        return singleQubitActions

    def Rn(self, theta, axisSelection):
        return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * axisSelection

    def ZXZ_Rotation_Generation(self, angles):
        return self.unitary_normalization(self.Rn(angles[1], Z) @ self.Rn(angles[0], X) @ self.Rn(angles[2], Z))

    def unitary_normalization(self, unitary_in):
        if unitary_in[0][0] == 0:
            unitary_in = unitary_in / unitary_in[0][1]
        else:
            unitary_in = unitary_in / unitary_in[0][0]

        return unitary_in

    def canonicalActionCalculation(self, c0, c1, c2, index=1):

        # twoQubitAction = 0

        if index == 1:
            b = 1 / 2 * (c0 + c1 - c2)
        elif index == 2:
            b = 1 / 2 * (-c0 + c1 + c2)
        elif index == 3:
            b = 1 / 2 * (c0 - c1 + c2)
        else:
            print("wrong input index")

        twoQubitAction = - b / np.pi

        return twoQubitAction
