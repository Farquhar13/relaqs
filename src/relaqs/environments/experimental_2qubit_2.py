from typing import Tuple
import cmath
import random
import numpy as np
import scipy.linalg as la

from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
from qutip.qip.operations import cnot, cphase
from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from relaqs.api.utils import normalize
import gymnasium as gym

# Use Numba for JIT compiling critical helper functions.
from numba import njit

# ------------------------------------------------------------------------------
# Define Fixed Matrices and Operators
# ------------------------------------------------------------------------------

sig_p = np.array([[0, 1.], [0, 0]])
sig_m = np.array([[0, 0], [1., 0]])
X = np.array([[0, 1.], [1., 0]])
Z = np.array([[1., 0], [0, -1.]])
I = np.array([[1., 0], [0, 1.]])
Y = np.array([[0, -1.j], [1.j, 0]])

H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
S = np.array([[1., 0], [0, 1.j]])
Sdagger = np.array([[1., 0], [0, -1.j]])

# Two-qubit single-qubit gates
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

# Two-qubit gate basis
XX = tensor(Qobj(X), Qobj(X)).data.toarray()
YY = tensor(Qobj(Y), Qobj(Y)).data.toarray()
ZZ = tensor(Qobj(Z), Qobj(Z)).data.toarray()
exchangeOperator1 = XX + YY
exchangeOperator2 = YY + ZZ
exchangeOperator3 = XX + ZZ

CNOT = cnot().data.toarray()
CZ = cphase(np.pi).data.toarray()

C_MATRIX = np.array([[1, 1, 1],
                     [-1, 1, -1],
                     [1, -1, -1]])
C_MATRIX_INV = np.linalg.inv(C_MATRIX)
HSH = H @ S @ H
SdaggerH = Sdagger @ H

# Magic basis and its dagger
B = 1 / np.sqrt(2) * np.array([[1., 0, 0, 1.j],
                               [0, 1.j, 1., 0],
                               [0, 1.j, -1., 0],
                               [1., 0, 0, -1.j]])
B_dagger = np.conj(B).T


# ------------------------------------------------------------------------------
# Numba-Accelerated Helper Function
# ------------------------------------------------------------------------------
# This function takes a 3D NumPy array of Hamiltonians (shape: (n, m, m)),
# the number of steps per Haar basis, the current Haar order,
# and the number of time bins, and returns a 3D array (num_time_bins x m x m)
# with the appropriate sign factors accumulated.
@njit
def compute_H_tot_numba(H_stack, steps_per_Haar, current_Haar_num, num_time_bins):
    n = H_stack.shape[0]
    m1 = H_stack.shape[1]
    m2 = H_stack.shape[2]
    H_tot = np.zeros((num_time_bins, m1, m2), dtype=H_stack.dtype)
    for ii in range(n):
        # Compute the effective Haar order (note: current_Haar_num is 1-indexed)
        Haar_order = current_Haar_num - np.floor(ii / steps_per_Haar)
        order = int(Haar_order)
        if order < 1:
            order = 1  # safeguard against negative orders
        for jj in range(num_time_bins):
            exponent = int(np.floor(jj / (2 ** (order - 1))))
            factor = (-1) ** exponent
            H_tot[jj] += factor * H_stack[ii]
    return H_tot


# ------------------------------------------------------------------------------
# Propagation Helper Function (Not JIT-compiled due to Qobj calls)
# ------------------------------------------------------------------------------
def propagate_U(U_init, H_tot_list, final_time, num_time_bins, jump_ops):
    U = U_init.copy()
    for H_tot in H_tot_list:
        for jj in range(num_time_bins):
            # Compute Liouvillian from H_tot[jj]. (Using Qobj conversion)
            L = liouvillian(Qobj(H_tot[jj]), jump_ops, data_only=False, chi=None).data.toarray()
            Ut = la.expm(final_time / num_time_bins * L)
            U = Ut @ U
    return U


# ------------------------------------------------------------------------------
# Environment Class
# ------------------------------------------------------------------------------
class ExpNoisyTwoQubitEnv(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "action_space_size": 27,
            "U_initial": II,  # starting with I
            "U_target": CZ,  # target for CZ
            "final_time": 30E-9,
            "num_Haar_basis": 3,
            "steps_per_Haar": 1,
            "detuning_list": np.random.normal(0, np.pi / 100 / 30E-9, size=(2, 100)).tolist(),
            "save_data_every_step": 1,
            "verbose": True,
            "relaxation_rates_list": [[0], [0], [0], [0]],
            "relaxation_ops": [sigmam1, sigmam2, Qobj(Z1), Qobj(Z2)],
            "observation_space_size": 2 * 16 + 2
        }

    def hamiltonian(self, detuning1, detuning2, alpha1, alpha2, g_eff,
                    gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2, index=1):
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
        return selfEnergyTerms + interactionEnergy + Qubit1ControlTerms + Qubit2ControlTerms

    def __init__(self, env_config):
        self.final_time = env_config["final_time"]
        self.PiFreq = np.pi / self.final_time
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=-0.1 * np.ones(27), high=0.1 * np.ones(27))
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
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        # Arrays for storing Hamiltonians for each gate branch
        self.H1_1_array = []
        self.H2_1_array = []
        self.H1_2_array = []
        self.H2_2_array = []
        self.H1_3_array = []
        self.H2_3_array = []
        self.H1_4_array = []
        self.L_array = []
        self.U_array = []
        self.U = self.U_initial.copy()
        self.state = self.unitary_to_observation(self.U_initial)
        self.prev_fidelity = 0
        self.alpha_max = self.PiFreq / 2
        self.g_eff_max = self.PiFreq / 2
        self.gamma_phase_max = np.pi
        self.gamma_magnitude_max = self.PiFreq / 2
        self.transition_history = []
        self.env_config = env_config
        self.initialActions = self.KakActionCalculation()

    def detuning_update(self):
        self.detuning = [random.sample(self.detuning_list[0], k=1)[0],
                         random.sample(self.detuning_list[1], k=1)[0]]

    def update_target_unitary(self, U):
        self._U_target = self.unitary_to_superoperator(U)
        self.unitary_U_target = U
        self.initialActions = self.KakActionCalculation()

    def unitary_to_superoperator(self, U):
        return (spre(Qobj(U)) * spost(Qobj(U.conjugate().transpose()))).data.toarray()

    def get_relaxation_rate(self):
        return [random.sample(rate_list, k=1)[0] for rate_list in self.relaxation_rates_list]

    def get_observation(self):
        normalized_detuning = [normalize(self.detuning[0], self.detuning_list[0]),
                               normalize(self.detuning[1], self.detuning_list[1])]
        return np.append(normalized_detuning, self.unitary_to_observation(self.unitary_U_target))

    def compute_fidelity(self):
        return float(np.abs(np.trace(self._U_target.conjugate().transpose() @ self.U))) / self.U.shape[0]

    def unitary_to_observation(self, U):
        # Vectorized conversion: pairs (magnitude, normalized phase) per element.
        mags = np.abs(U.flatten())
        phases = np.angle(U.flatten())
        norm_phases = (phases / (2 * np.pi)) + 0.5
        obs = np.empty(2 * len(mags))
        obs[0::2] = mags
        obs[1::2] = norm_phases
        return obs

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
        self.L_array = []
        self.U_array = []
        self.prev_fidelity = 0
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning = 0
        self.detuning_update()
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1)
        self.initialActions = self.KakActionCalculation()

        # --- First Single Qubit Gate ---
        if self.current_Haar_num == 1:
            alpha1_1 = self.alpha_max * (action[0] + self.initialActions[0])
            alpha2_1 = self.alpha_max * (action[1] + self.initialActions[1])
            gamma_magnitude1_1 = self.gamma_magnitude_max * (action[2] + self.initialActions[2])
            gamma_magnitude2_1 = self.gamma_magnitude_max * (action[3] + self.initialActions[3])
            gamma_phase1_1 = self.gamma_phase_max * (action[4] + self.initialActions[4])
            gamma_phase2_1 = self.gamma_phase_max * (action[5] + self.initialActions[5])
        elif self.current_Haar_num == 2:
            alpha1_1 = - self.alpha_max * (action[0] + self.initialActions[0])
            alpha2_1 = - self.alpha_max * (action[1] + self.initialActions[1])
            gamma_magnitude1_1 = self.gamma_magnitude_max * (action[2] + self.initialActions[2])
            gamma_magnitude2_1 = self.gamma_magnitude_max * (action[3] + self.initialActions[3])
            gamma_phase1_1 = self.gamma_phase_max * (action[4] + self.initialActions[4])
            gamma_phase2_1 = self.gamma_phase_max * (action[5] + self.initialActions[5])
        else:
            alpha1_1 = self.alpha_max * action[0]
            alpha2_1 = self.alpha_max * action[1]
            gamma_magnitude1_1 = self.gamma_magnitude_max * action[2]
            gamma_magnitude2_1 = self.gamma_magnitude_max * action[3]
            gamma_phase1_1 = self.gamma_phase_max * action[4]
            gamma_phase2_1 = self.gamma_phase_max * action[5]

        # --- First Two Qubit Gate ---
        if self.current_Haar_num == 1:
            g_eff1 = self.g_eff_max * (action[6] + self.initialActions[6])
        else:
            g_eff1 = self.g_eff_max * action[6]

        # --- Second Single Qubit Gate ---
        if self.current_Haar_num == 1:
            alpha1_2 = self.alpha_max * (action[7] + self.initialActions[7])
            alpha2_2 = self.alpha_max * (action[8] + self.initialActions[8])
            gamma_magnitude1_2 = self.gamma_magnitude_max * (action[9] + self.initialActions[9])
            gamma_magnitude2_2 = self.gamma_magnitude_max * (action[10] + self.initialActions[10])
            gamma_phase1_2 = self.gamma_phase_max * (action[11] + self.initialActions[11])
            gamma_phase2_2 = self.gamma_phase_max * (action[12] + self.initialActions[12])
        elif self.current_Haar_num == 2:
            alpha1_2 = - self.alpha_max * (action[7] + self.initialActions[7])
            alpha2_2 = - self.alpha_max * (action[8] + self.initialActions[8])
            gamma_magnitude1_2 = self.gamma_magnitude_max * (action[9] + self.initialActions[9])
            gamma_magnitude2_2 = self.gamma_magnitude_max * (action[10] + self.initialActions[10])
            gamma_phase1_2 = self.gamma_phase_max * (action[11] + self.initialActions[11])
            gamma_phase2_2 = self.gamma_phase_max * (action[12] + self.initialActions[12])
        else:
            alpha1_2 = self.alpha_max * action[7]
            alpha2_2 = self.alpha_max * action[8]
            gamma_magnitude1_2 = self.gamma_magnitude_max * action[9]
            gamma_magnitude2_2 = self.gamma_magnitude_max * action[10]
            gamma_phase1_2 = self.gamma_phase_max * action[11]
            gamma_phase2_2 = self.gamma_phase_max * action[12]

        # --- Second Two Qubit Gate ---
        if self.current_Haar_num == 1:
            g_eff2 = self.g_eff_max * (action[13] + self.initialActions[13])
        else:
            g_eff2 = self.g_eff_max * action[13]

        # --- Third Single Qubit Gate ---
        if self.current_Haar_num == 1:
            alpha1_3 = self.alpha_max * (action[14] + self.initialActions[14])
            alpha2_3 = self.alpha_max * (action[15] + self.initialActions[15])
            gamma_magnitude1_3 = self.gamma_magnitude_max * (action[16] + self.initialActions[16])
            gamma_magnitude2_3 = self.gamma_magnitude_max * (action[17] + self.initialActions[17])
            gamma_phase1_3 = self.gamma_phase_max * (action[18] + self.initialActions[18])
            gamma_phase2_3 = self.gamma_phase_max * (action[19] + self.initialActions[19])
        elif self.current_Haar_num == 2:
            alpha1_3 = - self.alpha_max * (action[14] + self.initialActions[14])
            alpha2_3 = - self.alpha_max * (action[15] + self.initialActions[15])
            gamma_magnitude1_3 = self.gamma_magnitude_max * (action[16] + self.initialActions[16])
            gamma_magnitude2_3 = self.gamma_magnitude_max * (action[17] + self.initialActions[17])
            gamma_phase1_3 = self.gamma_phase_max * (action[18] + self.initialActions[18])
            gamma_phase2_3 = self.gamma_phase_max * (action[19] + self.initialActions[19])
        else:
            alpha1_3 = self.alpha_max * action[14]
            alpha2_3 = self.alpha_max * action[15]
            gamma_magnitude1_3 = self.gamma_magnitude_max * action[16]
            gamma_magnitude2_3 = self.gamma_magnitude_max * action[17]
            gamma_phase1_3 = self.gamma_phase_max * action[18]
            gamma_phase2_3 = self.gamma_phase_max * action[19]

        # --- Third Two Qubit Gate ---
        if self.current_Haar_num == 1:
            g_eff3 = self.g_eff_max * (action[20] + self.initialActions[20])
        else:
            g_eff3 = self.g_eff_max * action[20]

        # --- Fourth Single Qubit Gate ---
        if self.current_Haar_num == 1:
            alpha1_4 = self.alpha_max * (action[21] + self.initialActions[21])
            alpha2_4 = self.alpha_max * (action[22] + self.initialActions[22])
            gamma_magnitude1_4 = self.gamma_magnitude_max * (action[23] + self.initialActions[23])
            gamma_magnitude2_4 = self.gamma_magnitude_max * (action[24] + self.initialActions[24])
            gamma_phase1_4 = self.gamma_phase_max * (action[25] + self.initialActions[25])
            gamma_phase2_4 = self.gamma_phase_max * (action[26] + self.initialActions[26])
        elif self.current_Haar_num == 2:
            alpha1_4 = - self.alpha_max * (action[21] + self.initialActions[21])
            alpha2_4 = - self.alpha_max * (action[22] + self.initialActions[22])
            gamma_magnitude1_4 = self.gamma_magnitude_max * (action[23] + self.initialActions[23])
            gamma_magnitude2_4 = self.gamma_magnitude_max * (action[24] + self.initialActions[24])
            gamma_phase1_4 = self.gamma_phase_max * (action[25] + self.initialActions[25])
            gamma_phase2_4 = self.gamma_phase_max * (action[26] + self.initialActions[26])
        else:
            alpha1_4 = self.alpha_max * action[21]
            alpha2_4 = self.alpha_max * action[22]
            gamma_magnitude1_4 = self.gamma_magnitude_max * action[23]
            gamma_magnitude2_4 = self.gamma_magnitude_max * action[24]
            gamma_phase1_4 = self.gamma_phase_max * action[25]
            gamma_phase2_4 = self.gamma_phase_max * action[26]

        # Set noise operators (jump_ops)
        jump_ops = [np.sqrt(self.relaxation_rate[ii]) * op for ii, op in enumerate(self.relaxation_ops)]

        # Hamiltonian calculations for each gate block.
        H2_1 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff1, 0, 0, 0, 0, index=1)
        H2_2 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff2, 0, 0, 0, 0, index=1)
        H2_3 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff3, 0, 0, 0, 0, index=1)

        H1_1 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_1, alpha2_1, 0,
                                gamma_magnitude1_1, gamma_phase1_1, gamma_magnitude2_1, gamma_phase2_1)
        H1_2 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_2, alpha2_2, 0,
                                gamma_magnitude1_2, gamma_phase1_2, gamma_magnitude2_2, gamma_phase2_2)
        H1_3 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_3, alpha2_3, 0,
                                gamma_magnitude1_3, gamma_phase1_3, gamma_magnitude2_3, gamma_phase2_3)
        H1_4 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_4, alpha2_4, 0,
                                gamma_magnitude1_4, gamma_phase1_4, gamma_magnitude2_4, gamma_phase2_4)

        self.H2_1_array.append(H2_1)
        self.H2_2_array.append(H2_2)
        self.H2_3_array.append(H2_3)
        self.H1_1_array.append(H1_1)
        self.H1_2_array.append(H1_2)
        self.H1_3_array.append(H1_3)
        self.H1_4_array.append(H1_4)

        # For each Hamiltonian branch, first convert the Python list to a 3D NumPy array.
        H_tot1_1 = compute_H_tot_numba(np.stack(self.H1_1_array), self.steps_per_Haar, self.current_Haar_num,
                                       num_time_bins)
        H_tot2_1 = compute_H_tot_numba(np.stack(self.H2_1_array), self.steps_per_Haar, self.current_Haar_num,
                                       num_time_bins)
        H_tot1_2 = compute_H_tot_numba(np.stack(self.H1_2_array), self.steps_per_Haar, self.current_Haar_num,
                                       num_time_bins)
        H_tot2_2 = compute_H_tot_numba(np.stack(self.H2_2_array), self.steps_per_Haar, self.current_Haar_num,
                                       num_time_bins)
        H_tot1_3 = compute_H_tot_numba(np.stack(self.H1_3_array), self.steps_per_Haar, self.current_Haar_num,
                                       num_time_bins)
        H_tot2_3 = compute_H_tot_numba(np.stack(self.H2_3_array), self.steps_per_Haar, self.current_Haar_num,
                                       num_time_bins)
        H_tot1_4 = compute_H_tot_numba(np.stack(self.H1_4_array), self.steps_per_Haar, self.current_Haar_num,
                                       num_time_bins)

        # Combine all H_tot arrays in the order desired.
        H_tot_list = [H_tot1_1, H_tot2_1, H_tot1_2, H_tot2_2, H_tot1_3, H_tot2_3, H_tot1_4]
        self.U = propagate_U(np.eye(16, dtype=np.complex128), H_tot_list, self.final_time, num_time_bins, jump_ops)

        # Compute fidelity and reward.
        fidelity = self.compute_fidelity()
        reward = (-3 * np.log10(1.0000001 - fidelity) + np.log10(1.0000001 - self.prev_fidelity)) + \
                 (3 * fidelity - self.prev_fidelity)
        self.prev_fidelity = fidelity
        self.state = self.get_observation()

        if self.current_Haar_num == self.num_Haar_basis:
            self.transition_history.append([fidelity, reward, *action, *self.U.flatten()])

        # Episode termination flags.
        truncated = fidelity >= 1
        terminated = (self.current_Haar_num >= self.num_Haar_basis) and (
                    self.current_step_per_Haar >= self.steps_per_Haar)

        if self.current_step_per_Haar == self.steps_per_Haar:
            self.current_Haar_num += 1
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

        info = {}
        return (self.state, reward, terminated, truncated, info)

    def canonicalDecomposition(self):
        # --- Canonical decomposition routines (from your original code) ---
        def decompose_one_qubit_product(U: np.ndarray, validate_input: bool = True, atol: float = 1e-8,
                                        rtol: float = 1e-5):
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

        def KAK_2q(U: np.ndarray, rounding: int = 19) -> Tuple[
            float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, float, float]:
            U = U / np.linalg.det(U) ** 0.25
            assert np.isclose(np.linalg.det(U), 1), "Determinant of U is not 1"
            U_prime = B_dagger @ U @ B
            Theta = lambda U: np.conj(U)
            M_squared = Theta(np.conj(U_prime).T) @ U_prime
            if rounding is not None:
                M_squared = np.round(M_squared, rounding)
            D, P = np.linalg.eig(M_squared)
            if np.isclose(np.linalg.det(P), -1):
                P[:, 0] *= -1
            K2 = np.conj(P).T
            assert np.allclose(K2 @ K2.T, np.identity(4)), "K2 is not orthogonal"
            assert np.isclose(np.linalg.det(K2), 1), "Determinant of K2 is not 1"
            A = np.sqrt(D)
            if np.isclose(np.prod(A), -1):
                A[0] *= -1
            A = np.diag(A)
            assert np.isclose(np.linalg.det(A), 1), "Determinant of A is not 1"
            K1 = U_prime @ np.conj(K2).T @ np.conj(A).T
            assert np.allclose(K1 @ K1.T, np.identity(4)), "K1 is not orthogonal"
            assert np.isclose(np.linalg.det(K1), 1), "Determinant of K1 is not 1"
            L = B @ K1 @ B_dagger
            R = B @ K2 @ B_dagger
            phase1, L1, L2 = decompose_one_qubit_product(L)
            phase2, R1, R2 = decompose_one_qubit_product(R)
            theta_vec = np.angle(np.diag(A))[:3]
            a0, a1, a2 = C_MATRIX_INV @ theta_vec
            c0, c1, c2 = 2 * a1, -2 * a0, 2 * a2
            CAN = lambda c0, c1, c2: la.expm(1j / 2 * (c0 * np.kron(X, X) + c1 * np.kron(Y, Y) + c2 * np.kron(Z, Z)))
            assert np.allclose(U, (phase1 * np.kron(L1, L2)) @ CAN(c0, c1, c2) @ (phase2 * np.kron(R1, R2)),
                               atol=1e-03), "U does not equal KAK"
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
        singleQubitActions[0] = np.mod((z_angle_after + z_angle_before) / np.pi, 2)
        singleQubitActions[1] = np.mod(x_angle / np.pi, 2)
        singleQubitActions[2] = np.mod(- z_angle_before / np.pi, 2)
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
        if index == 1:
            b = 0.5 * (c0 + c1 - c2)
        elif index == 2:
            b = 0.5 * (-c0 + c1 + c2)
        elif index == 3:
            b = 0.5 * (c0 - c1 + c2)
        else:
            print("Wrong input index")
            b = 0.0
        return -b / np.pi