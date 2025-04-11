from typing import Tuple, List, Dict, Any

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

# Pre-define basic operators as constants
sig_p = np.array([[0, 1.], [0, 0]])
sig_m = np.array([[0, 0], [1., 0]])
X = np.array([[0, 1.], [1., 0]])
Z = np.array([[1., 0], [0, -1.]])
I = np.array([[1., 0], [0, 1.]])
Y = np.array([[0, -1.j], [1.j, 0]])

H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
S = np.array([[1., 0], [0, 1.j]])
Sdagger = np.array([[1., 0], [0, -1.j]])

# Pre-compute two-qubit operators only once
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

# Pre-compute exchange operators
XX = tensor(Qobj(X), Qobj(X)).data.toarray()
YY = tensor(Qobj(Y), Qobj(Y)).data.toarray()
ZZ = tensor(Qobj(Z), Qobj(Z)).data.toarray()
exchangeOperator1 = XX + YY
exchangeOperator2 = YY + ZZ
exchangeOperator3 = XX + ZZ

# Pre-define gate constants
CNOT = cnot().data.toarray()
CZ = cphase(np.pi).data.toarray()

# Coefficient matrix for canonical parameters
C_MATRIX = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1]])
C_MATRIX_INV = np.linalg.inv(C_MATRIX)
HSH = H @ S @ H
SdaggerH= Sdagger @ H

class NoisyTwoQubitEnv(gym.Env):
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
            # "observation_space_size": 2*256 + 1 + 4 + 2 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
            "observation_space_size": 2 * 16 + 1 + 4 + 2
            # 2*16 = (complex number)*(target unitary matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
        }

    def __init__(self, env_config):
        # Store configuration
        self.env_config = env_config
        self.final_time = env_config["final_time"]
        self.PiFreq = np.pi / self.final_time
        self.detuning_list = env_config["detuning_list"]
        self._U_target = self.unitary_to_superoperator(env_config["U_target"])
        self.unitary_U_target = env_config["U_target"]
        self.U_initial = self.unitary_to_superoperator(env_config["U_initial"])
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.steps_per_Haar = env_config["steps_per_Haar"]
        self.verbose = env_config["verbose"]
        self.relaxation_rates_list = env_config["relaxation_rates_list"]
        self.relaxation_ops = env_config["relaxation_ops"]

        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(env_config["observation_space_size"],)
        )
        self.action_space = gym.spaces.Box(
            low=-0.1 * np.ones(27), high=0.1 * np.ones(27)
        )

        # Precompute values
        self.alpha_max = self.PiFreq / 2
        self.g_eff_max = self.PiFreq / 2
        self.gamma_phase_max = np.pi
        self.gamma_magnitude_max = self.PiFreq / 2

        # State variables
        self.detuning = [0, 0]
        self.relaxation_rate = None
        self.U = None
        self.unitary_U = None
        self.state = None
        self.prev_fidelity = 0
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.transition_history = []

        # Haar arrays
        self.H_arrays = {}
        self.H_tots = {}

        # Initialize KAK decomposition
        self.initialActions = None
        self.reset()

    def hamiltonian(self, detuning1, detuning2, alpha1, alpha2, g_eff, gamma_magnitude1, gamma_phase1, gamma_magnitude2,
                    gamma_phase2, index=1):
        """Construct the Hamiltonian with the given parameters"""
        selfEnergyTerms = (detuning1 + alpha1) * Z1 + (detuning2 + alpha2) * Z2
        Qubit1ControlTerms = gamma_magnitude1 * (np.cos(gamma_phase1) * X1 + np.sin(gamma_phase1) * Y1)
        Qubit2ControlTerms = gamma_magnitude2 * (np.cos(gamma_phase2) * X2 + np.sin(gamma_phase2) * Y2)

        # Select the interaction term based on index
        if index == 1:
            interactionEnergy = g_eff * exchangeOperator1
        elif index == 2:
            interactionEnergy = g_eff * exchangeOperator2
        elif index == 3:
            interactionEnergy = g_eff * exchangeOperator3
        else:
            interactionEnergy = 0
            if self.verbose:
                print("interaction kind not specified")

        return selfEnergyTerms + interactionEnergy + Qubit1ControlTerms + Qubit2ControlTerms

    def detuning_update(self):
        """Update detuning values by random sampling"""
        self.detuning = [
            random.sample(self.detuning_list[0], k=1)[0],
            random.sample(self.detuning_list[1], k=1)[0]
        ]

    def update_target_unitary(self, U):
        """Update the target unitary and recalculate dependent values"""
        self._U_target = self.unitary_to_superoperator(U)
        self.unitary_U_target = U
        self.initialActions = self.KakActionCalculation()

    def unitary_to_superoperator(self, U):
        """Convert unitary to superoperator"""
        return (spre(Qobj(U)) * spost(Qobj(U.conjugate().transpose()))).data.toarray()

    def get_relaxation_rate(self):
        """Get relaxation rates by random sampling"""
        return [random.sample(rates, k=1)[0] for rates in self.relaxation_rates_list]

    def get_observation(self):
        """Construct the observation vector"""
        normalized_detuning = [
            normalize(self.detuning[0], self.detuning_list[0]),
            normalize(self.detuning[1], self.detuning_list[1])
        ]
        normalized_relaxation_rates = [
            normalize(rate, rates)
            for rate, rates in zip(self.relaxation_rate, self.relaxation_rates_list)
        ]

        return np.append(
            [self.compute_fidelity()] + normalized_relaxation_rates + normalized_detuning,
            self.unitary_to_observation(self.unitary_U_target)
        )

    def compute_fidelity(self):
        """Compute fidelity between current and target unitary"""
        U_target_dagger = self.unitary_to_superoperator(self.unitary_U_target.conjugate().transpose())
        F = float(np.abs(np.trace(U_target_dagger @ self.U))) / (self.U.shape[0])
        return F

    def unitary_to_observation(self, U):
        """Convert unitary to observation vector"""
        return np.array(
            [(abs(x), (cmath.phase(x) / 2 / np.pi + 1) / 2) for x in U.flatten()],
            dtype=np.float64
        ).reshape(-1)

    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        # Reset state variables
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.prev_fidelity = 0
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning_update()
        self.U = self.U_initial.copy()
        self.unitary_U = np.eye(4)

        # Reset arrays
        self.H_arrays = {
            'H1_1': [], 'H2_1': [], 'H1_2': [], 'H2_2': [],
            'H1_3': [], 'H2_3': [], 'H1_4': []
        }
        self.H_tots = {
            'H_tot1_1': [], 'H_tot2_1': [], 'H_tot1_2': [], 'H_tot2_2': [],
            'H_tot1_3': [], 'H_tot2_3': [], 'H_tot1_4': []
        }

        # Calculate KAK decomposition
        self.initialActions = self.KakActionCalculation()

        # Get initial observation
        self.state = self.get_observation()

        return self.state, {}

    def update_total_H(self, H_array, num_time_bins):
        """Update the total Hamiltonian for each time bin"""
        # Initialize with zeros of the right shape - assuming all H_elem have the same shape
        if len(H_array) > 0:
            example_shape = H_array[0].shape
            H_tot = [np.zeros(example_shape) for _ in range(num_time_bins)]
        else:
            # Fallback if H_array is empty
            H_tot = [0] * num_time_bins

        for ii, H_elem in enumerate(H_array):
            Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar)
            for jj in range(num_time_bins):
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))
                if jj < len(H_tot):
                    H_tot[jj] += factor * H_elem

        return H_tot

    def propagate(self, H_tot_list, jump_ops, delta_t, num_time_bins):
        """Propagate the system through time"""
        for jj in range(num_time_bins):
            # Calculate Liouvillian
            L = liouvillian(Qobj(H_tot_list[jj]), jump_ops, data_only=False, chi=None).data.toarray()

            # Calculate propagation operators
            Ut = la.expm(delta_t * L)
            unitary_Ut = la.expm(-1j * H_tot_list[jj] * delta_t)

            # Update total propagation
            self.U = Ut @ self.U
            self.unitary_U = unitary_Ut @ self.unitary_U

    def get_base_idx(self, section_id):
        """Calculate base index for parameters based on section ID"""
        base_idx = 0
        for i in range(1, section_id):
            # Add 6 parameters for single-qubit gates, 1 for two-qubit gates
            if i in [1, 3, 5, 7]:
                base_idx += 6
            elif i in [2, 4, 6]:
                base_idx += 1
        return base_idx

    def process_actions(self, action, section_id):
        """Process actions for a specific gate section"""
        # Calculate base index dynamically
        base_idx = self.get_base_idx(section_id)

        if section_id in [1, 3, 5, 7]:  # Single qubit gates
            # Process based on current Haar number (sign adjustment)
            if self.current_Haar_num == 1:
                # Use initial actions for Haar 1
                alpha1 = self.alpha_max * (action[base_idx] + self.initialActions[base_idx])
                alpha2 = self.alpha_max * (action[base_idx + 1] + self.initialActions[base_idx + 1])
                gamma_magnitude1 = self.gamma_magnitude_max * (action[base_idx + 2] + self.initialActions[base_idx + 2])
                gamma_magnitude2 = self.gamma_magnitude_max * (action[base_idx + 3] + self.initialActions[base_idx + 3])
                gamma_phase1 = self.gamma_phase_max * (action[base_idx + 4] + self.initialActions[base_idx + 4])
                gamma_phase2 = self.gamma_phase_max * (action[base_idx + 5] + self.initialActions[base_idx + 5])
            elif self.current_Haar_num == 2:
                # Negate alpha for Haar 2
                alpha1 = -self.alpha_max * (action[base_idx] + self.initialActions[base_idx])
                alpha2 = -self.alpha_max * (action[base_idx + 1] + self.initialActions[base_idx + 1])
                gamma_magnitude1 = self.gamma_magnitude_max * (action[base_idx + 2] + self.initialActions[base_idx + 2])
                gamma_magnitude2 = self.gamma_magnitude_max * (action[base_idx + 3] + self.initialActions[base_idx + 3])
                gamma_phase1 = self.gamma_phase_max * (action[base_idx + 4] + self.initialActions[base_idx + 4])
                gamma_phase2 = self.gamma_phase_max * (action[base_idx + 5] + self.initialActions[base_idx + 5])
            else:
                # No initial actions for Haar 3+
                alpha1 = self.alpha_max * action[base_idx]
                alpha2 = self.alpha_max * action[base_idx + 1]
                gamma_magnitude1 = self.gamma_magnitude_max * action[base_idx + 2]
                gamma_magnitude2 = self.gamma_magnitude_max * action[base_idx + 3]
                gamma_phase1 = self.gamma_phase_max * action[base_idx + 4]
                gamma_phase2 = self.gamma_phase_max * action[base_idx + 5]

            return alpha1, alpha2, gamma_magnitude1, gamma_magnitude2, gamma_phase1, gamma_phase2

        elif section_id in [2, 4, 6]:  # Two-qubit gates
            # Two-qubit gates only have g_eff parameter
            if self.current_Haar_num == 1:
                g_eff = self.g_eff_max * (action[base_idx] + self.initialActions[base_idx])
            else:
                g_eff = self.g_eff_max * action[base_idx]

            return g_eff

        return None  # Should never reach here

    def step(self, action):
        """Execute one time step within the environment"""
        num_time_bins = 2 ** (self.current_Haar_num - 1)

        # Process all gate sections in sequence
        # 1. Process gates parameters (7 sections)
        params = {}

        # Single qubit gates (4 sections)
        for section in [1, 3, 5, 7]:
            params[f'section_{section}'] = self.process_actions(action, section)

        # Two-qubit gates (3 sections)
        for section in [2, 4, 6]:
            params[f'section_{section}'] = self.process_actions(action, section)

        # Extract parameters for each section
        alpha1_1, alpha2_1, gamma_magnitude1_1, gamma_magnitude2_1, gamma_phase1_1, gamma_phase2_1 = params['section_1']
        g_eff1 = params['section_2']
        alpha1_2, alpha2_2, gamma_magnitude1_2, gamma_magnitude2_2, gamma_phase1_2, gamma_phase2_2 = params['section_3']
        g_eff2 = params['section_4']
        alpha1_3, alpha2_3, gamma_magnitude1_3, gamma_magnitude2_3, gamma_phase1_3, gamma_phase2_3 = params['section_5']
        g_eff3 = params['section_6']
        alpha1_4, alpha2_4, gamma_magnitude1_4, gamma_magnitude2_4, gamma_phase1_4, gamma_phase2_4 = params['section_7']

        # 2. Set up noise operators
        jump_ops = [np.sqrt(rate) * op for rate, op in zip(self.relaxation_rate, self.relaxation_ops)]

        # 3. Calculate Hamiltonians
        hamiltonians = {
            'H1_1': self.hamiltonian(
                self.detuning[0], self.detuning[1], alpha1_1, alpha2_1, 0,
                gamma_magnitude1_1, gamma_phase1_1, gamma_magnitude2_1, gamma_phase2_1
            ),
            'H2_1': self.hamiltonian(
                self.detuning[0], self.detuning[1], 0, 0, g_eff1, 0, 0, 0, 0, index=1
            ),
            'H1_2': self.hamiltonian(
                self.detuning[0], self.detuning[1], alpha1_2, alpha2_2, 0,
                gamma_magnitude1_2, gamma_phase1_2, gamma_magnitude2_2, gamma_phase2_2
            ),
            'H2_2': self.hamiltonian(
                self.detuning[0], self.detuning[1], 0, 0, g_eff2, 0, 0, 0, 0, index=1
            ),
            'H1_3': self.hamiltonian(
                self.detuning[0], self.detuning[1], alpha1_3, alpha2_3, 0,
                gamma_magnitude1_3, gamma_phase1_3, gamma_magnitude2_3, gamma_phase2_3
            ),
            'H2_3': self.hamiltonian(
                self.detuning[0], self.detuning[1], 0, 0, g_eff3, 0, 0, 0, 0, index=1
            ),
            'H1_4': self.hamiltonian(
                self.detuning[0], self.detuning[1], alpha1_4, alpha2_4, 0,
                gamma_magnitude1_4, gamma_phase1_4, gamma_magnitude2_4, gamma_phase2_4
            )
        }

        # 4. Add Hamiltonians to arrays
        for name, H in hamiltonians.items():
            self.H_arrays[name].append(H)

        # 5. Calculate total Hamiltonians for each time bin
        for name in self.H_arrays:
            self.H_tots[f'H_tot{name[1:]}'] = self.update_total_H(self.H_arrays[name], num_time_bins)

        # 6. Reset propagators before applying time evolution
        self.U = np.eye(16)
        self.unitary_U = np.eye(4)

        # 7. Propagate for each H_tot in sequence
        delta_t = self.final_time / num_time_bins
        H_tot_sequence = [
            self.H_tots['H_tot1_1'], self.H_tots['H_tot2_1'],
            self.H_tots['H_tot1_2'], self.H_tots['H_tot2_2'],
            self.H_tots['H_tot1_3'], self.H_tots['H_tot2_3'],
            self.H_tots['H_tot1_4']
        ]

        for H_tot in H_tot_sequence:
            self.propagate(H_tot, jump_ops, delta_t, num_time_bins)

        # 8. Calculate reward and fidelity
        fidelity = self.compute_fidelity()
        reward = (-3 * np.log10(1.0000001 - fidelity) + np.log10(1.0000001 - self.prev_fidelity)) + (
                3 * fidelity - self.prev_fidelity)
        self.prev_fidelity = fidelity

        # 9. Update state
        self.state = self.get_observation()

        # 10. Record history if at the last Haar basis
        if self.current_Haar_num == self.num_Haar_basis:
            self.transition_history.append([fidelity, reward, *action, *self.U.flatten()])

        # 11. Determine if episode is over
        truncated = (fidelity >= 1)
        terminated = (self.current_Haar_num >= self.num_Haar_basis) and (
                self.current_step_per_Haar >= self.steps_per_Haar)

        # 12. Update Haar wavelet counters
        if self.current_step_per_Haar == self.steps_per_Haar:
            self.current_Haar_num += 1
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

        return self.state, reward, terminated, truncated, {}

    def canonicalDecomposition(self):
        """Decompose the target unitary into canonical form using KAK decomposition"""

        # Helper functions for decomposition
        def to_su(u: np.ndarray) -> np.ndarray:
            return u * complex(np.linalg.det(u)) ** (-1 / np.shape(u)[0])

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

        def KAK_2q(U: np.ndarray, rounding: int = 19) -> Tuple:
            # Map U(4) to SU(4) (and phase)
            U = U / np.linalg.det(U) ** 0.25
            assert np.isclose(np.linalg.det(U), 1), "Determinant of U is not 1"

            # Unconjugate U into the magic basis
            B = 1 / np.sqrt(2) * np.array([
                [1., 0, 0, 1.j],
                [0, 1.j, 1., 0],
                [0, 1.j, -1., 0],
                [1., 0, 0, -1.j]
            ])

            U_prime = np.conj(B).T @ U @ B

            # Isolating the maximal torus
            Theta = lambda U: np.conj(U)
            M_squared = Theta(np.conj(U_prime).T) @ U_prime

            if rounding is not None:
                M_squared = np.round(M_squared, rounding)  # For numerical stability

            # Diagonalizing M^2
            D, P = np.linalg.eig(M_squared)

            # Check and correct for det(P) = -1
            if np.isclose(np.linalg.det(P), -1):
                P[:, 0] *= -1  # Multiply the first eigenvector by -1

            # Extracting K2
            K2 = np.conj(P).T
            assert np.allclose(K2 @ K2.T, np.identity(4)), "K2 is not orthogonal"
            assert np.isclose(np.linalg.det(K2), 1), "Determinant of K2 is not 1"

            # Extracting A
            A = np.sqrt(D)
            if np.isclose(np.prod(A), -1):
                A[0] *= -1  # Multiply the first eigenvalue by -1
            A = np.diag(A)  # Turn the list of eigenvalues into a diagonal matrix
            assert np.isclose(np.linalg.det(A), 1), "Determinant of A is not 1"

            # Extracting K1
            K1 = U_prime @ np.conj(K2).T @ np.conj(A).T
            assert np.allclose(K1 @ K1.T, np.identity(4)), "K1 is not orthogonal"
            assert np.isclose(np.linalg.det(K1), 1), "Determinant of K1 is not 1"

            # Extracting Local Gates
            L = B @ K1 @ np.conj(B).T  # Left Local Product
            R = B @ K2 @ np.conj(B).T  # Right Local Product

            phase1, L1, L2 = decompose_one_qubit_product(L)  # L1 (top), L2(bottom)
            phase2, R1, R2 = decompose_one_qubit_product(R)  # R1 (top), R2(bottom)

            # Extracting the Canonical Parameters
            theta_vec = np.angle(np.diag(A))[:3]  # theta vector
            a0, a1, a2 = C_MATRIX_INV @ theta_vec  # Computing the "a"-vector

            # Unpack Parameters
            c0, c1, c2 = 2 * a1, -2 * a0, 2 * a2  # Unpack parameters

            return phase1, L1, L2, phase2, R1, R2, c0, c1, c2

        # Call KAK decomposition on the target unitary
        return KAK_2q(self.unitary_U_target)

    def KakActionCalculation(self):
        """Calculate initial actions based on KAK decomposition"""
        # Get KAK decomposition
        phase1, L1, L2, phase2, R1, R2, c0, c1, c2 = self.canonicalDecomposition()

        # Initialize action array
        initialActions = np.zeros(27)

        # Apply single qubit action calculations
        initialActions[0] = self.singleQubitActionCalculation(R1)[0]
        initialActions[1] = self.singleQubitActionCalculation(R2)[0]
        initialActions[2] = self.singleQubitActionCalculation(R1)[1]
        initialActions[3] = self.singleQubitActionCalculation(R2)[1]
        initialActions[4] = self.singleQubitActionCalculation(R1)[2]
        initialActions[5] = self.singleQubitActionCalculation(R2)[2]

        # First two-qubit interaction
        initialActions[6] = self.canonicalActionCalculation(c0, c1, c2, 1)

        # Second section: Single qubit gates based on Hadamard
        initialActions[7] = self.singleQubitActionCalculation(H)[0]
        initialActions[8] = self.singleQubitActionCalculation(H)[0]
        initialActions[9] = self.singleQubitActionCalculation(H)[1]
        initialActions[10] = self.singleQubitActionCalculation(H)[1]
        initialActions[11] = self.singleQubitActionCalculation(H)[2]
        initialActions[12] = self.singleQubitActionCalculation(H)[2]

        # Second two-qubit interaction
        initialActions[13] = self.canonicalActionCalculation(c0, c1, c2, 2)

        # Third section: Single qubit gates based on H·S·H

        initialActions[14] = self.singleQubitActionCalculation(HSH)[0]
        initialActions[15] = self.singleQubitActionCalculation(HSH)[0]
        initialActions[16] = self.singleQubitActionCalculation(HSH)[1]
        initialActions[17] = self.singleQubitActionCalculation(HSH)[1]
        initialActions[18] = self.singleQubitActionCalculation(HSH)[2]
        initialActions[19] = self.singleQubitActionCalculation(HSH)[2]

        # Third two-qubit interaction
        initialActions[20] = self.canonicalActionCalculation(c0, c1, c2, 3)

        # Fourth section: Single qubit gates based on L1·Sdagger·H and L2·Sdagger·H
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
        """Convert a single-qubit unitary into action parameters using ZXZ decomposition"""
        # Initialize action vector
        singleQubitActions = np.zeros(3)

        # Use ZXZ Euler decomposition

        x_angle, z_angle_after, z_angle_before = OneQubitEulerDecomposer(basis='ZXZ').angles(U)


        # Map Euler angles to gate parameters
        singleQubitActions[0] = np.mod((z_angle_after + z_angle_before) / np.pi, 2)  # alpha
        singleQubitActions[1] = np.mod(x_angle / np.pi, 2)  # gamma_magnitude
        singleQubitActions[2] = np.mod(-z_angle_before / np.pi, 2)  # gamma_phase

        return singleQubitActions

    def Rn(self, theta, axisSelection):
        """Generate rotation matrix around given axis"""
        return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * axisSelection

    def ZXZ_Rotation_Generation(self, angles):
        """Generate unitary from ZXZ Euler angles"""
        return self.unitary_normalization(self.Rn(angles[1], Z) @ self.Rn(angles[0], X) @ self.Rn(angles[2], Z))

    def unitary_normalization(self, unitary_in):
        """Normalize unitary matrix"""
        if unitary_in[0][0] == 0:
            unitary_in = unitary_in / unitary_in[0][1]
        else:
            unitary_in = unitary_in / unitary_in[0][0]

        return unitary_in

    def canonicalActionCalculation(self, c0, c1, c2, index=1):
        """Calculate two-qubit action parameter from canonical KAK parameters"""
        twoQubitAction = 0

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