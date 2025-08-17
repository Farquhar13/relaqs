# import sys
# sys.path.append('./src/')

import ray
# from ray.rllib.algorithms.ddpg import DDPGConfig
# from ray.tune.registry import register_env
from relaqs.environments.noisy_two_qubit_analytical_env import AnalyticalNoisyTwoQubitEnv
from relaqs.environments.noisy_two_qubit import NoisyTwoQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import logging
import warnings

from relaqs.quantum_noise_data.get_data import get_month_of_single_qubit_data, get_month_of_all_qubit_data
from relaqs import quantum_noise_data
from relaqs import QUANTUM_NOISE_DATA_DIR
from relaqs import RESULTS_DIR

from qutip.operators import *
from qutip.qip.operations import cnot
from relaqs.api.utils import *
from relaqs.api import gates
from relaqs.api.utils import *
import os

import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import datetime

logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Box bound precision lowered by casting.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ibm_nairobi data chosen because sample_noise_parameter function defaults to ibm_nairobi machine + data for qubit label 1 unless otherwise specified
t1_t2_noise_file = "april/ibm_nairobi_month_is_4.json"

# detuning data
detuning_noise_file = "qubit_detuning_data.json"


def run(env=NoisyTwoQubitEnv, n_training_episodes=1, U_initial = None, U_target = None, save=True, plot=True):

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")

    env_config = env.get_default_env_config()
    env_config["U_target"] = U_target.get_matrix()
    env_config["U_initial"] = U_initial.get_matrix()
    env_config["num_Haar_basis"] = 3
    env_config["steps_per_Haar"] = 1
    # env_config['verbose'] = False

    # ---------------------> Get quantum noise data <-------------------------
    # t1_list1, t2_list2, detuning_list1 = sample_noise_parameters(t1_t2_noise_file, detuning_noise_file, qubit_label="1")
    # t1_list2, t2_list2, detuning_list2 = sample_noise_parameters(t1_t2_noise_file, detuning_noise_file, qubit_label="2")
    # # env_config["relaxation_rates_list"] = [t1_list1, t2_list2, t1_list2, t2_list2]  # using real T1 data
    # env_config["detuning_list"] = [detuning_list1, detuning_list2]

    alg_config.environment(env, env_config=env_config)

    alg_config.rollouts(batch_mode="complete_episodes")

    # ---------------------------------Alg Configs---------------------------------

    if isinstance(env, AnalyticalNoisyTwoQubitEnv):
        # Analytical
        alg_config.actor_hiddens = [1] * 1  # ~3.5 M params
        alg_config.critic_hiddens = [1] * 1
        alg_config.train_batch_size = 1

    else:
        alg_config.actor_hiddens = [800] * 6  # ~3.5 M params
        alg_config.critic_hiddens = [800] * 6
        alg_config.train_batch_size = 1024

    alg_config.actor_lr = 5e-5
    alg_config.critic_lr = 1e-4

    # TD3 stabilisers
    alg_config.twin_q = True
    alg_config.policy_delay = 3  # update actor every 3 critic steps
    alg_config.smooth_target_policy = True
    alg_config.target_noise = 0.2
    alg_config.target_noise_clip = 0.5

    alg_config.use_state_preprocessor = True
    alg_config.grad_clip = 1.0

    # Replay
    alg_config.num_steps_sampled_before_learning_starts = 5_000
    alg_config.replay_buffer_config["capacity"] = 500_000

    # Exploration
    alg_config.exploration_config = {
        "type": "OrnsteinUhlenbeckNoise",
        "ou_base_scale": 0.2,
        "ou_theta": 0.15,
        "ou_sigma": 0.2,
        "initial_scale": 1.0,
        "final_scale": 0.1,
        "scale_timesteps": 40_000,
    }

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"

    # ---------------------------------------------------------------------

    alg = alg_config.build()

    n_training_episodes *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']

    training_start_time = get_time()
    # ---------------------> Train Agent <-------------------------
    try:
        for i in range(n_training_episodes):
            result = alg.train()
            print(f"---- Iterations completed: {i + 1}/{n_training_episodes} ----")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        interrupted = True  # <-- FLAG
    else:
        interrupted = False  # <-- FLAG
    finally:
        # Always free Ray resources
        alg.stop()  # <-- flush workers, save checkpoints correctly
        ray.shutdown()  # <-- terminates all worker processes
    training_end_time = get_time()
    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed Time: {training_elapsed_time}")

    print(f'U_Initial:\n{env_config["U_initial"]}\n\nU_Target:\n{env_config["U_target"]}')


    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg,
                         save_path=RESULTS_DIR + "two-qubit gates/" + datetime.datetime.now().strftime(
                             "%Y-%m-%d_%H-%M-%S/"))
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
        # --------------------------------------------------------------

    # Save U_initial and U_target to the same results directory
    np.savez(os.path.join(save_dir, "gates.npz"),
                U_initial=env_config['U_initial'],
                 U_target=env_config['U_target'])

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        env_string = "{SAME GATE EVERY EPISODE} "
        initial_gate_title = f'{U_initial}'
        target_gate_title = f'{U_target}'
        plot_data(save_dir=save_dir,
                  figure_title=env_string + "Initial Gates: " + initial_gate_title + f"Target Gates: " + target_gate_title,
                  plot_filename='Training')
        print("Plots Created")
        # --------------------------------------------------------------


def main():

    #Analytical Setup
    # env = AnalyticalNoisyTwoQubitEnv
    #If you want to run the analytical baseline on the EXACT same SU(4) as a previously trained agent
    load_previous_gates = False
    path = 'file_path'

    env = NoisyTwoQubitEnv
    n_training_episodes = 60

    U_initial = gates.II()
    U_target = gates.Cnot()

    if isinstance(env, AnalyticalNoisyTwoQubitEnv):
        n_training_episodes = 10

    if load_previous_gates:
        data = np.load(path)
        U_initial = gates.temp(gate_array=data['U_initial'])
        U_target = gates.temp(gate_array=data['U_target'])

    save = True
    plot = True

    run(env, n_training_episodes, U_initial=U_initial, U_target=U_target, save=save, plot=plot)


if __name__ == "__main__":
    main()

