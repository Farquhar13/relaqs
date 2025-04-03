# import sys
# sys.path.append('./src/')

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.noisy_two_qubit_env import NoisyTwoQubitEnv
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

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import datetime

logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Box bound precision lowered by casting.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# def env_creator(config):
#     return NoisyTwoQubitEnv(config)
#
# def save_grad_to_file(resultdict):
#     try:
#         policydict = resultdict["default_policy"]
#         stats = policydict["learner_stats"]
#         grad_gnorm = stats["grad_gnorm"]
#         with open("gradfile", "a") as f:
#             f.write(f"{grad_gnorm}\n")
#     except KeyError:
#         print(f"Failed to extract grad_gnorm from: {resultdict}")
#
# def inject_logging(alg, logging_func):
#     og_ts = alg.training_step
#     def new_training_step():
#         result = og_ts()
#         # do logging here
#         logging_func(result)
#         return result
#     alg.training_step = new_training_step

def run(n_training_episodes=1, save=True, plot=True):
    ray.init(num_cpus=14)


    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
        
    env_config = NoisyTwoQubitEnv.get_default_env_config()
    CNOT = cnot().data.toarray()
    env_config["U_target"] = CNOT

    alg_config.environment(NoisyTwoQubitEnv, env_config=env_config)
    
    alg_config.rollouts(batch_mode="complete_episodes")

        ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [30,30,30]
    alg_config.exploration_config["scale_timesteps"] = 10000
    alg_config.train_batch_size = 1
        # ---------------------------------Alg Configs---------------------------------
    # alg_config.actor_lr = 4e-5
    # alg_config.critic_lr = 5e-4
    #
    # alg_config.actor_hidden_activation = "relu"
    # alg_config.critic_hidden_activation = "relu"
    # alg_config.num_steps_sampled_before_learning_starts = 100
    # alg_config.actor_hiddens = [300, 300, 300, 300, 300]
    # alg_config.exploration_config["scale_timesteps"] = 1000
    # alg_config.train_batch_size = 512
    # ---------------------------------------------------------------------


    alg = alg_config.build()

    # n_training_episodes *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']

    update_every_percent = 2
    update_interval = max(1, int(n_training_episodes * (update_every_percent / 100)))
    # list_of_results = []

    training_start_time = get_time()
    # ---------------------> Train Agent <-------------------------
    for i in range(n_training_episodes):
        alg.train()
        # list_of_results.append(result['hist_stats'])
        # Print update every x%
        if (i + 1) % int(update_interval) == 0 or (i + 1) == n_training_episodes:
            percent_complete = (i + 1) / n_training_episodes * 100
            print(f"Training Progress: {percent_complete:.0f}% complete")

    training_end_time = get_time()
    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed Time: {training_elapsed_time}")

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, save_path = RESULTS_DIR + "two-qubit gates/"+"CNOT" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/"))
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
        # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir)
        print("Plots Created")
        # --------------------------------------------------------------


def main():
    # n_training_episodes = 20
    n_training_episodes = 1
    save = True
    plot = True
    run(n_training_episodes, save, plot)

if __name__ == "__main__":
    main()
