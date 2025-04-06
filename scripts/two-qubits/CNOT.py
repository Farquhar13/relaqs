import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments import NoisyTwoQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs import RESULTS_DIR
from qutip.operators import *
from qutip import cnot
import numpy as np
import datetime

def env_creator(config):
    return NoisyTwoQubitEnv(config)

def run(n_training_iterations=1, save=True, plot=True):
    ray.init()

    register_env("my_env", env_creator)

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    
    env_config = NoisyTwoQubitEnv.get_default_env_config()
    CNOT = cnot().data.toarray()
    env_config["U_target"] = CNOT

    alg_config.environment("my_env", env_config=env_config)

    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = NoisyTwoQubitEnv.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [300] * 5
    alg_config.exploration_config["scale_timesteps"] = 10000
    print(alg_config.algo_class)
    print(alg_config["framework"])

    alg = alg_config.build()
    # ---------------------------------------------------------------------
    list_of_results = []
    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()
        list_of_results.append(result['hist_stats'])
    # -------------------------------------------------------------

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, results=list_of_results, save_path = RESULTS_DIR + "two-qubit gates/"+"CNOT" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/"))
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    n_training_iterations = 1
    save = True
    plot = True
    run(n_training_iterations, save, plot)
