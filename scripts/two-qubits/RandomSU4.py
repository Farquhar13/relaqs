import ray
from relaqs.environments.noisy_two_qubit_env import NoisyTwoQubitEnv
from relaqs.environments.two_qubit_changing_gate import TwoQubitChangingEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import logging
import warnings
from qutip.operators import *
from qutip.qip.operations import cnot
from relaqs.api.utils import *
from relaqs.api import gates
import datetime
import numpy as np
import warnings
import logging

np.seterr(divide='ignore', invalid='ignore')
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Box bound precision lowered by casting.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(env=TwoQubitChangingEnv, n_training_episodes=1, u_target_list = [gates.RandomSUN()], u_initial_list = [gates.RandomSUN()], save=True, plot=True):
    ray.init()

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")

    env_config = env.get_default_env_config()

    env_config["num_Haar_basis"] = 3
    env_config["steps_per_Haar"] = 1

    env_config["U_target_list"] = u_target_list
    env_config["U_initial_list"] = u_initial_list

    # env_config['verbose'] = False

    alg_config.environment(env, env_config=env_config)

    alg_config.rollouts(batch_mode="complete_episodes")
    # ---------------------------------Alg Configs---------------------------------
    # alg_config.actor_lr = 4e-5
    # alg_config.critic_lr = 5e-4

    alg_config.actor_lr = 1e-5
    alg_config.critic_lr = 2e-4
    # alg_config.actor_hiddens = [1200] * 14
    # alg_config.critic_hiddens = [800] * 10
    alg_config.actor_hiddens = [500] * 6
    alg_config.critic_hiddens = [300] * 4

    alg_config.train_batch_size = 512

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 100
    alg_config.exploration_config["scale_timesteps"] = 1000

    # ---------------------------------------------------------------------

    alg = alg_config.build()

    n_training_episodes *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']

    training_start_time = get_time()
    # ---------------------> Train Agent <-------------------------
    for i in range(n_training_episodes):
        alg.train()
        print(
            f'--------------------------Iterations completed: {i + 1}/{n_training_episodes}--------------------------')

    training_end_time = get_time()
    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed Time: {training_elapsed_time}")

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg,
                         save_path=RESULTS_DIR + "two-qubit gates/" + datetime.datetime.now().strftime(
                             "%Y-%m-%d_%H-%M-%S/"))
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        env_string = f"[Noisy 2 Qubit] "
        initial_gate_title = " ".join(f"{target_gate}-" for target_gate in env_config["U_initial_list"])
        target_gate_title = " ".join(f"{target_gate}-" for target_gate in env_config["U_target_list"])
        plot_data(save_dir = save_dir, figure_title=env_string + "Initial Gates: "+ initial_gate_title + f"Target Gates: " + target_gate_title, plot_filename='Training')
        print("Plots Created")
    # --------------------------------------------------------------


def main():
    env = TwoQubitChangingEnv
    n_training_episodes = 40

    u_initial_list = [gates.I(4)]
    u_target_list = [gates.RandomSUN()]

    # n_training_episodes = 1
    save = True
    plot = True
    run(env, n_training_episodes=n_training_episodes, u_target_list = u_target_list, u_initial_list=u_initial_list,save=save, plot=plot)


if __name__ == "__main__":
    main()

