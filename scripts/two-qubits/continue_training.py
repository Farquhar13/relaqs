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

def get_subpath_after_results(path):
    marker = 'results/'
    idx = path.find(marker)
    if idx == -1:
        return None
    return path[idx + len(marker):]

def continue_training(training_dir, n_training_episodes=1, save=True, plot=True):

    # ---------------------> Configure algorithm and Environment <-------------------------
    model = load_model(os.path.join(training_dir, "model_checkpoints"))
    model_config = model.config.copy()
    env_config = model_config['env_config']


    n_training_episodes *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']

    training_start_time = get_time()
    # ---------------------> Train Agent <-------------------------
    try:
        for i in range(n_training_episodes):
            result = model.train()
            print(f"---- Iterations completed: {i + 1}/{n_training_episodes} ----")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        interrupted = True  # <-- FLAG
    else:
        interrupted = False  # <-- FLAG
    finally:
        # Always free Ray resources
        model.stop()  # <-- flush workers, save checkpoints correctly
        ray.shutdown()  # <-- terminates all worker processes
    training_end_time = get_time()
    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed Time: {training_elapsed_time}")

    print(f'U_Initial:\n{env_config["U_initial"]}\n\nU_Target:\n{env_config["U_target"]}')

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = model.workers.local_worker().env
        sr = SaveResults(env, model,
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
        env_string = f"Training Continued From {get_subpath_after_results(training_dir)}"
        plot_data(save_dir=save_dir,
                  figure_title=env_string,
                  plot_filename='Training')
        print("Plots Created")
        # --------------------------------------------------------------


def main():
    training_dir = "file_path"
    n_training_episodes = 60
    save = True
    plot = True
    continue_training(training_dir = training_dir, n_training_episodes = n_training_episodes, save = save,
                      plot = plot)


if __name__ == "__main__":
    main()

