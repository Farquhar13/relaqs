# import sys
# sys.path.append('./src/')

import ray
# from ray.rllib.algorithms.ddpg import DDPGConfig
# from ray.tune.registry import register_env
from relaqs.environments.noisy_two_qubit_analytical_env import AnalyticalNoisyTwoQubitEnv
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

# ibm_nairobi data chosen because sample_noise_parameter function defaults to ibm_nairobi machine + data for qubit label 1 unless otherwise specified
t1_t2_noise_file = "april/ibm_nairobi_month_is_4.json"

# detuning data
detuning_noise_file = "qubit_detuning_data.json"

def run(env = AnalyticalNoisyTwoQubitEnv, n_training_episodes=1, save=True, plot=True):
    ray.init(num_cpus=14)

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
        
    env_config = env.get_default_env_config()
    CNOT = cnot().data.toarray()
    env_config["U_target"] = CNOT
    env_config["num_Haar_basis"] = 3
    env_config["steps_per_Haar"] = 1
    # env_config['verbose'] = False

    # ---------------------> Get quantum noise data <-------------------------
    t1_list1, t2_list2, detuning_list1 = sample_noise_parameters(t1_t2_noise_file, detuning_noise_file, qubit_label= "1")
    t1_list2, t2_list2, detuning_list2 = sample_noise_parameters(t1_t2_noise_file, detuning_noise_file, qubit_label="2")
    # env_config["relaxation_rates_list"] = [t1_list1, t2_list2, t1_list2, t2_list2]  # using real T1 data
    # env_config["detuning_list"] = [detuning_list1, detuning_list2]


    alg_config.environment(env, env_config=env_config)
    
    alg_config.rollouts(batch_mode="complete_episodes")

    # ---------------------------------Alg Configs---------------------------------

    alg_config.actor_hiddens = [768] * 6  # ~3.5 M params
    alg_config.critic_hiddens = [768] * 6

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

    # Replay & batching
    alg_config.train_batch_size = 1024
    alg_config.num_steps_sampled_before_learning_starts = 10_000
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


    alg_config.lr_schedule = [[0, 8e-5],
                              [50_000, 4e-5],
                              [100_000, 2e-5],]
    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    # Rollouts (CPU parallelism)
    # # alg_config.rollouts(
    # #     num_rollout_workers=11,
    # #     num_envs_per_worker=1,
    # #     rollout_fragment_length=200,
    # # )
    #
# -------------------------------------------

    # alg_config.train_batch_size = 1
    # alg_config.actor_hiddens = [1] * 1
    # alg_config.critic_hiddens = [1] * 1
    # alg_config.train_batch_size = 1

    # alg_config.actor_hidden_activation = "relu"
    # alg_config.critic_hidden_activation = "relu"
    # alg_config.num_steps_sampled_before_learning_starts = 1024
    # alg_config.exploration_config["scale_timesteps"] = 4096
    # alg_config.num_steps_sampled_before_learning_starts = 100
    # alg_config.exploration_config["scale_timesteps"] = 1_000

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
    # env = NoisyTwoQubitEnv
    env = ScratchNoisyTwoQubitEnv
    # env = BaselineEnv
    # env = ScratchNoisyTwoQubitEnv2
    # env = ExpNoisyTwoQubitEnv
    n_training_episodes = 40
    # n_training_episodes = 1
    save = True
    plot = True
    run(env, n_training_episodes, save, plot)

if __name__ == "__main__":
    main()

