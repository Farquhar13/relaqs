from relaqs.environments.noisy_two_qubit_analytical_env import AnalyticalNoisyTwoQubitEnv
from relaqs.environments.two_qubit_changing_gate import TwoQubitChangingEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import *
from relaqs.api import gates
from relaqs.api.utils import *
import logging
import warnings
from qutip.operators import *
from qutip.qip.operations import cnot
import datetime
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Box bound precision lowered by casting.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def do_inferencing(env, train_alg, curr_gate):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.env_config
    target_gate = curr_gate.get_matrix()  # Set new target gate for inference

    inference_env_config["U_target_list"] = [curr_gate]
    # inference_env_config['verbose'] = False
    inference_env_config["U_target"] = target_gate
    env_class = type(env)
    inference_env = env_class(inference_env_config)

    # ------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    episode_reward = 0.0
    done = False

    obs, info = inference_env.reset()  # Start with the inference environment
    while not done:

        # Compute an action (`a`).
        action = train_alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward

        if done:
            return inference_env, target_gate, inference_env.transition_history[-1]

def inference_and_save(inference_list, save_dir, train_alg, n_episodes_for_inferencing):
    for curr_gate in inference_list:

        gate_save_dir = os.path.join(save_dir, str(curr_gate))
        plot_filename = f'inference_{curr_gate}.png'
        os.makedirs(gate_save_dir, exist_ok=True)

        figure_title = f"[NOISY 2 Qubit] Inferencing on Multiple Different {str(curr_gate)}."
        transition_history = []

        for inference_iteration in range(n_episodes_for_inferencing):
            # -----------------------> Inferencing <---------------------------
            env = train_alg.workers.local_worker().env
            inference_env, target_gate, history = do_inferencing(env, train_alg, curr_gate)
            transition_history.append(history)

        df = pd.DataFrame(transition_history)
        # df.to_pickle(env_data_title + "env_data.pkl")  # easier to load than csv
        df.to_csv(gate_save_dir +  "env_data.csv", index=False)  # backup in case pickle doesn't work
        multiple_inference_visuals(df, figure_title=figure_title, save_dir=gate_save_dir, plot_filename=plot_filename,
                               gate=curr_gate)

def main():

    base_path = "file_path"
    n_episodes_for_inferencing = 1000

    model = load_model(os.path.join(base_path, "model_checkpoints"))
    inferencing_gate = [gates.Cz(), gates.Cnot(), gates.RandomSUN()]

    print(f'\nStarting inferencing\n')
    inference_start = get_time()
    inference_and_save(inference_list=inferencing_gate, save_dir=base_path, train_alg=model,
                       n_episodes_for_inferencing=n_episodes_for_inferencing)
    inference_end = get_time()
    inference_elapsed_time = inference_end - inference_start

    print(f"Inference Time + Saving Inference + Inference Visuals: {inference_elapsed_time}")
    # print(f'Results saved to: {save_dir}')

if __name__ == '__main__':
    main()