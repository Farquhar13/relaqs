
import ray
import numpy as np
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments.changing_target_gate import ChangingTargetEnv, NoisyChangingTargetEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs.api import gates
from relaqs.api.gates import Gate

class XY_combination(Gate):
    def __str__(self):
        return "XY_combination"
    
    @classmethod
    def get_matrix(self):
        """
        aX + bY, a^2 + b^2 = 1
        """
        theta = np.random.uniform(0, 2*np.pi)
        a = np.sin(theta)
        b = np.cos(theta)
        return a*gates.X().get_matrix() + b*gates.Y().get_matrix()

def run(env=ChangingTargetEnv, n_training_iterations=1, save=True, plot=True):
    ray.init()

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = env.get_default_env_config()
    #env_config["target_generation_function"] = XY_combination
    #env_config["U_target_list"] = [gates.X().get_matrix(), gates.Y().get_matrix()]
    alg_config.environment(env, env_config=env_config)

    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 100
    alg_config.actor_hiddens = [300,300,300, 300, 300]
    alg_config.exploration_config["scale_timesteps"] = 1000
    alg_config.train_batch_size = 256

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    # ---------------------> Train Agent <-------------------------
    results = [alg.train() for _ in range(n_training_iterations)]
    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, target_gate_string="arbitrary-gate")
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        env_string = "Noisy" if isinstance(env, NoisyChangingTargetEnv) else "Noiseless"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length, figure_title=env_string +" Random SU(2), bigger network, batch=256 ")
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    env = NoisyChangingTargetEnv
    n_training_iterations = 50
    save = True
    plot = True
    run(env, n_training_iterations, save, plot)
