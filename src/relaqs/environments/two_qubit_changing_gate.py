import random
from relaqs.environments import NoisyTwoQubitEnv
from relaqs.api.gates import *
from relaqs.api.utils import normalize

class TwoQubitChangingEnv(NoisyTwoQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        config_dict = super().get_default_env_config()
        config_dict["U_target_list"] = []
        config_dict["U_initial_list"] = []
        config_dict["target_generation_function"] = RandomSUN
        config_dict["initial_generation_function"] = RandomSUN
        config_dict["observation_space_size"] = 512 + 2 + 4 + 512 + 512
        return config_dict

    def __init__(self, env_config):
        super().__init__(env_config)
        self.U_target_list = env_config["U_target_list"]
        self.U_initial_list = env_config["U_initial_list"]
        self.target_generation_function = env_config["target_generation_function"]
        self.initial_generation_function = env_config["initial_generation_function"]

    def set_target_gate(self):
        if len(self.U_target_list) == 0:
            U = self.target_generation_function().get_matrix()
        else:
            U = random.choice(self.U_target_list).get_matrix()
        self.U_target = self.unitary_to_superoperator(U)
        self.U_target_dm = U.copy()

    def set_initial_gate(self):
        if len(self.U_initial_list) == 0:
            U = self.initial_generation_function().get_matrix()
        else:
            U = random.choice(self.U_initial_list).get_matrix()
        self.U_initial = self.unitary_to_superoperator(U)
        self.U_initial_dm = U.copy()

    def reset(self, *, seed=None, options=None):
        _, info = super().reset()
        self.set_target_gate()
        self.set_initial_gate()
        self.U = self.U_initial.copy()
        starting_observation = self.get_observation()
        return starting_observation, info

    def get_observation(self):
        net_target_gate = self.U_target @ self.U_initial.conjugate().transpose()
        U_diff_action_target = self.unitary_to_observation(net_target_gate @ self.U.conjugate().transpose())

        normalized_detuning = np.array([normalize(self.detuning[0], self.detuning_list[0]),
                               normalize(self.detuning[1], self.detuning_list[1])])
        normalized_relaxation_rates = np.array([normalize(self.relaxation_rate[0], self.relaxation_rates_list[0]),
                                       normalize(self.relaxation_rate[1], self.relaxation_rates_list[1]),
                                       normalize(self.relaxation_rate[2], self.relaxation_rates_list[2]),
                                       normalize(self.relaxation_rate[3], self.relaxation_rates_list[3])])
        return np.concatenate([normalized_detuning, normalized_relaxation_rates,
                        self.unitary_to_observation(net_target_gate), self.unitary_to_observation(self.U),
                               U_diff_action_target], axis=0)

    def return_env_config(self):
        env_config = super().get_default_env_config()
        env_config.update({
            "num_Haar_basis": self.num_Haar_basis,
            "steps_per_Haar": self.steps_per_Haar,
            "verbose": self.verbose,
            "U_init": self.U_initial,
            "U_target": self.U_target,
            "target_generation_function": self.target_generation_function,
            "initial_generation_function": self.initial_generation_function,
            "U_target_list": self.U_target_list,
            "U_initial_list": self.U_initial_list,
            "detuning_list": self.detuning_list,  # qubit detuning
            "relaxation_rates_list": self.relaxation_rates_list,
            "relaxation_ops": self.relaxation_ops,
        })
        return env_config


