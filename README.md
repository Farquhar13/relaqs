
# Relaqs
Relqs is a Python framework that uses reinforcement learning to optimize pulse control to implement high-fidelity quantum gates in noiseless and noisy environments.

A key feature is the option to train an agent that generalizes over chaning noise parameters and/or target gates.

For our noise model we use relaxation rates (T1, T2), and detuning which are resampled at the start of each episode. The noise parameters are provided to the `liouvillian` function from QuTip to simulate the noisy dymanics of Hamiltonian evolution.

<img width="1423" height="661" alt="image" src="https://github.com/user-attachments/assets/acada4d6-582f-4375-9403-e8a4c6715ed1" />

## Scripts
The `scripts` folder contains example files to train an RL agent in our environments. Consider them templates that you can run and modify.

Some examples:

* `single_gate.py` learn optimal pulses for a single-qubit gate
* `changing_gate.py` learn a random SU(2), with the target gate changing on each episode
* `two-qubts/CZ.py` learn the two-qubit CZ gate
* `two-qubits/RandomSU4.py` learn a random SU(4), with the target gate changing on each episode

## Environments
`src/relaqs/environments`

Single-qubit gates:
* `single_qubit_env.py` base noiseless environment for learning a fixed target gate
* `noisy_single_qubit_env.py` noisy environment for learning fixed target gate
* `changing_target_gate.py` contains classes for noiseless or noisy evolution with a changing target gate on each episode

Two-qubit gates:
* `noisy_two_qubit.py` noisy environment to learn a fixed 2-qubit target gate
* `two_qubit_changing_gate.py` noisy environment to learn changing 2-qubit target gates
* `noisy_two_qubit_analytical_env.py` noisy environment used to test the analytic solution (not RL) used for a baseline


## Install relaqs
```
pip install -e .
```

### Required Packages:
The require packages should be installed automatically by pip.  Legacy install instructions are below:





One may install required packages in a number of ways. Either by manually installing, or by executing one of the commands below:

### Direct install:
        pip install -U "ray[rllib]"
        pip install torch
        conda install tensorboardX
        pip install tensorflow-probability
        pip install qutip

### Install with pip:
`pip install -r requirements.txt`

### Install in a conda environment (verified on M1 Mac):
If you are using a conda environment running `conda env create -n <ENVNAME> --file requirements.yml` will install packages using a combination of `pip` and `conda`. This is useful on M1 Macs where some packages aren't available through `pip`.
