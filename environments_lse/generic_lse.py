# This code is part of the work "Warm-Start Variational Quantum Policy Iteration", N. Meyer et al. (2024).
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
from environments_lse.base_environment_lse import BaseEnvironmentLSE


class GenericLSE(BaseEnvironmentLSE):

    def __init__(self, num_qubits: int = 4, perturbation: float = 0.0):
        super().__init__()
        self._num_qubits = num_qubits
        self._perturbation = perturbation

        self._num_states = 2 ** num_qubits

        self._dynamics_policy_matrix = self._construct_dynamics_policy_matrix()

    def lse(self, policy: np.ndarray = None, gamma: float = 1.0
            ) -> tuple[np.ndarray, None, None, np.ndarray]:
        """
        Set up LSE (1 - gamma * P @ Pi) Q_pi = R for given discount factor (R only dummy, no explicit Pi).

        :param policy: Deterministic policy (action index for each state), randomly selected if None is provided
        :param gamma: Discount factor for underlying MPP, must be in (0.0, 1.0]
        :return: System matrix A_pi = I - gamma * P @ Pi, None,
                 None, raw dynamics-policy matrix P * Pi
        """
        assert policy is None, 'Policy is inherently constructed random!'
        system_matrix = np.eye(self._num_states) - gamma * self._dynamics_policy_matrix
        return system_matrix, None, None, self._dynamics_policy_matrix

    @property
    def non_terminal_states(self) -> list[int]:
        return list(range(self._num_states))

    @property
    def terminal_states(self) -> list[int]:
        return []

    def print_policy(self, policy: np.ndarray):
        raise NotImplementedError

    @property
    def env_map(self):
        return None

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_actions(self) -> int:
        return 1

    def _construct_dynamics_policy_matrix(self) -> np.ndarray:
        """
        Sets up the dynamics-policy matrix for a deterministic / stochastic random environment.

        :return: |S| x |S| dimensional dynamics matrix P @ Pi
        """
        # for each state, select a random consecutive state, make sure each state is transitioned to at most log(N) times
        next_states = np.random.choice(np.repeat(np.arange(self.num_states), self._num_qubits), size=self.num_states, replace=False)
        dynamics_matrix = np.zeros((self._num_states, self._num_states))
        transition_probabilities = np.array([1.0])  # 1.0 for deterministic case
        if 0.0 < self._perturbation:
            # local stochasticity in logarithmic neighborhood
            logarithmic_neighborhood = self._num_qubits // 2
            neighborhood = np.arange(-logarithmic_neighborhood, logarithmic_neighborhood+1)
            transition_probabilities = self._exponential_transition_probabilities(np.abs(neighborhood / logarithmic_neighborhood))

        for state in range(self._num_states):
            # iterate over possible next states (only one for deterministic dynamics, log(N) many in stochastic case)
            for transition_probability, next_state \
                    in zip(transition_probabilities,
                           range(next_states[state], next_states[state] + len(transition_probabilities) + 1)):
                # take care that for stochastic dynamics, one might need to `wrap the edges`
                dynamics_matrix[state, next_state % self._num_states] = transition_probability
        assert np.all(np.isclose(1.0, np.sum(dynamics_matrix, axis=1))), 'Something went wrong, dynamics matrix is not row-stochastic!'
        return dynamics_matrix

    def _exponential_transition_probabilities(self, distances: np.ndarray):
        """
        Compute exponentially declining transition probabilities within a given neighborhood.
        The permutation parameter `self._perturbation` defines the strength of decline (0.0 deterministic, 1.0 uniform).

        :param distances: Normalized distances.
        :return: Normalized transition probabilities
        """
        assert np.all(0 <= distances) and np.all(distances <= 1), 'All distances have to be normalized!'
        assert 0 < self._perturbation, 'Can only compute dynamics for perturbation > 0!'
        transition_probabilities = np.exp((1 - (1 / self._perturbation)) * distances)
        return transition_probabilities / np.sum(transition_probabilities)


if __name__ == '__main__':
    env_ = GenericLSE(num_qubits=4, perturbation=0.0)
    A_pi_, _, _, _ = env_.lse()
