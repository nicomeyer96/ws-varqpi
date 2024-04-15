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


# Some hardcoded metadata for the FrozenLake environment
NUM_ACTIONS = 4
LEFT, DOWN, RIGHT, UP = [0, -1], [1, 0], [0, 1], [-1, 0]
ACTIONS = [LEFT, DOWN, RIGHT, UP]
SYMBOLS = ['<', 'v', '>', '^']
MAPS = {'4x4': ["SFFF",
                "FHFH",
                "FFFH",
                "HFFG"],
        '8x8': ["SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG"]
        }
REWARDS = {
    'S': 0.0, 'F': 0.0, 'H': 0.0,  # normal state, or termination by tripping into hole
    'G': 1.0  # goal state
    }


class FrozenLakeLSE(BaseEnvironmentLSE):
    """
    Class for constructing an LSE formulation of the (potentially stochastic) FrozenLake environment.
    (Inspired by https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
    """

    def __init__(self, map_name: str = '4x4', perturbation: float = 0.0):
        """
        Initialize to specific grid size and stochasticity.

        :param map_name: Instance of environment to use.
        :param perturbation: Probability of `slipping` to perpendicular states, must be in [0, 1/3]
        """
        super().__init__()

        self._map = MAPS.get(map_name, None)
        if self._map is None:
            raise NotImplementedError(f'FrozenLake map {map_name} could not be found.')
        self._num_rows, self._num_cols = len(self._map), len(self._map[0])
        self._num_states = self._num_rows * self._num_cols

        # check for perturbation strength (the bound 1/3 should prevent transitioning to the intended state with lower
        # probability than to the perpendicular ones)
        assert 0 <= perturbation <= 1/3, 'Perturbation must be between 0 and 1/3!'
        self._perturbation = perturbation

        # construct dynamics matrix and (normalized) reward vector
        self._dynamics_matrix, self._reward_vector = self._construct_dynamics_matrix_and_reward_vector()

    def lse(self, policy: np.ndarray = None, gamma: float = 1.0
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Set up LSE (1 - gamma * P @ Pi) Q_pi = R for given deterministic policy and discount factor.

        :param policy: Deterministic policy (action index for each state), randomly selected if None is provided
        :param gamma: Discount factor for underlying MPP, must be in (0.0, 1.0]
        :return: System matrix A_pi = I - gamma * P @ Pi, right side R = [sum_{s'} p(s'|s,a) pi(a|s)]_{s,a},
                 underlying deterministic policy pi, raw dynamics-policy matrix P * Pi
        """
        # select random deterministic policy, in None is provided
        if policy is None:
            policy = np.random.randint(low=0, high=NUM_ACTIONS, size=self._num_states)
        assert self.num_states == policy.shape[0], 'Provided policy has wrong shape.'
        assert np.all(policy < NUM_ACTIONS), 'Policy selects actions not possible in the environment.'
        # construct |S| x |S||A| dimensional policy matrix
        policy_matrix = np.zeros((self.num_states, self._num_states * NUM_ACTIONS))
        for row in range(self._num_rows):  # iterate over environment rows
            for col in range(self._num_cols):  # iterate over environment columns
                pos = Position(row, col, self._map)
                # assign according to greedy action index from policy
                policy_matrix[pos.index, pos.index * NUM_ACTIONS + policy[pos.index]] = 1.0
        assert np.all(np.isclose(1.0, np.sum(policy_matrix, axis=1))), 'Something went wrong, policy matrix is not row-stochastic!'
        system_matrix = np.eye(self._num_states * NUM_ACTIONS) - gamma * self._dynamics_matrix @ policy_matrix
        return system_matrix, self._reward_vector, policy, self._dynamics_matrix @ policy_matrix

    @property
    def non_terminal_states(self) -> list[int]:
        """
        List of non-terminal states in environment

        :return: List of indices of non-terminal states
        """
        non_terminal_state_indices = []
        for row in range(self._num_rows):
            for col in range(self._num_cols):
                pos = Position(row, col, self._map)
                if pos.type in ['S', 'F']:
                    non_terminal_state_indices.append(pos.index)
        return non_terminal_state_indices

    @property
    def terminal_states(self) -> list[int]:
        """
        List of terminal states in environment

        :return: List of indices of terminal states
        """
        terminal_state_indices = []
        for row in range(self._num_rows):
            for col in range(self._num_cols):
                pos = Position(row, col, self._map)
                if pos.type in ['H', 'G']:
                    terminal_state_indices.append(pos.index)
        return terminal_state_indices

    def print_policy(self, policy: np.ndarray) -> None:
        """
        Bring the policy in a nice format and print it.

        :param policy: The policy indicating the optimal action for each state
        """
        for row_index, row in enumerate(self._map):
            for col_index, elem_type in enumerate(row):
                if 'H' == elem_type:
                    print(f'|x', end='')
                elif 'G' == elem_type:
                    print(f'|o', end='')
                else:
                    print(f'|{SYMBOLS[policy[row_index * self._num_rows + col_index]]}', end='')
            print('|')

    @property
    def env_map(self):
        return self._map

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def _construct_dynamics_matrix_and_reward_vector(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sets up the dynamics for a deterministic / stochastic FrozenLake environment.

        :return: |S||A| x |S| dimensional dynamics matrix P, |S||A| dimensional reward vector R
        """
        dynamics_matrix = np.zeros((self._num_states * NUM_ACTIONS, self._num_states))
        reward_vector = np.zeros((self._num_states * NUM_ACTIONS, ))
        for row in range(self._num_rows):  # iterate over environment rows
            for col in range(self._num_cols):  # iterate over environment columns
                pos = Position(row, col, self._map)
                # terminal states
                for action_index, action in enumerate(ACTIONS):  # iterate over four possible actions
                    # terminal state -> stay in state for each action, reward stays as 0
                    if pos.type in ['H', 'G']:
                        dynamics_matrix[pos.index * NUM_ACTIONS + action_index, pos.index] = 1.0
                        continue
                    # non-terminal state -> evaluate new state
                    next_pos = Position(row + action[0], col + action[1], self._map)
                    # assign corresponding transition probability, 1.0 for deterministic dynamics
                    dynamics_matrix[pos.index * NUM_ACTIONS + action_index, next_pos.index] += 1 - 2 * self._perturbation
                    # assign reward value, depending on type of next state
                    reward_vector[pos.index * NUM_ACTIONS + action_index] += (1 - 2 * self._perturbation) * REWARDS[next_pos.type]
                    if self._perturbation > 0.0:  # stochastic environment dynamics
                        # might transition to perpendicular states (can be original state if at edge of environment)
                        next_pos_per_0 = Position(row + action[1], col + action[0], self._map)
                        next_pos_per_1 = Position(row - action[1], col - action[0], self._map)
                        # add corresponding transition probabilities
                        dynamics_matrix[pos.index * NUM_ACTIONS + action_index, next_pos_per_0.index] += self._perturbation
                        dynamics_matrix[pos.index * NUM_ACTIONS + action_index, next_pos_per_1.index] += self._perturbation
                        # add corresponding rewards
                        reward_vector[pos.index * NUM_ACTIONS + action_index] += self._perturbation * REWARDS[next_pos_per_0.type]
                        reward_vector[pos.index * NUM_ACTIONS + action_index] += self._perturbation * REWARDS[next_pos_per_1.type]
        assert np.all(np.isclose(1.0, np.sum(dynamics_matrix, axis=1))), 'Something went wrong, dynamics matrix is not row-stochastic!'
        assert min(reward_vector) >= 0, 'Something went wrong, reward vector contains negative elements!'
        # normalize reward vector to length 1 to allow for encoding as quantum state
        reward_vector /= np.sqrt(np.sum(np.square(reward_vector)))
        assert 1.0 - np.sum(np.square(reward_vector)) < 1e-6, 'Something went wrong, reward vector was not sufficiently normalized'
        return dynamics_matrix, reward_vector


class Position:
    """ Simple data holder for two-dimensional position vector """
    def __init__(self, row: int, col: int, env_map: list[str]):
        """
        Initialize two-dimensional position vector, make sure we stay in environment by mapping to the closest state.

        :param row: Row index
        :param col: Column index
        :param env_map: Layout of environment
        """
        self._num_rows, self._num_cols = len(env_map), len(env_map[0])
        self._row = min(self._num_rows - 1, max(0, row))
        self._col = min(self._num_cols - 1, max(0, col))
        self._map = env_map

    @property
    def index(self) -> int:
        """ Return flattened state index """
        return self._row * self._num_cols + self._col

    @property
    def type(self) -> str:
        """ Return state type at given position """
        return self._map[self._row][self._col]


if __name__ == '__main__':
    env_ = FrozenLakeLSE()
    A_pi_, R_, _, _ = env_.lse()
