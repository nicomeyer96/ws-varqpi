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


import argparse
import numpy as np
import pennylane as qml
import torch
import scipy
import warnings


def circuit_u3(weights: torch.tensor) -> None:
    """
    Variational circuit with U3 variational gates and nearest-neighbor CNOT entanglement

    :param weights: Weights of the ansatz, have to be of shape [depth, num_qubits, 3]
    """
    assert 3 == len(weights.shape) and 3 == weights.shape[2]
    # extract circuit size and depth from shape of provided weights
    depth, num_qubits = weights.shape[0], weights.shape[1]

    # initial layer
    for qubit in range(num_qubits):
        qml.U3(*weights[0, qubit], wires=qubit)

    # consecutive layers
    for layer in range(1, depth):
        # nearest-neighbor CZ-entanglement
        for qubit in range(num_qubits - 1):
            qml.CNOT(wires=(qubit, qubit + 1))
        if num_qubits > 2:
            qml.CNOT(wires=(num_qubits - 1, 0))
        # parameterized rotations
        for qubit in range(weights.shape[1]):
            qml.U3(*weights[layer, qubit], wires=qubit)


def validate_with_classical_solution(policy: np.ndarray, Q_pi_val: np.ndarray, non_terminal_states: list[int],
                                     threshold: float = 0.001):
    """
    Compare greedy policy obtained by variational LSE solver with classical policy iteration.
    The default threshold of 0.001 is motivated by the default setting of 1000 shots for variational policy evaluation.

    :param policy: Greedy policy obtained from variational policy evaluation with consecutive policy improvement.
    :param Q_pi_val: State-action value function obtained from classical policy evaluation.
    :param non_terminal_states: List of non-terminal states.
    :param threshold: Threshold for which to consider Q-values as identical (default: 0.001).
    """
    # iterate over non-terminal states (policy for terminal states does not matter)
    for state in non_terminal_states:
        greedy_action = policy[state]
        greedy_action_val = np.argmax(Q_pi_val[state])
        # check if variational and classical greedy action do not align
        if not greedy_action == greedy_action_val:
            # check if difference in associated Q-values is above threshold
            if np.abs(Q_pi_val[state, greedy_action] - Q_pi_val[state, greedy_action_val]) > threshold:
                print('WARNING: Variational and classical greedy policy do not align!')
                break


def unitary_decomposition(matrix: np.ndarray, validate: float = 1e-10) -> list[np.ndarray]:
    """
    This method realizes and constructive decomposition of real matrices into an affine combination of 4 unitaries.
    The construction is based on a proof from
    "Additive Decomposition of Real Matrices", C.-K. Li et al., Linear and Multilinear Algebra 50.4 (321-326), 2002.

    :param matrix: Arbitrary real square input matrix.
    :param validate: Validate accuracy of the decomposition up to this threshold (numerical inaccuracies might occur).
    :return: Additive decomposition A/||A|| = (X + X^t + Y - Z) / 2.
    """

    # make sure the matrix is square and has an even number of row / columns
    # (in principle the construction can be extended to non-even dimensions)
    assert 0 == matrix.shape[0] % 2 and matrix.shape[0] == matrix.shape[1]
    dim = matrix.shape[0]

    # norm matrix, this is nor problem as the variational LSE solver only prepares a proportional solution
    normed_matrix = matrix / np.linalg.norm(matrix)

    # perform SVD decomposition of the normed matrix
    # Note: This step would eliminate any potential quantum advantage if done classically.
    #       Finding a method for more efficient quantum access was out of the scope of this work.
    u, s, v_t = np.linalg.svd(normed_matrix, full_matrices=False)

    # swap to descending order of singular values to align with original paper
    u, s, v_t = u[:, ::-1], np.flip(s), v_t[::-1, :]

    # set up R as the direct sum of dim/2 copies of 1/sqrt(2) [[-1, 1], [1, 1]]
    r = scipy.linalg.block_diag(*[np.array([[-1, 1], [1, 1]]) / np.sqrt(2) for _ in range(dim // 2)])

    # set up X^, Y^, Z^ as the direct sum as described in the paper
    x_hat = scipy.linalg.block_diag(*[np.array([
        [(s[j] + s[j + 1]) / 2, np.sqrt(1 - ((s[j] + s[j + 1]) / 2) ** 2)],
        [-np.sqrt(1 - ((s[j] + s[j + 1]) / 2) ** 2), (s[j] + s[j + 1]) / 2]])
        for j in range(0, dim, 2)])
    y_hat = scipy.linalg.block_diag(*[np.array([
        [np.sqrt(1 - ((s[j + 1] - s[j]) / 2) ** 2), (s[j + 1] - s[j]) / 2],
        [(s[j + 1] - s[j]) / 2, -np.sqrt(1 - ((s[j + 1] - s[j]) / 2) ** 2)]])
        for j in range(0, dim, 2)])
    z_hat = scipy.linalg.block_diag(*[np.array([
        [np.sqrt(1 - ((s[j + 1] - s[j]) / 2) ** 2), -(s[j + 1] - s[j]) / 2],
        [-(s[j + 1] - s[j]) / 2, -np.sqrt(1 - ((s[j + 1] - s[j]) / 2) ** 2)]])
        for j in range(0, dim, 2)])

    # compute the orthogonal matrices X, Y, Z
    x = u @ r @ x_hat @ r.T @ v_t
    x_t = u @ r @ x_hat.T @ r.T @ v_t
    y = u @ r @ y_hat @ r.T @ v_t
    z = - u @ r @ z_hat @ r.T @ v_t

    # test if decomposition is good (numerical inaccuracies can occur due to SVD decomposition)
    reconstructed_matrix = 0.5 * np.linalg.norm(matrix) * (x + x_t + y + z)
    if not np.all(np.isclose(matrix, reconstructed_matrix, atol=validate)):
        warnings.warn('Decomposition into unitaries does not satisfy given accuracy threshold.')
    # test if decomposed terms are actually unitary
    for unitary in [x, y, z]:
        if not np.all(np.isclose(np.eye(dim), unitary @ np.conjugate(unitary).T)):
            warnings.warn('Some of the decomposed terms are not unitary.')

    # decomposition X + X^t + Y - Z  (factor of 1/2 does not matter in our case)
    return [x, x_t, y, z]


def parse():
    parser = argparse.ArgumentParser()
    choices_environment = ['4x4', '8x8']
    parser.add_argument('--environment', type=str, default='4x4', choices=choices_environment,
                        help='Size of FrozenLake environment')
    parser.add_argument('--perturbation', type=float, default=0.1,
                        help='Transition dynamics perturbation.')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Maximum number of policy evaluation and improvement cycles.')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor for LSE associated with RL environment.')
    parser.add_argument('--depth', type=int, default=12,
                        help='Depth of variational ansatz.')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Maximum number of steps to perform with variational LSE solver.')
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='Early termination threshold for loss of variational LSE solver.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility.')
    parser.add_argument('--warm_start', action='store_true',
                        help='Use WS-VarQPI algorithm (by default VarQPI is used).')
    args = parser.parse_args()
    return args
