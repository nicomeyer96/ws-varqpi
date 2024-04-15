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

from variational_lse_solver import VarLSESolver
from environments_lse import BaseEnvironmentLSE, FrozenLakeLSE
from utils import circuit_u3, parse, validate_with_classical_solution, unitary_decomposition


def variational_qpi(
        env: BaseEnvironmentLSE,
        iterations: int = 10,
        gamma: float = 0.9,
        depth: int = 12,
        steps: int = 10000,
        threshold: float = 1e-4,
        warm_start: bool = False,
        decompose: bool = False,
        seed: float = None,
        validate: bool = False
        ):
    """
    Warm-start variational quantum policy iteration as described in
    "Warm-Start Variational Quantum Policy Iteration", N. Meyer et al. (2024).

    :param env: Environment to perform policy iteration on
    :param iterations: Maximum number of repetitions of policy evaluation and improvement
    :param gamma: Discount factor of underlying MDP.
    :param depth: VQC depth.
    :param steps: Number of steps for each LSE solver setup.
    :param threshold: Loss threshold for early stopping.
    :param warm_start: Use warm start for parameter initialization.
    :param decompose: Decompose the system matrix into a sum of unitaries.
    :param seed: Fix random seed for initializing parameters and policy.
    :param validate: Validate solution against classical ground truth.
    """

    # fix seed if provided
    np.random.seed(seed)

    # set up initial LSE (with randomized initial deterministic policy)
    A_pi, R, policy, _ = env.lse(gamma=gamma)

    # initialize weights uniformly at random in [0, 2*pi]
    initial_weights = 2 * np.pi * np.random.rand(depth, int(np.log2(A_pi.shape[0])), 3)

    print('Initial policy:')
    env.print_policy(policy)

    # perform policy iteration for `max_iterations` steps
    for iteration in range(1, iterations + 1):

        # if `decompose` is True, use constructive decomposition of matrix into 4 unitaries
        # otherwise use the `direct` mode from the variational-lse-solver library with the full (non-unitary) matrix
        # Note: The results might deviate to some extent, due to matrix re-scaling and numerical inaccuracies
        A_pi_unitaries = []
        if decompose:
            A_pi_unitaries = unitary_decomposition(A_pi)

        # set up variational LSE solver
        var_lse_solver = VarLSESolver(
            a=A_pi_unitaries if decompose else A_pi,
            b=R,
            coeffs=[1.0, 1.0, 1.0, 1.0] if decompose else None,
            ansatz=circuit_u3,
            weights=initial_weights,
            method='hadamard' if decompose else 'direct',
            lr=0.01,
            steps=steps,
            threshold=threshold
        )

        # apply variational LSE solver
        Q_pi, weights = var_lse_solver.solve()

        # warm-start next parameters with previous ones
        if warm_start:
            initial_weights = weights

        # select maximizing actions -> policy improvement
        greedy_policy = np.argmax(np.reshape(Q_pi, (-1, env.num_actions)), axis=1)

        # validate with classical solution
        if validate:
            Q_pi_val = np.linalg.solve(A_pi, R)
            Q_pi_val = np.square(Q_pi_val / np.linalg.norm(Q_pi_val))
            validate_with_classical_solution(greedy_policy,
                                             np.reshape(Q_pi_val, (-1, env.num_actions)),
                                             env.non_terminal_states)

        # check termination, i.e. if policy (for non-terminal states) has not changed over two successive iterations
        if np.all(np.array((policy == greedy_policy))[env.non_terminal_states]):
            # print final policy
            print('Final policy:')
            env.print_policy(greedy_policy)
            print(f'TERMINATED after {iteration} iterations')
            break

        # update policy handle and associated LSE for next iteration
        A_pi, R, policy, _ = env.lse(greedy_policy, gamma=gamma)

        print(f'Policy after iteration {iteration}:')
        env.print_policy(policy)


if __name__ == '__main__':

    _args = parse()
    _env = FrozenLakeLSE(map_name=_args.environment, perturbation=_args.perturbation)

    # run (warm-start) variational quantum policy iteration
    variational_qpi(
        _env,
        iterations=_args.iterations,
        gamma=_args.gamma,
        depth=_args.depth,
        steps=_args.steps,
        threshold=_args.threshold,
        warm_start=_args.warm_start,
        decompose=False,  # set to True to decompose system matrix before using variational LSE solve
        seed=_args.seed,
        validate=False  # set to True to compare to classical ground truth after each iteration
    )
