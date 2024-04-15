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


from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseEnvironmentLSE(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def lse(self, policy: np.ndarray = None, **kwargs) -> tuple[Any, Any, Any, Any]:
        """
        Abstract method for setting up the LSE representation of the environment.

        :return: System matrix A_pi = I - gamma * P @ Pi, right side R = [sum_{s'} p(s'|s,a) pi(a|s)]_{s,a},
                 underlying deterministic policy pi, raw dynamics-policy matrix P * Pi
        """
        pass

    @abstractmethod
    def non_terminal_states(self) -> list[int]:
        """
        Abstract method for returning a list of non-terminal states.
        """
        pass

    @abstractmethod
    def terminal_states(self) -> list[int]:
        """
        Abstract method for returning a list of terminal states.
        """
        pass

    @abstractmethod
    def print_policy(self, policy: np.ndarray):
        """
        Abstract method for printing the environment following the policy.
        """
        pass

    @abstractmethod
    def env_map(self) -> Any:
        """
        Abstract method for returning the environment layout.
        """
        pass

    @abstractmethod
    def num_states(self) -> int:
        """
        Abstract method for returning the number of states in the environment
        """

    @abstractmethod
    def num_actions(self) -> int:
        """
        Abstract method for returning the number of actions possible in the environment
        """
