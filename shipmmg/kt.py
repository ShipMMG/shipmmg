#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from typing import List
import numpy as np
from scipy.misc import derivative
from scipy.interpolate import interp1d
from scipy.integrate import odeint


@dataclasses.dataclass
class KTParams:
    """Dataclass for setting KT parameters of KT simulation.

    Attributes:
        K (float): One of parameters in KT model. [1/s]
        T (float): One of parameters in KT model. [s]
    """

    K: float
    T: float


def simulate_kt(
    kt_params: KTParams,
    time_list: List[float],
    delta_list: List[float],
    r0: float = 0.0,
) -> np.ndarray:
    """KT simulation
    KT simulation by following equation of motion.

    :math:`T dr =  -r + K \\delta`

    Args:
        kt_params (KTParams):
            KT parameters.
        time_list (list[float]):
            time list of simulation.
        delta_list (list[float]):
            rudder angle list of simulation.
        r0 (float, optional):
            rate of turn [rad/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.

    Returns:
        numpy.ndarray:
            The result of KT simulation.
            shape = (time, num_of_params)
            num_of_params = 2 including :math:`r` and :math:`\\delta`.

    Examples:
        >>> kt_params = KTParams(K=0.15, T=60.0)
        >>> duration = 300
        >>> num_of_sampling = 3000
        >>> time_list = np.linspace(0.00, duration, num_of_sampling)
        >>> delta_list = 35 * np.pi / 180 * np.sin(3.0 * np.pi / Ts * time_list)
        >>> r0 = 0.0
        >>> result = simulate_kt(kt_params, time_list, delta_list, r0)
    """
    return simulate(
        K=kt_params.K, T=kt_params.T, time_list=time_list, delta_list=delta_list, r0=r0
    )


def simulate(
    K: float, T: float, time_list: List[float], delta_list: List[float], r0: float = 0.0
) -> np.ndarray:
    """KT simulation
    KT simulation by following equation of motion.

    :math:`T dr =  -r + K \\delta`

    Args:
        K (float):
            parameter K of KT model.
        T (float):
            parameter T of KT model.
        time_list (list[float]):
            time list of simulation.
        delta_list (list[float]):
            rudder angle list of simulation.
        r0 (float, optional):
            rate of turn [rad/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.

    Returns:
        numpy.ndarray:
            The result of KT simulation.
            shape = (time, num_of_results)
            num_of_results = 2 including :math:`r` and :math:`\\delta`.

    Examples:
        >>> K = 0.15
        >>> T = 60.0
        >>> duration = 300
        >>> num_of_sampling = 3000
        >>> time_list = np.linspace(0.00, duration, num_of_sampling)
        >>> delta_list = 35 * np.pi / 180 * np.sin(3.0 * np.pi / Ts * time_list)
        >>> r0 = 0.0
        >>> result = simulate(K, T, time_list, delta_list, r0)
    """
    spl_delta = interp1d(time_list, delta_list, "cubic", fill_value="extrapolate")

    def kt_eom(X, t):
        """Equation of Motion for KT simulation.
        d_r = 1/T * ( -r + K * delta)
        d_delta = derivative(spl_delta, t)
        """
        d_r = 1.0 / T * (-X[0] + K * X[1])
        d_delta = derivative(spl_delta, t)
        return [d_r, d_delta]

    X_init = np.array([r0, delta_list[0]])
    X_result = odeint(kt_eom, X_init, time_list)
    return X_result
