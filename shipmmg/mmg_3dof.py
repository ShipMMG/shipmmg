#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from typing import List
import numpy as np
from scipy.misc import derivative
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from .ship_obj_3dof import ShipObj3dof


@dataclasses.dataclass
class Mmg3DofBasicParams:
    """Dataclass for setting basic parameters of MMG 3DOF.

    Attributes:
        L_pp (float):
            Ship length between perpendiculars [m]
        B (float):
            Ship breadth [m]
        d (float):
            Ship draft [m]
        x_G (float):
            Longitudinal coordinate of center of gravity of ship [m]
        D_p (float):
            Propeller diameter [m]
        m (float):
            Ship mass [kg]
        I_zG (float):
            Moment of inertia of ship around center of gravity
        A_R (float):
            Profile area of movable part of mariner rudder [m^2]
        η (float):
            Ratio of propeller diameter to rudder span (=D_p/HR)
        m_x (float):
            Added masses of x axis direction [kg]
        m_y (float):
            Added masses of y axis direction [kg]
        J_z (float):
            Added moment of inertia
        f_α (float):
            Rudder lift gradient coefficient
        ϵ (float):
            Ratio of wake fraction at propeller and rudder positions
        t_R (float):
            Steering resistance deduction factor
        a_H (float):
            Rudder force increase factor
        x_H (float):
            Longitudinal coordinate of acting point of the additional lateral force component induced by steering
        γ_R_minus (float):
            Flow straightening coefficient if βR < 0
        γ_R_plus (float):
            Flow straightening coefficient if βR > 0
        l_R (float):
            Effective longitudinal coordinate of rudder position in formula of βR
        κ (float):
            An experimental constant for　expressing uR
        t_P (float):
            Thrust deduction factor
        w_P0 (float):
            Wake coefficient at propeller position in straight moving
        x_P (float):
            Effective Longitudinal coordinate of propeller position in formula of βP

    Note:
        For more information, please see the following articles.

        - Yasukawa, H., Yoshimura, Y. (2015) Introduction of MMG standard method for ship maneuvering predictions.
          J Mar Sci Technol 20, 37–52 https://doi.org/10.1007/s00773-014-0293-y
    """

    L_pp: float
    B: float
    d: float
    x_G: float
    D_p: float
    m: float
    I_zG: float
    A_R: float
    η: float
    m_x: float
    m_y: float
    J_z: float
    f_α: float
    ϵ: float
    t_R: float
    a_H: float
    x_H: float
    γ_R_minus: float
    γ_R_plus: float
    l_R: float
    κ: float
    t_P: float
    w_P0: float
    x_P: float


@dataclasses.dataclass
class Mmg3DofManeuveringParams:
    """Dataclass for setting maneuvering parameters of MMG 3ODF.

    Attributes:
        k_0 (float): One of manuevering parameters of coefficients representing K_T
        k_1 (float): One of manuevering parameters of coefficients representing K_T
        k_2 (float): One of manuevering parameters of coefficients representing K_T
        R_0_dash (float): One of manuevering parameters of Ship resistance coefficient in straight moving
        X_vv_dash (float): One of manuevering parameters of MMG 3DOF
        X_vr_dash (float): One of manuevering parameters of MMG 3DOF
        X_rr_dash (float): One of manuevering parameters of MMG 3DOF
        X_vvvv_dash (float): One of manuevering parameters of MMG 3DOF
        Y_v_dash (float): One of manuevering parameters of MMG 3DOF
        Y_r_dash (float): One of manuevering parameters of MMG 3DOF
        Y_vvv_dash (float): One of manuevering parameters of MMG 3DOF
        Y_vvr_dash (float): One of manuevering parameters of MMG 3DOF
        Y_vrr_dash (float): One of manuevering parameters of MMG 3DOF
        Y_rrr_dash (float): One of manuevering parameters of MMG 3DOF
        N_v_dash (float): One of manuevering parameters of MMG 3DOF
        N_r_dash (float): One of manuevering parameters of MMG 3DOF
        N_vvv_dash (float): One of manuevering parameters of MMG 3DOF
        N_vvr_dash (float): One of manuevering parameters of MMG 3DOF
        N_vrr_dash (float): One of manuevering parameters of MMG 3DOF
        N_rrr_dash (float): One of manuevering parameters of MMG 3DOF

    Note:
        For more information, please see the following articles.

        - Yasukawa, H., Yoshimura, Y. (2015) Introduction of MMG standard method for ship maneuvering predictions.
          J Mar Sci Technol 20, 37–52 https://doi.org/10.1007/s00773-014-0293-y
    """

    k_0: float
    k_1: float
    k_2: float
    R_0_dash: float
    X_vv_dash: float
    X_vr_dash: float
    X_rr_dash: float
    X_vvvv_dash: float
    Y_v_dash: float
    Y_r_dash: float
    Y_vvv_dash: float
    Y_vvr_dash: float
    Y_vrr_dash: float
    Y_rrr_dash: float
    N_v_dash: float
    N_r_dash: float
    N_vvv_dash: float
    N_vvr_dash: float
    N_vrr_dash: float
    N_rrr_dash: float


def simulate_mmg_3dof(
    basic_params: Mmg3DofBasicParams,
    maneuvering_params: Mmg3DofManeuveringParams,
    time_list: List[float],
    δ_list: List[float],
    npm_list: List[float],
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    ρ: float = 1.025,
    method: str = "RK45",
    t_eval=None,
    events=None,
    vectorized=False,
    **options
):
    """MMG 3DOF simulation
    MMG 3DOF simulation by follwoing equation of motion.

    .. math::

        m (\\dot{u}-vr)&=-m_x\\dot{u}+m_yvr+X_H+X_P+X_R

        m (\\dot{v}+ur)&=-m_y\\dot{v}+m_xur+Y_H+Y_R

        I_{zG}\\dot{r}&=-J_Z\\dot{r}+N_H+N_R

    Args:
        basic_params (Mmg3DofBasicParams):
            Basic paramters for MMG 3DOF simulation.
        maneuvering_params (Mmg3DofManeuveringParams):
            Maneuvering parameters for MMG 3DOF simulation.
        time_list (list[float]):
            time list of simulation.
        δ_list (list[float]):
            rudder angle list of simulation.
        npm_list (List[float]):
            npm list of simulation.
        u0 (float, optional):
            axial velocity [m/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        v0 (float, optional):
            lateral velocity [m/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        r0 (float, optional):
            rate of turn [rad/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        ρ (float, optional):
            seawater density [kg/m^3]
            Defaults to 1.025.
        method (str, optional):
            Integration method to use in
            `scipy.integrate.solve_ivp()
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_:

                "RK45" (default):
                    Explicit Runge-Kutta method of order 5(4).
                    The error is controlled assuming accuracy of the fourth-order method,
                    but steps are taken using the fifth-order accurate formula (local extrapolation is done).
                    A quartic interpolation polynomial is used for the dense output.
                    Can be applied in the complex domain.
                "RK23":
                    Explicit Runge-Kutta method of order 3(2).
                    The error is controlled assuming accuracy of the second-order method,
                    but steps are taken using the third-order accurate formula (local extrapolation is done).
                    A cubic Hermite polynomial is used for the dense output.
                    Can be applied in the complex domain.
                "DOP853":
                    Explicit Runge-Kutta method of order 8.
                    Python implementation of the “DOP853” algorithm originally written in Fortran.
                    A 7-th order interpolation polynomial accurate to 7-th order is used for the dense output.
                    Can be applied in the complex domain.
                "Radau":
                    Implicit Runge-Kutta method of the Radau IIA family of order 5.
                    The error is controlled with a third-order accurate embedded formula.
                    A cubic polynomial which satisfies the collocation conditions is used for the dense output.
                "BDF":
                    Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the derivative approximation.
                    A quasi-constant step scheme is used and accuracy is enhanced using the NDF modification.
                    Can be applied in the complex domain.
                "LSODA":
                    Adams/BDF method with automatic stiffness detection and switching.
                    This is a wrapper of the Fortran solver from ODEPACK.

        t_eval (array_like or None, optional):
            Times at which to store the computed solution, must be sorted and lie within t_span.
            If None (default), use points selected by the solver.
        events (callable, or list of callables, optional):
            Events to track. If None (default), no events will be tracked.
            Each event occurs at the zeros of a continuous function of time and state.
            Each function must have the signature event(t, y) and return a float.
            The solver will find an accurate value of t at which event(t, y(t)) = 0 using a root-finding algorithm.
            By default, all zeros will be found. The solver looks for a sign change over each step,
            so if multiple zero crossings occur within one step, events may be missed.
            Additionally each event function might have the following attributes:
                terminal (bool, optional):
                    Whether to terminate integration if this event occurs. Implicitly False if not assigned.
                direction (float, optional):
                    Direction of a zero crossing.
                    If direction is positive, event will only trigger when going from negative to positive,
                    and vice versa if direction is negative.
                    If 0, then either direction will trigger event. Implicitly 0 if not assigned.
            You can assign attributes like `event.terminal = True` to any function in Python.
        vectorized (bool, optional):
            Whether `fun` is implemented in a vectorized fashion. Default is False.
        options:
            Options passed to a chosen solver.
            All options available for already implemented solvers are listed in
            `scipy.integrate.solve_ivp()
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_:

    Returns:
        Bunch object with the following fields defined:
            t (ndarray, shape (`n_points`,)):
                Time points.
            y (ndarray, shape (`n_points`,)):
                Values of the solution at t.
            sol (OdeSolution):
                Found solution as OdeSolution instance from MMG 3DOF simulation.
            t_events (list of ndarray or None):
                Contains for each event type a list of arrays at which an event of that type event was detected.
                None if events was None.
            y_events (list of ndarray or None):
                For each value of t_events, the corresponding value of the solution.
                None if events was None.
            nfev (int):
                Number of evaluations of the right-hand side.
            njev (int):
                Number of evaluations of the jacobian.
            nlu (int):
                Number of LU decomposition.
            status (int):
                Reason for algorithm termination:
                    - -1: Integration step failed.
                    - 0: The solver successfully reached the end of `tspan`.
                    - 1: A termination event occurred.
            message (string):
                Human-readable description of the termination reason.
            success (bool):
                True if the solver reached the interval end or a termination event occurred (`status >= 0`).

    Examples:
        >>> duration = 200  # [s]
        >>> sampling = 2000
        >>> time_list = np.linspace(0.00, duration, sampling)
        >>> δ_list = np.full(len(time_list), 35.0 * np.pi / 180.0)
        >>> npm_list = np.full(len(time_list), 20.338)
        >>> basic_params = Mmg3DofBasicParams(
        >>>                     L_pp=7.00,
        >>>                     B=1.27,
        >>>                     d=0.46,
        >>>                     x_G=0.25,
        >>>                     D_p=0.216,
        >>>                     m=3.27*1.025,
        >>>                     I_zG=m*((0.25 * L_pp) ** 2),
        >>>                     A_R=0.0539,
        >>>                     η=D_p/0.345,
        >>>                     m_x=0.022*(0.5 * ρ * (L_pp ** 2) * d),
        >>>                     m_y=0.223*(0.5 * ρ * (L_pp ** 2) * d),
        >>>                     J_z=0.011*(0.5 * ρ * (L_pp ** 4) * d),
        >>>                     f_α=2.747,
        >>>                     ϵ=1.09,
        >>>                     t_R=0.387,
        >>>                     a_H=0.312,
        >>>                     x_H=-0.464*L_pp,
        >>>                     γ_R_minus=0.395,
        >>>                     γ_R_plus=0.640,
        >>>                     l_R=-0.710,
        >>>                     κ=0.50,
        >>>                     t_P=0.220,
        >>>                     w_P0=0.40,
        >>>                     x_P=-0.650,
        >>>                 )
        >>> maneuvering_params = Mmg3DofManeuveringParams(
        >>>                     k_0 = 0.2931,
        >>>                     k_1 = -0.2753,
        >>>                     k_2 = -0.1385,
        >>>                     R_0_dash = 0.022,
        >>>                     X_vv_dash = -0.040,
        >>>                     X_vr_dash = 0.002,
        >>>                     X_rr_dash = 0.011,
        >>>                     X_vvvv_dash = 0.771,
        >>>                     Y_v_dash = -0.315,
        >>>                     Y_r_dash = 0.083,
        >>>                     Y_vvv_dash = -1.607,
        >>>                     Y_vvr_dash = 0.379,
        >>>                     Y_vrr_dash = -0.391,
        >>>                     Y_rrr_dash = 0.008,
        >>>                     N_v_dash = -0.137,
        >>>                     N_r_dash = -0.049,
        >>>                     N_vvv_dash = -0.030,
        >>>                     N_vvr_dash = -0.294,
        >>>                     N_vrr_dash = 0.055,
        >>>                     N_rrr_dash = -0.013,
        >>>                )
        >>> sol = simulate_mmg_3dof(
        >>>                    basic_params,
        >>>                    maneuvering_params,
        >>>                    time_list,
        >>>                    δ_rad_list,
        >>>                    npm_list,
        >>>                    u0=2.29 * 0.512,
        >>>                )
        >>> result = sol.sol(time_list)


    Note:
        For more information, please see the following articles.

        - Yasukawa, H., Yoshimura, Y. (2015) Introduction of MMG standard method for ship maneuvering predictions.
          J Mar Sci Technol 20, 37–52 https://doi.org/10.1007/s00773-014-0293-y

    """
    return simulate(
        L_pp=basic_params.L_pp,
        B=basic_params.B,
        d=basic_params.d,
        x_G=basic_params.x_G,
        D_p=basic_params.D_p,
        m=basic_params.m,
        I_zG=basic_params.I_zG,
        A_R=basic_params.A_R,
        η=basic_params.η,
        m_x=basic_params.m_x,
        m_y=basic_params.m_y,
        J_z=basic_params.J_z,
        f_α=basic_params.f_α,
        ϵ=basic_params.ϵ,
        t_R=basic_params.t_R,
        a_H=basic_params.a_H,
        x_H=basic_params.x_H,
        γ_R_minus=basic_params.γ_R_minus,
        γ_R_plus=basic_params.γ_R_plus,
        l_R=basic_params.l_R,
        κ=basic_params.κ,
        t_P=basic_params.t_P,
        w_P0=basic_params.w_P0,
        x_P=basic_params.x_P,
        k_0=maneuvering_params.k_0,
        k_1=maneuvering_params.k_1,
        k_2=maneuvering_params.k_2,
        R_0_dash=maneuvering_params.R_0_dash,
        X_vv_dash=maneuvering_params.X_vv_dash,
        X_vr_dash=maneuvering_params.X_vr_dash,
        X_rr_dash=maneuvering_params.X_rr_dash,
        X_vvvv_dash=maneuvering_params.X_vvvv_dash,
        Y_v_dash=maneuvering_params.Y_v_dash,
        Y_r_dash=maneuvering_params.Y_r_dash,
        Y_vvv_dash=maneuvering_params.Y_vvv_dash,
        Y_vvr_dash=maneuvering_params.Y_vvr_dash,
        Y_vrr_dash=maneuvering_params.Y_vrr_dash,
        Y_rrr_dash=maneuvering_params.Y_rrr_dash,
        N_v_dash=maneuvering_params.N_v_dash,
        N_r_dash=maneuvering_params.N_r_dash,
        N_vvv_dash=maneuvering_params.N_vvv_dash,
        N_vvr_dash=maneuvering_params.N_vvr_dash,
        N_vrr_dash=maneuvering_params.N_vrr_dash,
        N_rrr_dash=maneuvering_params.N_rrr_dash,
        time_list=time_list,
        δ_list=δ_list,
        npm_list=npm_list,
        u0=u0,
        v0=v0,
        r0=r0,
        ρ=ρ,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )


def simulate(
    L_pp: float,
    B: float,
    d: float,
    x_G: float,
    D_p: float,
    m: float,
    I_zG: float,
    A_R: float,
    η: float,
    m_x: float,
    m_y: float,
    J_z: float,
    f_α: float,
    ϵ: float,
    t_R: float,
    a_H: float,
    x_H: float,
    γ_R_minus: float,
    γ_R_plus: float,
    l_R: float,
    κ: float,
    t_P: float,
    w_P0: float,
    x_P: float,
    k_0: float,
    k_1: float,
    k_2: float,
    R_0_dash: float,
    X_vv_dash: float,
    X_vr_dash: float,
    X_rr_dash: float,
    X_vvvv_dash: float,
    Y_v_dash: float,
    Y_r_dash: float,
    Y_vvv_dash: float,
    Y_vvr_dash: float,
    Y_vrr_dash: float,
    Y_rrr_dash: float,
    N_v_dash: float,
    N_r_dash: float,
    N_vvv_dash: float,
    N_vvr_dash: float,
    N_vrr_dash: float,
    N_rrr_dash: float,
    time_list: List[float],
    δ_list: List[float],
    npm_list: List[float],
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    ρ: float = 1.025,
    method: str = "RK45",
    t_eval=None,
    events=None,
    vectorized=False,
    **options
):
    """MMG 3DOF simulation
    MMG 3DOF simulation by follwoing equation of motion.

    .. math::

        m (\\dot{u}-vr)&=-m_x\\dot{u}+m_yvr+X_H+X_P+X_R

        m (\\dot{v}+ur)&=-m_y\\dot{v}+m_xur+Y_H+Y_R

        I_{zG}\\dot{r}&=-J_Z\\dot{r}+N_H+N_R

    Args:
        L_pp (float):
            Ship length between perpendiculars [m]
        B (float):
            Ship breadth [m]
        d (float):
            Ship draft [m]
        x_G (float):
            Longitudinal coordinate of center of gravity of ship
        D_p (float):
            Propeller diameter [m]
        m (float):
            Ship mass [kg]
        I_zG (float):
            Moment of inertia of ship around center of gravity
        A_R (float):
            Profile area of movable part of mariner rudder [m^2]
        η (float):
            Ratio of propeller diameter to rudder span (=D_p/HR)
        m_x (float):
            Added masses of x axis direction [kg]
        m_y (float):
            Added masses of y axis direction [kg]
        J_z (float):
            Added moment of inertia
        f_α (float):
            Rudder lift gradient coefficient
        ϵ (float):
            Ratio of wake fraction at propeller and rudder positions
        t_R (float):
            Steering resistance deduction factor
        a_H (float):
            Rudder force increase factor
        x_H (float):
            Longitudinal coordinate of acting point of the additional lateral force component induced by steering
        γ_R_minus (float):
            Flow straightening coefficient if βR < 0
        γ_R_plus (float):
            Flow straightening coefficient if βR > 0
        l_R (float):
            Effective longitudinal coordinate of rudder position in formula of βR
        κ (float):
            An experimental constant for　expressing uR
        t_P (float):
            Thrust deduction factor
        w_P0 (float):
            Wake coefficient at propeller position in straight moving
        x_P (float):
            Effective Longitudinal coordinate of propeller position in formula of βP
        k_0 (float):
            One of manuevering parameters of coefficients representing K_T
        k_1 (float):
            One of manuevering parameters of coefficients representing K_T
        k_2 (float):
            One of manuevering parameters of coefficients representing K_T
        R_0_dash (float):
            One of manuevering parameters of MMG 3DOF
        X_vv_dash (float):
            One of manuevering parameters of MMG 3DOF
        X_vr_dash (float):
            One of manuevering parameters of MMG 3DOF
        X_rr_dash (float):
            One of manuevering parameters of MMG 3DOF
        X_vvvv_dash (float):
            One of manuevering parameters of MMG 3DOF
        Y_v_dash (float):
            One of manuevering parameters of MMG 3DOF
        Y_r_dash (float):
            One of manuevering parameters of MMG 3DOF
        Y_vvv_dash (float):
            One of manuevering parameters of MMG 3DOF
        Y_vvr_dash (float):
            One of manuevering parameters of MMG 3DOF
        Y_vrr_dash (float):
            One of manuevering parameters of MMG 3DOF
        Y_rrr_dash (float):
            One of manuevering parameters of MMG 3DOF
        N_v_dash (float):
            One of manuevering parameters of MMG 3DOF
        N_r_dash (float):
            One of manuevering parameters of MMG 3DOF
        N_vvv_dash (float):
            One of manuevering parameters of MMG 3DOF
        N_vvr_dash (float):
            One of manuevering parameters of MMG 3DOF
        N_vrr_dash (float):
            One of manuevering parameters of MMG 3DOF
        N_rrr_dash (float):
            One of manuevering parameters of MMG 3DOF
        time_list (list[float]):
            time list of simulation.
        δ_list (list[float]):
            rudder angle list of simulation.
        npm_list (List[float]):
            npm list of simulation.
        u0 (float, optional):
            axial velocity [m/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        v0 (float, optional):
            lateral velocity [m/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        r0 (float, optional):
            rate of turn [rad/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        ρ (float, optional):
            seawater density [kg/m^3]
            Defaults to 1.025.
        method (str, optional):
            Integration method to use in
            `scipy.integrate.solve_ivp()
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_:

                "RK45" (default):
                    Explicit Runge-Kutta method of order 5(4).
                    The error is controlled assuming accuracy of the fourth-order method,
                    but steps are taken using the fifth-order accurate formula (local extrapolation is done).
                    A quartic interpolation polynomial is used for the dense output.
                    Can be applied in the complex domain.
                "RK23":
                    Explicit Runge-Kutta method of order 3(2).
                    The error is controlled assuming accuracy of the second-order method,
                    but steps are taken using the third-order accurate formula (local extrapolation is done).
                    A cubic Hermite polynomial is used for the dense output.
                    Can be applied in the complex domain.
                "DOP853":
                    Explicit Runge-Kutta method of order 8.
                    Python implementation of the “DOP853” algorithm originally written in Fortran.
                    A 7-th order interpolation polynomial accurate to 7-th order is used for the dense output.
                    Can be applied in the complex domain.
                "Radau":
                    Implicit Runge-Kutta method of the Radau IIA family of order 5.
                    The error is controlled with a third-order accurate embedded formula.
                    A cubic polynomial which satisfies the collocation conditions is used for the dense output.
                "BDF":
                    Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the derivative approximation.
                    A quasi-constant step scheme is used and accuracy is enhanced using the NDF modification.
                    Can be applied in the complex domain.
                "LSODA":
                    Adams/BDF method with automatic stiffness detection and switching.
                    This is a wrapper of the Fortran solver from ODEPACK.

        t_eval (array_like or None, optional):
            Times at which to store the computed solution, must be sorted and lie within t_span.
            If None (default), use points selected by the solver.
        events (callable, or list of callables, optional):
            Events to track. If None (default), no events will be tracked.
            Each event occurs at the zeros of a continuous function of time and state.
            Each function must have the signature event(t, y) and return a float.
            The solver will find an accurate value of t at which event(t, y(t)) = 0 using a root-finding algorithm.
            By default, all zeros will be found. The solver looks for a sign change over each step,
            so if multiple zero crossings occur within one step, events may be missed.
            Additionally each event function might have the following attributes:
                terminal (bool, optional):
                    Whether to terminate integration if this event occurs. Implicitly False if not assigned.
                direction (float, optional):
                    Direction of a zero crossing.
                    If direction is positive, event will only trigger when going from negative to positive,
                    and vice versa if direction is negative.
                    If 0, then either direction will trigger event. Implicitly 0 if not assigned.
            You can assign attributes like `event.terminal = True` to any function in Python.
        vectorized (bool, optional):
            Whether `fun` is implemented in a vectorized fashion. Default is False.
        options:
            Options passed to a chosen solver.
            All options available for already implemented solvers are listed in
            `scipy.integrate.solve_ivp()
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_:

    Returns:
        Bunch object with the following fields defined:
            t (ndarray, shape (`n_points`,)):
                Time points.
            y (ndarray, shape (`n_points`,)):
                Values of the solution at t.
            sol (OdeSolution):
                Found solution as OdeSolution instance from MMG 3DOF simulation.
            t_events (list of ndarray or None):
                Contains for each event type a list of arrays at which an event of that type event was detected.
                None if events was None.
            y_events (list of ndarray or None):
                For each value of t_events, the corresponding value of the solution.
                None if events was None.
            nfev (int):
                Number of evaluations of the right-hand side.
            njev (int):
                Number of evaluations of the jacobian.
            nlu (int):
                Number of LU decomposition.
            status (int):
                Reason for algorithm termination:
                    - -1: Integration step failed.
                    - 0: The solver successfully reached the end of `tspan`.
                    - 1: A termination event occurred.
            message (string):
                Human-readable description of the termination reason.
            success (bool):
                True if the solver reached the interval end or a termination event occurred (`status >= 0`).

    Examples:
        >>> duration = 200  # [s]
        >>> sampling = 2000
        >>> time_list = np.linspace(0.00, duration, sampling)
        >>> δ_list = np.full(len(time_list), 35.0 * np.pi / 180.0)
        >>> npm_list = np.full(len(time_list), 17.95)
        >>> L_pp=7.00
        >>> B=1.27
        >>> d=0.46
        >>> x_G=0.25
        >>> D_p=0.216
        >>> m=3.27*1.025
        >>> I_zG=m*((0.25 * L_pp) ** 2)
        >>> A_R=0.0539
        >>> η=D_p/0.345
        >>> m_x=0.022*(0.5 * ρ * (L_pp ** 2) * d)
        >>> m_y=0.223*(0.5 * ρ * (L_pp ** 2) * d)
        >>> J_z=0.011*(0.5 * ρ * (L_pp ** 4) * d)
        >>> f_α=2.747
        >>> ϵ=1.09
        >>> t_R=0.387
        >>> a_H=0.312
        >>> x_H=-0.464*L_pp
        >>> γ_R=0.395
        >>> l_R=-0.710
        >>> κ=0.50
        >>> t_P=0.220
        >>> w_P0=0.40
        >>> x_P=-0.650
        >>> k_0 = 0.2931
        >>> k_1 = -0.2753
        >>> k_2 = -0.1385
        >>> R_0_dash = 0.022
        >>> X_vv_dash = -0.040
        >>> X_vr_dash = 0.002
        >>> X_rr_dash = 0.011
        >>> X_vvvv_dash = 0.771
        >>> Y_v_dash = -0.315
        >>> Y_r_dash = 0.083
        >>> Y_vvv_dash = -1.607
        >>> Y_vvr_dash = 0.379
        >>> Y_vrr_dash = -0.391
        >>> Y_rrr_dash = 0.008
        >>> N_v_dash = -0.137
        >>> N_r_dash = -0.049
        >>> N_vvv_dash = -0.030
        >>> N_vvr_dash = -0.294
        >>> N_vrr_dash = 0.055
        >>> N_rrr_dash = -0.013
        >>> sol = simulate_mmg_3dof(
        >>>                    L_pp=L_pp,
        >>>                    B=B,
        >>>                    d=d,
        >>>                    x_G=x_G,
        >>>                    D_p=D_p,
        >>>                    m=m,
        >>>                    I_zG=I_zG,
        >>>                    A_R=A_R,
        >>>                    η=η,
        >>>                    m_x=m_x,
        >>>                    m_y=m_y,
        >>>                    J_z=J_z,
        >>>                    f_α=f_α,
        >>>                    ϵ=ϵ,
        >>>                    t_R=t_R,
        >>>                    a_H=a_H,
        >>>                    x_H=x_H,
        >>>                    γ_R=γ_R,
        >>>                    l_R=l_R,
        >>>                    κ=κ,
        >>>                    t_P=t_P,
        >>>                    w_P0=w_P0,
        >>>                    x_P=x_P,
        >>>                    k_0=k_0,
        >>>                    k_1=k_1,
        >>>                    k_2=k_2,
        >>>                    X_0=X_0,
        >>>                    X_ββ=X_ββ,
        >>>                    X_βγ=X_βγ,
        >>>                    X_γγ=X_γγ,
        >>>                    X_vvvv_dash=X_vvvv_dash,
        >>>                    Y_β=Y_β,
        >>>                    Y_γ=Y_γ,
        >>>                    Y_βββ=Y_βββ,
        >>>                    Y_vvr_dash=Y_vvr_dash,
        >>>                    Y_vrr_dash=Y_vrr_dash,
        >>>                    Y_rrr_dash=Y_rrr_dash,
        >>>                    N_β=N_β,
        >>>                    N_γ=N_γ,
        >>>                    N_vvv_dash=N_vvv_dash,
        >>>                    N_vvr_dash=N_vvr_dash,
        >>>                    N_vrr_dash=N_vrr_dash,
        >>>                    N_rrr_dash=N_rrr_dash,
        >>>                    time_list,
        >>>                    δ_rad_list,
        >>>                    npm_list,
        >>>                    u0=2.29 * 0.512,
        >>>                )
        >>> result = sol.sol(time_list)

    Note:
        For more information, please see the following articles.

        - Yasukawa, H., Yoshimura, Y. (2015) Introduction of MMG standard method for ship maneuvering predictions.
          J Mar Sci Technol 20, 37–52 https://doi.org/10.1007/s00773-014-0293-y

    """
    spl_δ = interp1d(time_list, δ_list, "cubic", fill_value="extrapolate")
    spl_npm = interp1d(time_list, npm_list, "cubic", fill_value="extrapolate")

    def mmg_3dof_eom_solve_ivp(t, X):

        u, v, r, δ, npm = X

        U = np.sqrt(u ** 2 + (v - r * x_G) ** 2)

        β = 0.0 if U == 0.0 else np.arcsin(-(v - r * x_G) / U)
        v_dash = 0.0 if U == 0.0 else v / U
        r_dash = 0.0 if U == 0.0 else r * L_pp / U

        # w_P = w_P0
        w_P = w_P0 * np.exp(-4.0 * (β - x_P * r_dash) ** 2)

        J = 0.0 if npm == 0.0 else (1 - w_P) * u / (npm * D_p)
        K_T = k_0 + k_1 * J + k_2 * J ** 2
        β_R = β - l_R * r_dash
        γ_R = γ_R_minus if β_R < 0.0 else γ_R_plus
        v_R = U * γ_R * β_R
        u_R = (
            np.sqrt(η * (κ * ϵ * 8.0 * k_0 * npm ** 2 * D_p ** 4 / np.pi) ** 2)
            if J == 0.0
            else u
            * (1 - w_P)
            * ϵ
            * np.sqrt(
                η * (1.0 + κ * (np.sqrt(1.0 + 8.0 * K_T / (np.pi * J ** 2)) - 1)) ** 2
                + (1 - η)
            )
        )
        U_R = np.sqrt(u_R ** 2 + v_R ** 2)
        α_R = δ - np.arctan2(v_R, u_R)
        F_N = 0.5 * A_R * ρ * f_α * (U_R ** 2) * np.sin(α_R)

        X_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U ** 2)
            * (
                -R_0_dash
                + X_vv_dash * (v_dash ** 2)
                + X_vr_dash * v_dash * r_dash
                + X_rr_dash * (r_dash ** 2)
                + X_vvvv_dash * (v_dash ** 4)
            )
        )
        X_R = -(1 - t_R) * F_N * np.sin(δ)
        X_P = (1 - t_P) * ρ * K_T * npm ** 2 * D_p ** 4
        Y_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U ** 2)
            * (
                Y_v_dash * v_dash
                + Y_r_dash * r_dash
                + Y_vvv_dash * (v_dash ** 3)
                + Y_vvr_dash * (v_dash ** 2) * r_dash
                + Y_vrr_dash * v_dash * (r_dash ** 2)
                + Y_rrr_dash * (r_dash ** 3)
            )
        )
        Y_R = -(1 + a_H) * F_N * np.cos(δ)
        N_H = (
            0.5
            * ρ
            * (L_pp ** 2)
            * d
            * (U ** 2)
            * (
                N_v_dash * v_dash
                + N_r_dash * r_dash
                + N_vvv_dash * (v_dash ** 3)
                + N_vvr_dash * (v_dash ** 2) * r_dash
                + N_vrr_dash * v_dash * (r_dash ** 2)
                + N_rrr_dash * (r_dash ** 3)
            )
        )
        N_R = -(-0.5 + a_H * x_H) * F_N * np.cos(δ)
        d_u = ((X_H + X_R + X_P) + (m + m_y) * v * r + x_G * m * (r ** 2)) / (m + m_x)
        d_v = (
            (x_G ** 2) * (m ** 2) * u * r
            - (N_H + N_R) * x_G * m
            + ((Y_H + Y_R) - (m + m_x) * u * r) * (I_zG + J_z + (x_G ** 2) * m)
        ) / ((I_zG + J_z + (x_G ** 2) * m) * (m + m_y) - (x_G ** 2) * (m ** 2))
        d_r = (N_H + N_R - x_G * m * (d_v + u * r)) / (I_zG + J_z + (x_G ** 2) * m)
        d_δ = derivative(spl_δ, t)
        d_npm = derivative(spl_npm, t)
        return [d_u, d_v, d_r, d_δ, d_npm]

    sol = solve_ivp(
        mmg_3dof_eom_solve_ivp,
        [time_list[0], time_list[-1]],
        [u0, v0, r0, δ_list[0], npm_list[0]],
        dense_output=True,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )
    return sol


def get_sub_values_from_simulation_result(
    u_list: List[float],
    v_list: List[float],
    r_list: List[float],
    δ_list: List[float],
    npm_list: List[float],
    basic_params: Mmg3DofBasicParams,
    maneuvering_params: Mmg3DofManeuveringParams,
    ρ: float = 1.025,
    return_all_vals: bool = False,
):
    """Get sub values of MMG calculation from simulation result.

    Args:
        u_list (List[float]):
            u list of MMG simulation result.
        v_list (List[float]):
            v list of MMG simulation result.
        r_list (List[float]):
            r list of MMG simulation result.
        δ_list (List[float]):
            δ list of MMG simulation result.
        npm_list (List[float]):
            npm list of MMG simulation result.
        basic_params (Mmg3DofBasicParams):
            u of MMG simulation result.
        maneuvering_params (Mmg3DofManeuveringParams):
            u of MMG simulation result.
        ρ (float, optional):
            seawater density [kg/m^3]
            Defaults to 1.025.
        return_all_vals (bool, optional):
            Whether all sub values are returned or not.
            Defaults to false.
    Returns:
        X_H_list (List[float]): List of X_H
        X_R_list (List[float]): List of X_R
        X_P_list (List[float]): List of X_P
        Y_H_list (List[float]): List of Y_H
        Y_R_list (List[float]): List of Y_R
        N_H_list (List[float]): List of N_H
        N_R_list (List[float]): List of N_R
        U_list (List[float], optional): List of U if return_all_vals is True
        β_list (List[float], optional): List of β if return_all_vals is True
        v_dash_list (List[float], optional): List of v_dash if return_all_vals is True
        r_dash_list (List[float], optional): List of r_dash if return_all_vals is True
        w_P_list (List[float], optional): List of w_P if return_all_vals is True
        J_list (List[float], optional): List of J if return_all_vals is True
        K_T_list (List[float], optional): List of K_T if return_all_vals is True
        v_R_list (List[float], optional): List of v_R if return_all_vals is True
        u_R_list (List[float], optional): List of u_R if return_all_vals is True
        U_R_list (List[float], optional): List of U_R if return_all_vals is True
        α_R_list (List[float], optional): List of α_R if return_all_vals is True
        F_N_list (List[float], optional): List of F_N if return_all_vals is True
    """
    U_list = list(
        map(
            lambda u, v, r: np.sqrt(u ** 2 + (v - r * basic_params.x_G) ** 2),
            u_list,
            v_list,
            r_list,
        )
    )
    β_list = list(
        map(
            lambda U, v, r: 0.0
            if U == 0.0
            else np.arcsin(-(v - r * basic_params.x_G) / U),
            U_list,
            v_list,
            r_list,
        )
    )
    v_dash_list = list(map(lambda U, v: 0.0 if U == 0.0 else v / U, U_list, v_list))
    r_dash_list = list(
        map(lambda U, r: 0.0 if U == 0.0 else r * basic_params.L_pp / U, U_list, r_list)
    )
    β_P_list = list(
        map(
            lambda β, r_dash: β - basic_params.x_P * r_dash,
            β_list,
            r_dash_list,
        )
    )
    # w_P_list = [basic_params.w_P0 for i in range(len(r_dash_list))]
    w_P_list = list(
        map(lambda β_P: basic_params.w_P0 * np.exp(-4.0 * β_P ** 2), β_P_list)
    )
    J_list = list(
        map(
            lambda w_P, u, npm: 0.0
            if npm == 0.0
            else (1 - w_P) * u / (npm * basic_params.D_p),
            w_P_list,
            u_list,
            npm_list,
        )
    )
    K_T_list = list(
        map(
            lambda J: maneuvering_params.k_0
            + maneuvering_params.k_1 * J
            + maneuvering_params.k_2 * J ** 2,
            J_list,
        )
    )
    β_R_list = list(
        map(
            lambda β, r_dash: β - basic_params.l_R * r_dash,
            β_list,
            r_dash_list,
        )
    )
    γ_R_list = list(
        map(
            lambda β_R: basic_params.γ_R_minus if β_R < 0.0 else basic_params.γ_R_plus,
            β_R_list,
        )
    )
    v_R_list = list(
        map(
            lambda U, γ_R, β_R: U * γ_R * β_R,
            U_list,
            γ_R_list,
            β_R_list,
        )
    )
    u_R_list = list(
        map(
            lambda u, J, npm, K_T, w_P: np.sqrt(
                basic_params.η
                * (
                    basic_params.κ
                    * basic_params.ϵ
                    * 8.0
                    * maneuvering_params.k_0
                    * npm ** 2
                    * basic_params.D_p ** 4
                    / np.pi
                )
                ** 2
            )
            if J == 0.0
            else u
            * (1 - w_P)
            * basic_params.ϵ
            * np.sqrt(
                basic_params.η
                * (
                    1.0
                    + basic_params.κ * (np.sqrt(1.0 + 8.0 * K_T / (np.pi * J ** 2)) - 1)
                )
                ** 2
                + (1 - basic_params.η)
            ),
            u_list,
            J_list,
            npm_list,
            K_T_list,
            w_P_list,
        )
    )
    U_R_list = list(
        map(lambda u_R, v_R: np.sqrt(u_R ** 2 + v_R ** 2), u_R_list, v_R_list)
    )
    α_R_list = list(
        map(lambda δ, u_R, v_R: δ - np.arctan2(v_R, u_R), δ_list, u_R_list, v_R_list)
    )
    F_N_list = list(
        map(
            lambda U_R, α_R: 0.5
            * basic_params.A_R
            * ρ
            * basic_params.f_α
            * (U_R ** 2)
            * np.sin(α_R),
            U_R_list,
            α_R_list,
        )
    )
    X_H_list = list(
        map(
            lambda U, v_dash, r_dash: 0.5
            * ρ
            * basic_params.L_pp
            * basic_params.d
            * (U ** 2)
            * (
                -maneuvering_params.R_0_dash
                + maneuvering_params.X_vv_dash * (v_dash ** 2)
                + maneuvering_params.X_vr_dash * v_dash * r_dash
                + maneuvering_params.X_rr_dash * (r_dash ** 2)
                + maneuvering_params.X_vvvv_dash * (v_dash ** 4)
            ),
            U_list,
            v_dash_list,
            r_dash_list,
        )
    )
    X_R_list = list(
        map(lambda F_N, δ: -(1 - basic_params.t_R) * F_N * np.sin(δ), F_N_list, δ_list)
    )
    X_P_list = list(
        map(
            lambda K_T, npm: (1 - basic_params.t_P)
            * ρ
            * K_T
            * npm ** 2
            * basic_params.D_p ** 4,
            K_T_list,
            npm_list,
        )
    )
    Y_H_list = list(
        map(
            lambda U, v_dash, r_dash: 0.5
            * ρ
            * basic_params.L_pp
            * basic_params.d
            * (U ** 2)
            * (
                maneuvering_params.Y_v_dash * v_dash
                + maneuvering_params.Y_r_dash * r_dash
                + maneuvering_params.Y_vvv_dash * (v_dash ** 3)
                + maneuvering_params.Y_vvr_dash * (v_dash ** 2) * r_dash
                + maneuvering_params.Y_vrr_dash * v_dash * (r_dash ** 2)
                + maneuvering_params.Y_rrr_dash * (r_dash ** 3)
            ),
            U_list,
            v_dash_list,
            r_dash_list,
        )
    )
    Y_R_list = list(
        map(lambda F_N, δ: -(1 - basic_params.t_R) * F_N * np.cos(δ), F_N_list, δ_list)
    )
    N_H_list = list(
        map(
            lambda U, v_dash, r_dash: 0.5
            * ρ
            * (basic_params.L_pp ** 2)
            * basic_params.d
            * (U ** 2)
            * (
                maneuvering_params.N_v_dash * v_dash
                + maneuvering_params.N_r_dash * r_dash
                + maneuvering_params.N_vvv_dash * (v_dash ** 3)
                + maneuvering_params.N_vvr_dash * (v_dash ** 2) * r_dash
                + maneuvering_params.N_vrr_dash * v_dash * (r_dash ** 2)
                + maneuvering_params.N_rrr_dash * (r_dash ** 3)
            ),
            U_list,
            v_dash_list,
            r_dash_list,
        )
    )
    N_R_list = list(
        map(
            lambda F_N, δ: -(-0.5 + basic_params.a_H * basic_params.x_H)
            * F_N
            * np.cos(δ),
            F_N_list,
            δ_list,
        )
    )
    if return_all_vals:
        return (
            X_H_list,
            X_R_list,
            X_P_list,
            Y_H_list,
            Y_R_list,
            N_H_list,
            N_R_list,
            U_list,
            β_list,
            v_dash_list,
            r_dash_list,
            β_P_list,
            w_P_list,
            J_list,
            K_T_list,
            β_R_list,
            γ_R_list,
            v_R_list,
            u_R_list,
            U_R_list,
            α_R_list,
            F_N_list,
        )
    else:
        return (
            X_H_list,
            X_R_list,
            X_P_list,
            Y_H_list,
            Y_R_list,
            N_H_list,
            N_R_list,
        )


def zigzag_test_mmg_3dof(
    basic_params: Mmg3DofBasicParams,
    maneuvering_params: Mmg3DofManeuveringParams,
    target_δ_rad: float,
    target_ψ_rad_deviation: float,
    time_list: List[float],
    npm_list: List[float],
    δ0: float = 0.0,
    δ_rad_rate: float = 1.0 * np.pi / 180,
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    ψ0: float = 0.0,
    ρ: float = 1.025,
    method: str = "RK45",
    t_eval=None,
    events=None,
    vectorized=False,
    **options
):
    """Zig-zag test simulation

    Args:
        basic_params (Mmg3DofBasicParams):
            Basic paramters for MMG 3DOF simulation.
        maneuvering_params (Mmg3DofManeuveringParams):
            Maneuvering parameters for MMG 3DOF simulation.
        target_δ_rad (float):
            target absolute value of rudder angle.
        target_ψ_rad_deviation (float):
            target absolute value of psi deviation from ψ0[rad].
        time_list (list[float]):
            time list of simulation.
        npm_list (List[float]):
            npm list of simulation.
        δ0 (float):
            Initial rudder angle [rad].
            Defaults to 0.0.
        δ_rad_rate (float):
            Initial rudder angle rate [rad/s].
            Defaults to 1.0.
        u0 (float, optional):
            axial velocity [m/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        v0 (float, optional):
            lateral velocity [m/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        r0 (float, optional):
            rate of turn [rad/s] in initial condition (`time_list[0]`).
            Defaults to 0.0.
        ψ0 (float, optional):
            Inital azimuth [rad] in initial condition (`time_list[0]`)..
            Defaults to 0.0.
        ρ (float, optional):
            seawater density [kg/m^3]
            Defaults to 1.025.
        method (str, optional):
            Integration method to use in
            `scipy.integrate.solve_ivp()
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_:

                "RK45" (default):
                    Explicit Runge-Kutta method of order 5(4).
                    The error is controlled assuming accuracy of the fourth-order method,
                    but steps are taken using the fifth-order accurate formula (local extrapolation is done).
                    A quartic interpolation polynomial is used for the dense output.
                    Can be applied in the complex domain.
                "RK23":
                    Explicit Runge-Kutta method of order 3(2).
                    The error is controlled assuming accuracy of the second-order method,
                    but steps are taken using the third-order accurate formula (local extrapolation is done).
                    A cubic Hermite polynomial is used for the dense output.
                    Can be applied in the complex domain.
                "DOP853":
                    Explicit Runge-Kutta method of order 8.
                    Python implementation of the “DOP853” algorithm originally written in Fortran.
                    A 7-th order interpolation polynomial accurate to 7-th order is used for the dense output.
                    Can be applied in the complex domain.
                "Radau":
                    Implicit Runge-Kutta method of the Radau IIA family of order 5.
                    The error is controlled with a third-order accurate embedded formula.
                    A cubic polynomial which satisfies the collocation conditions is used for the dense output.
                "BDF":
                    Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the derivative approximation.
                    A quasi-constant step scheme is used and accuracy is enhanced using the NDF modification.
                    Can be applied in the complex domain.
                "LSODA":
                    Adams/BDF method with automatic stiffness detection and switching.
                    This is a wrapper of the Fortran solver from ODEPACK.

        t_eval (array_like or None, optional):
            Times at which to store the computed solution, must be sorted and lie within t_span.
            If None (default), use points selected by the solver.
        events (callable, or list of callables, optional):
            Events to track. If None (default), no events will be tracked.
            Each event occurs at the zeros of a continuous function of time and state.
            Each function must have the signature event(t, y) and return a float.
            The solver will find an accurate value of t at which event(t, y(t)) = 0 using a root-finding algorithm.
            By default, all zeros will be found. The solver looks for a sign change over each step,
            so if multiple zero crossings occur within one step, events may be missed.
            Additionally each event function might have the following attributes:
                terminal (bool, optional):
                    Whether to terminate integration if this event occurs. Implicitly False if not assigned.
                direction (float, optional):
                    Direction of a zero crossing.
                    If direction is positive, event will only trigger when going from negative to positive,
                    and vice versa if direction is negative.
                    If 0, then either direction will trigger event. Implicitly 0 if not assigned.
            You can assign attributes like `event.terminal = True` to any function in Python.
        vectorized (bool, optional):
            Whether `fun` is implemented in a vectorized fashion. Default is False.
        options:
            Options passed to a chosen solver.
            All options available for already implemented solvers are listed in
            `scipy.integrate.solve_ivp()
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_:

    Returns:
        final_δ_list (list[float])) : list of rudder angle.
        final_u_list (list[float])) : list of surge velocity.
        final_v_list (list[float])) : list of sway velocity.
        final_r_list (list[float])) : list of rate of turn.
    """
    target_ψ_rad_deviation = np.abs(target_ψ_rad_deviation)

    final_δ_list = [0.0] * len(time_list)
    final_u_list = [0.0] * len(time_list)
    final_v_list = [0.0] * len(time_list)
    final_r_list = [0.0] * len(time_list)

    next_stage_index = 0
    target_δ_rad = -target_δ_rad  # for changing in while loop
    ψ = ψ0

    while next_stage_index < len(time_list):
        target_δ_rad = -target_δ_rad
        start_index = next_stage_index

        # Make delta list
        δ_list = [0.0] * (len(time_list) - start_index)
        if start_index == 0:
            δ_list[0] = δ0
            u0 = u0
            v0 = v0
            r0 = r0
        else:
            δ_list[0] = final_δ_list[start_index - 1]
            u0 = final_u_list[start_index - 1]
            v0 = final_v_list[start_index - 1]
            r0 = final_r_list[start_index - 1]

        for i in range(start_index + 1, len(time_list)):
            Δt = time_list[i] - time_list[i - 1]
            if target_δ_rad > 0:
                δ = δ_list[i - 1 - start_index] + δ_rad_rate * Δt
                if δ >= target_δ_rad:
                    δ = target_δ_rad
                δ_list[i - start_index] = δ
            elif target_δ_rad <= 0:
                δ = δ_list[i - 1 - start_index] - δ_rad_rate * Δt
                if δ <= target_δ_rad:
                    δ = target_δ_rad
                δ_list[i - start_index] = δ

        sol = simulate_mmg_3dof(
            basic_params,
            maneuvering_params,
            time_list[start_index:],
            δ_list,
            npm_list[start_index:],
            u0=u0,
            v0=v0,
            r0=r0,
        )
        sim_result = sol.sol(time_list[start_index:])
        u_list = sim_result[0]
        v_list = sim_result[1]
        r_list = sim_result[2]
        ship = ShipObj3dof(L=basic_params.L_pp, B=basic_params.B)
        ship.load_simulation_result(time_list, u_list, v_list, r_list, psi0=ψ)

        # get finish index
        target_ψ_rad = ψ0 + target_ψ_rad_deviation
        if target_δ_rad < 0:
            target_ψ_rad = ψ0 - target_ψ_rad_deviation
        ψ_list = ship.psi
        bool_ψ_list = [True if ψ < target_ψ_rad else False for ψ in ψ_list]
        if target_δ_rad < 0:
            bool_ψ_list = [True if ψ > target_ψ_rad else False for ψ in ψ_list]
        over_index_list = [i for i, flag in enumerate(bool_ψ_list) if flag is False]
        next_stage_index = len(time_list)
        if len(over_index_list) > 0:
            ψ = ψ_list[over_index_list[0]]
            next_stage_index = over_index_list[0] + start_index
            final_δ_list[start_index:next_stage_index] = δ_list[: over_index_list[0]]
            final_u_list[start_index:next_stage_index] = u_list[: over_index_list[0]]
            final_v_list[start_index:next_stage_index] = v_list[: over_index_list[0]]
            final_r_list[start_index:next_stage_index] = r_list[: over_index_list[0]]
        else:
            final_δ_list[start_index:next_stage_index] = δ_list
            final_u_list[start_index:next_stage_index] = u_list
            final_v_list[start_index:next_stage_index] = v_list
            final_r_list[start_index:next_stage_index] = r_list

    return final_δ_list, final_u_list, final_v_list, final_r_list
