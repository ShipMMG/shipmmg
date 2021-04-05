#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from typing import List
import numpy as np
from scipy.misc import derivative
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


@dataclasses.dataclass
class Mmg3DofBasicParams:
    """Dataclass for setting basic parameters of MMG 3ODF.

    Attributes:
        L_pp (float):
            Ship length between perpendiculars [m]
        B (float):
            Ship breadth [m]
        d (float):
            Ship draft [m]
        nabla (float):
            Displacement volume of ship [m^3]
        x_G (float):
            Longitudinal coordinate of center of gravity of ship [-]
        D_p (float):
            Propeller diameter [m]
        m_ (float):
            Ship mass (non-dimensionalized) [-]
        I_zG (float):
            Moment of inertia of ship around center of gravity [-]
        Λ (float):
            Rudder aspect ratio [-]
        A_R_Ld (float):
            Profile area of movable part of mariner rudder[-]
        η (float):
            Ratio of propeller diameter to rudder span (=D_p/HR)
        m_x_ (float):
            Added masses of x axis direction (non-dimensionalized)
        m_y_ (float):
            Added masses of y axis direction (non-dimensionalized)
        J_z (float):
            Added moment of inertia (non-dimensionalized)
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
        γ_R (float):
            Flow straightening coefficient
        l_R (float):
            Effective longitudinal coordinate of rudder position in formula of βR
        κ (float):
            An experimental constant for　expressing uR
        t_P (float):
            Thrust deduction factor
        w_P0 (float):
            Wake coefficient at propeller position in straight moving

    Note:
        For more information, please see the following articles.

        - Yasukawa, H., Yoshimura, Y. (2015) Introduction of MMG standard method for ship maneuvering predictions.
          J Mar Sci Technol 20, 37–52 https://doi.org/10.1007/s00773-014-0293-y
    """

    L_pp: float
    B: float
    d: float
    nabla: float
    x_G: float
    D_p: float
    m_: float
    I_zG: float
    Λ: float
    A_R_Ld: float
    η: float
    m_x_: float
    m_y_: float
    J_z: float
    f_α: float
    ϵ: float
    t_R: float
    a_H: float
    x_H: float
    γ_R: float
    l_R: float
    κ: float
    t_P: float
    w_P0: float


@dataclasses.dataclass
class Mmg3DofManeuveringParams:
    """Dataclass for setting maneuvering parameters of MMG 3ODF.

    Attributes:
        C_1 (float): One of manuevering parameters of MMG 3DOF
        C_2 (float): One of manuevering parameters of MMG 3DOF
        C_3 (float): One of manuevering parameters of MMG 3DOF
        X_0 (float): One of manuevering parameters of MMG 3DOF
        X_ββ (float): One of manuevering parameters of MMG 3DOF
        X_βγ (float): One of manuevering parameters of MMG 3DOF
        X_γγ (float): One of manuevering parameters of MMG 3DOF
        X_ββββ (float): One of manuevering parameters of MMG 3DOF
        Y_β (float): One of manuevering parameters of MMG 3DOF
        Y_γ (float): One of manuevering parameters of MMG 3DOF
        Y_βββ (float): One of manuevering parameters of MMG 3DOF
        Y_ββγ (float): One of manuevering parameters of MMG 3DOF
        Y_βγγ (float): One of manuevering parameters of MMG 3DOF
        Y_γγγ (float): One of manuevering parameters of MMG 3DOF
        N_β (float): One of manuevering parameters of MMG 3DOF
        N_γ (float): One of manuevering parameters of MMG 3DOF
        N_βββ (float): One of manuevering parameters of MMG 3DOF
        N_ββγ (float): One of manuevering parameters of MMG 3DOF
        N_βγγ (float): One of manuevering parameters of MMG 3DOF
        N_γγγ (float): One of manuevering parameters of MMG 3DOF

    Note:
        For more information, please see the following articles.

        - Yasukawa, H., Yoshimura, Y. (2015) Introduction of MMG standard method for ship maneuvering predictions.
          J Mar Sci Technol 20, 37–52 https://doi.org/10.1007/s00773-014-0293-y
    """

    C_1: float
    C_2: float
    C_3: float
    X_0: float
    X_ββ: float
    X_βγ: float
    X_γγ: float
    X_ββββ: float
    Y_β: float
    Y_γ: float
    Y_βββ: float
    Y_ββγ: float
    Y_βγγ: float
    Y_γγγ: float
    N_β: float
    N_γ: float
    N_βββ: float
    N_ββγ: float
    N_βγγ: float
    N_γγγ: float


def simulate_mmg_3dof(
    basic_params: Mmg3DofBasicParams,
    maneuvering_params: Mmg3DofManeuveringParams,
    R_0_func: interp1d,
    time_list: List[float],
    delta_list: List[float],
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

        m (\\dot{u}-vr)&=-m_x\\dot{u}+m_yvr+X_H+R_0+X_P+X_R

        m (\\dot{v}+ur)&=-m_y\\dot{v}+m_xur+Y_H+Y_R

        I_{zG}\\dot{r}&=-J_Z\\dot{r}+N_H+N_R

    Args:
        basic_params (Mmg3DofBasicParams):
            Basic paramters for MMG 3DOF simulation.
        maneuvering_params (Mmg3DofManeuveringParams):
            Maneuvering parameters for MMG 3DOF simulation.
        R_0_func (scipy.interpolate.interp1d):
            R_0 function which input value is `u`.
        time_list (list[float]):
            time list of simulation.
        delta_list (list[float]):
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
        >>> duration = 50  # [s]
        >>> sampling = 2000
        >>> time_list = np.linspace(0.00, duration, sampling)
        >>> delta_list = np.full(len(time_list), 30.0 * np.pi / 180.0)
        >>> npm_list = np.full(len(time_list), 20.338)
        >>> R0_func = scipy.interpolate.interp1d(np.linspace(0.0, 10.0, 10), np.full(10, 0.0))
        >>> basic_params = Mmg3DofBasicParams(
        >>>                     L_pp=2.19048,
        >>>                     B=0.3067,
        >>>                     d=0.10286,
        >>>                     nabla=0.04495,
        >>>                     x_G=0.0,
        >>>                     D_p=0.0756,
        >>>                     m_=0.1822,
        >>>                     I_zG=0.01138,
        >>>                     Λ=2.1683,
        >>>                     A_R_Ld=0.01867,
        >>>                     η=0.7916,
        >>>                     m_x_=0.00601,
        >>>                     m_y_=0.1521,
        >>>                     J_z=0.00729,
        >>>                     f_α=(6.13 * 2.1683) / (2.25 + 2.1683),
        >>>                     ϵ=0.90,
        >>>                     t_R=0.441,
        >>>                     a_H=0.232,
        >>>                     x_H=-0.711,
        >>>                     γ_R=0.4115,
        >>>                     l_R=-0.774,
        >>>                     κ=0.713,
        >>>                     t_P=0.20,
        >>>                     w_P0=0.326,
        >>>                 )
        >>> maneuvering_params = Mmg3DofManeuveringParams(
        >>>                    C_1=0.48301,
        >>>                    C_2=-0.29765,
        >>>                    C_3=-0.16423,
        >>>                    X_0=-0.07234,
        >>>                    X_ββ=-0.23840,
        >>>                    X_βγ=-0.03231 + 0.1521,
        >>>                    X_γγ=-0.06405,
        >>>                    X_ββββ=-0.30047,
        >>>                    Y_β=0.85475,
        >>>                    Y_γ=0.11461 + 0.00601,
        >>>                    Y_βββ=6.73201,
        >>>                    Y_ββγ=-2.23689,
        >>>                    Y_βγγ=3.38577,
        >>>                    Y_γγγ=-0.04151,
        >>>                    N_β=0.096567,
        >>>                    N_γ=-0.036501,
        >>>                    N_βββ=0.14090,
        >>>                    N_ββγ=-0.46158,
        >>>                    N_βγγ=0.01648,
        >>>                    N_γγγ=-0.030404,
        >>>                )
        >>> sol = simulate_mmg_3dof(
        >>>                    basic_params,
        >>>                    maneuvering_params,
        >>>                    R0_func,
        >>>                    time_list,
        >>>                    delta_rad_list,
        >>>                    npm_list,
        >>>                    u0=1.21,
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
        nabla=basic_params.nabla,
        x_G=basic_params.x_G,
        D_p=basic_params.D_p,
        m_=basic_params.m_,
        I_zG=basic_params.I_zG,
        Λ=basic_params.Λ,
        A_R_Ld=basic_params.A_R_Ld,
        η=basic_params.η,
        m_x_=basic_params.m_x_,
        m_y_=basic_params.m_y_,
        J_z=basic_params.J_z,
        f_α=basic_params.f_α,
        ϵ=basic_params.ϵ,
        t_R=basic_params.t_R,
        a_H=basic_params.a_H,
        x_H=basic_params.x_H,
        γ_R=basic_params.γ_R,
        l_R=basic_params.l_R,
        κ=basic_params.κ,
        t_P=basic_params.t_P,
        w_P0=basic_params.w_P0,
        C_1=maneuvering_params.C_1,
        C_2=maneuvering_params.C_2,
        C_3=maneuvering_params.C_3,
        X_0=maneuvering_params.X_0,
        X_ββ=maneuvering_params.X_ββ,
        X_βγ=maneuvering_params.X_βγ,
        X_γγ=maneuvering_params.X_γγ,
        X_ββββ=maneuvering_params.X_ββββ,
        Y_β=maneuvering_params.Y_β,
        Y_γ=maneuvering_params.Y_γ,
        Y_βββ=maneuvering_params.Y_βββ,
        Y_ββγ=maneuvering_params.Y_ββγ,
        Y_βγγ=maneuvering_params.Y_βγγ,
        Y_γγγ=maneuvering_params.Y_γγγ,
        N_β=maneuvering_params.N_β,
        N_γ=maneuvering_params.N_γ,
        N_βββ=maneuvering_params.N_βββ,
        N_ββγ=maneuvering_params.N_ββγ,
        N_βγγ=maneuvering_params.N_βγγ,
        N_γγγ=maneuvering_params.N_γγγ,
        R_0_func=R_0_func,
        time_list=time_list,
        delta_list=delta_list,
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
    nabla: float,
    x_G: float,
    D_p: float,
    m_: float,
    I_zG: float,
    Λ: float,
    A_R_Ld: float,
    η: float,
    m_x_: float,
    m_y_: float,
    J_z: float,
    f_α: float,
    ϵ: float,
    t_R: float,
    a_H: float,
    x_H: float,
    γ_R: float,
    l_R: float,
    κ: float,
    t_P: float,
    w_P0: float,
    C_1: float,
    C_2: float,
    C_3: float,
    X_0: float,
    X_ββ: float,
    X_βγ: float,
    X_γγ: float,
    X_ββββ: float,
    Y_β: float,
    Y_γ: float,
    Y_βββ: float,
    Y_ββγ: float,
    Y_βγγ: float,
    Y_γγγ: float,
    N_β: float,
    N_γ: float,
    N_βββ: float,
    N_ββγ: float,
    N_βγγ: float,
    N_γγγ: float,
    R_0_func: interp1d,
    time_list: List[float],
    delta_list: List[float],
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

        m (\\dot{u}-vr)&=-m_x\\dot{u}+m_yvr+X_H+R_0+X_P+X_R

        m (\\dot{v}+ur)&=-m_y\\dot{v}+m_xur+Y_H+Y_R

        I_{zG}\\dot{r}&=-J_Z\\dot{r}+N_H+N_R

    Args:
        L_pp (float):
            Ship length between perpendiculars [m]
        B (float):
            Ship breadth [m]
        d (float):
            Ship draft [m]
        nabla (float):
            Displacement volume of ship [m^3]
        x_G (float):
            Longitudinal coordinate of center of gravity of ship [-]
        D_p (float):
            Propeller diameter [m]
        m_ (float):
            Ship mass (non-dimensionalized) [-]
        I_zG (float):
            Moment of inertia of ship around center of gravity [-]
        Λ (float):
            Rudder aspect ratio [-]
        A_R_Ld (float):
            Profile area of movable part of mariner rudder[-]
        η (float):
            Ratio of propeller diameter to rudder span (=D_p/HR)
        m_x_ (float):
            Added masses of x axis direction (non-dimensionalized)
        m_y_ (float):
            Added masses of y axis direction (non-dimensionalized)
        J_z (float):
            Added moment of inertia (non-dimensionalized)
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
        γ_R (float):
            Flow straightening coefficient
        l_R (float):
            Effective longitudinal coordinate of rudder position in formula of βR
        κ (float):
            An experimental constant for　expressing uR
        t_P (float):
            Thrust deduction factor
        w_P0 (float):
            Wake coefficient at propeller position in straight moving
        C_1 (float):
            One of manuevering parameters of MMG 3DOF
        C_2 (float):
            One of manuevering parameters of MMG 3DOF
        C_3 (float):
            One of manuevering parameters of MMG 3DOF
        X_0 (float):
            One of manuevering parameters of MMG 3DOF
        X_ββ (float):
            One of manuevering parameters of MMG 3DOF
        X_βγ (float):
            One of manuevering parameters of MMG 3DOF
        X_γγ (float):
            One of manuevering parameters of MMG 3DOF
        X_ββββ (float):
            One of manuevering parameters of MMG 3DOF
        Y_β (float):
            One of manuevering parameters of MMG 3DOF
        Y_γ (float):
            One of manuevering parameters of MMG 3DOF
        Y_βββ (float):
            One of manuevering parameters of MMG 3DOF
        Y_ββγ (float):
            One of manuevering parameters of MMG 3DOF
        Y_βγγ (float):
            One of manuevering parameters of MMG 3DOF
        Y_γγγ (float):
            One of manuevering parameters of MMG 3DOF
        N_β (float):
            One of manuevering parameters of MMG 3DOF
        N_γ (float):
            One of manuevering parameters of MMG 3DOF
        N_βββ (float):
            One of manuevering parameters of MMG 3DOF
        N_ββγ (float):
            One of manuevering parameters of MMG 3DOF
        N_βγγ (float):
            One of manuevering parameters of MMG 3DOF
        N_γγγ (float):
            One of manuevering parameters of MMG 3DOF
        R_0_func (scipy.interpolate.interp1d):
            R_0 function which input value is `u`.
        time_list (list[float]):
            time list of simulation.
        delta_list (list[float]):
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
        >>> duration = 50  # [s]
        >>> sampling = 2000
        >>> time_list = np.linspace(0.00, duration, sampling)
        >>> delta_list = np.full(len(time_list), 30.0 * np.pi / 180.0)
        >>> npm_list = np.full(len(time_list), 20.338)
        >>> R0_func = scipy.interpolate.interp1d(np.linspace(0.0, 10.0, 10), np.full(10, 0.0))
        >>> L_pp=2.19048
        >>> B=0.3067
        >>> d=0.10286
        >>> nabla=0.04495
        >>> x_G=0.0
        >>> D_p=0.0756
        >>> m_=0.1822
        >>> I_zG=0.01138
        >>> Λ=2.1683
        >>> A_R_Ld=0.01867
        >>> η=0.7916
        >>> m_x_=0.00601
        >>> m_y_=0.1521
        >>> J_z=0.00729
        >>> f_α=(6.13 * 2.1683) / (2.25 + 2.1683)
        >>> ϵ=0.90
        >>> t_R=0.441
        >>> a_H=0.232
        >>> x_H=-0.711
        >>> γ_R=0.4115
        >>> l_R=-0.774
        >>> κ=0.713
        >>> t_P=0.20
        >>> w_P0=0.326
        >>> C_1=0.48301
        >>> C_2=-0.29765
        >>> C_3=-0.16423
        >>> X_0=-0.07234
        >>> X_ββ=-0.23840
        >>> X_βγ=-0.03231 + 0.1521
        >>> X_γγ=-0.06405
        >>> X_ββββ=-0.30047
        >>> Y_β=0.85475
        >>> Y_γ=0.11461 + 0.00601
        >>> Y_βββ=6.73201
        >>> Y_ββγ=-2.23689
        >>> Y_βγγ=3.38577
        >>> Y_γγγ=-0.04151
        >>> N_β=0.096567
        >>> N_γ=-0.036501
        >>> N_βββ=0.14090
        >>> N_ββγ=-0.46158
        >>> N_βγγ=0.01648
        >>> N_γγγ=-0.030404
        >>> sol = simulate_mmg_3dof(
        >>>                    L_pp=L_pp,
        >>>                    B=B,
        >>>                    d=d,
        >>>                    nabla=nabla,
        >>>                    x_G=x_G,
        >>>                    D_p=D_p,
        >>>                    m_=m_,
        >>>                    I_zG=I_zG,
        >>>                    Λ=Λ,
        >>>                    A_R_Ld=A_R_Ld,
        >>>                    η=η,
        >>>                    m_x_=m_x_,
        >>>                    m_y_=m_y_,
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
        >>>                    C_1=C_1,
        >>>                    C_2=C_2,
        >>>                    C_3=C_3,
        >>>                    X_0=X_0,
        >>>                    X_ββ=X_ββ,
        >>>                    X_βγ=X_βγ,
        >>>                    X_γγ=X_γγ,
        >>>                    X_ββββ=X_ββββ,
        >>>                    Y_β=Y_β,
        >>>                    Y_γ=Y_γ,
        >>>                    Y_βββ=Y_βββ,
        >>>                    Y_ββγ=Y_ββγ,
        >>>                    Y_βγγ=Y_βγγ,
        >>>                    Y_γγγ=Y_γγγ,
        >>>                    N_β=N_β,
        >>>                    N_γ=N_γ,
        >>>                    N_βββ=N_βββ,
        >>>                    N_ββγ=N_ββγ,
        >>>                    N_βγγ=N_βγγ,
        >>>                    N_γγγ=N_γγγ,
        >>>                    R0_func,
        >>>                    time_list,
        >>>                    delta_rad_list,
        >>>                    npm_list,
        >>>                    u0=1.21,
        >>>                )
        >>> result = sol.sol(time_list)

    Note:
        For more information, please see the following articles.

        - Yasukawa, H., Yoshimura, Y. (2015) Introduction of MMG standard method for ship maneuvering predictions.
          J Mar Sci Technol 20, 37–52 https://doi.org/10.1007/s00773-014-0293-y

    """
    spl_delta = interp1d(time_list, delta_list, "cubic", fill_value="extrapolate")
    spl_npm = interp1d(time_list, npm_list, "cubic", fill_value="extrapolate")

    def mmg_3dof_eom_solve_ivp(t, X):

        u, v, r, δ, npm = X

        U = np.sqrt(u ** 2 + (v - r * x_G) ** 2)
        β = 0.0 if U == 0.0 else np.arcsin(-(v - r * x_G) / U)

        γ_dash = 0.0 if U == 0.0 else r * L_pp / U
        J = 0.0 if npm == 0.0 else (1 - w_P0) * u / (npm * D_p)
        K_T = C_1 + C_2 * J + C_3 * J ** 2
        v_R = U * γ_R * (np.sin(β) - l_R * γ_dash)
        u_R = (
            np.sqrt(η * (κ * ϵ * 8.0 * C_1 * npm ** 2 * D_p ** 4 / np.pi) ** 2)
            if J == 0.0
            else u
            * (1 - w_P0)
            * ϵ
            * np.sqrt(
                η * (1.0 + κ * (np.sqrt(1.0 + 8.0 * K_T / (np.pi * J ** 2)) - 1)) ** 2
                + (1 - η)
            )
        )
        U_R = np.sqrt(u_R ** 2 + v_R ** 2)
        α_R = δ - np.arctan2(v_R, u_R)
        F_N = A_R_Ld * f_α * (U_R ** 2) * np.sin(α_R)

        X_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U ** 2)
            * (
                X_0
                + X_ββ * β ** 2
                + X_βγ * β * γ_dash
                + X_γγ * γ_dash ** 2
                + X_ββββ * β ** 4
            )
        )
        R_0 = R_0_func(u)
        X_R = -(1 - t_R) * F_N * np.sin(δ) / L_pp
        X_P = (1 - t_P) * ρ * K_T * npm ** 2 * D_p ** 4 * (2 / (ρ * d * L_pp ** 2))
        Y_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U ** 2)
            * (
                Y_β * β
                + Y_γ * γ_dash
                + Y_ββγ * (β ** 2) * γ_dash
                + Y_βγγ * β * (γ_dash ** 2)
                + Y_βββ * (β ** 3)
                + Y_γγγ * (γ_dash ** 3)
            )
        )
        Y_R = -(1 + a_H) * F_N * np.cos(δ) / L_pp
        N_H = (
            0.5
            * ρ
            * (L_pp ** 2)
            * d
            * (U ** 2)
            * (
                N_β * β
                + N_γ * γ_dash
                + N_ββγ * (β ** 2) * γ_dash
                + N_βγγ * β * (γ_dash ** 2)
                + N_βββ * (β ** 3)
                + N_γγγ * (γ_dash ** 3)
            )
        )
        N_R = -(-0.5 + a_H * x_H) * F_N * np.cos(δ) / L_pp ** 2
        d_u = ((X_H - R_0 + X_R + X_P) + (m_ + m_y_) * v * r) / (m_ + m_x_)
        d_v = ((Y_H + Y_R) - (m_ + m_x_) * u * r) / (m_ + m_y_)
        d_r = (N_H + N_R) / (I_zG + J_z)
        d_δ = derivative(spl_delta, t)
        d_npm = derivative(spl_npm, t)
        return [d_u, d_v, d_r, d_δ, d_npm]

    sol = solve_ivp(
        mmg_3dof_eom_solve_ivp,
        [time_list[0], time_list[-1]],
        [u0, v0, r0, delta_list[0], npm_list[0]],
        dense_output=True,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )
    return sol
