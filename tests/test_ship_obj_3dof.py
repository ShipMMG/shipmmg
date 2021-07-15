#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_ship_obj_3dof.

- pytest code of shipmmg/ship_obj_3dof.py
"""

import os

import matplotlib.pyplot as plt

import numpy as np

import pytest

import shipmmg


@pytest.fixture
def ship_kt(test_kt_params):
    """Fixture for testing in this file."""
    duration = 50
    num_of_sampling = 1000
    time_list = np.linspace(0.00, duration, num_of_sampling)
    Ts = 50.0
    δ_list = 10 * np.pi / 180 * np.sin(2.0 * np.pi / Ts * time_list)
    sol = shipmmg.kt.simulate_kt(test_kt_params, time_list, δ_list, 0.0)
    assert sol.success
    simulation_result = sol.sol(time_list)
    u_list = np.full(len(time_list), 20 * (1852.0 / 3600))
    v_list = np.zeros(len(time_list))
    r_list = simulation_result[0]
    ship = shipmmg.ship_obj_3dof.ShipObj3dof(L=100, B=10)
    ship.load_simulation_result(time_list, u_list, v_list, r_list, δ=δ_list)
    return ship


def test_estimate_KT_by_LSM(ship_kt):
    """Check shipmmg.ship_obj_3dof.estimate_KT_LSM()."""
    K, T = ship_kt.estimate_KT_by_LSM()


def test_estimate_KT_by_LSM_RSR(ship_kt):
    """Check shipmmg.ship_obj_3dof.estimate_KT_LSM_RSR()."""
    K_list, T_list = ship_kt.estimate_KT_by_LSM_RSR(5, 10)


def test_Ship3DOF_drawing_function_kt(ship_kt, tmpdir):
    """Check drawing functions of Ship3DOF class by using KT simulation results."""
    # Ship3DOF.draw_xy_trajectory()
    save_fig_path = os.path.join(str(tmpdir), "test.png")
    ship_kt.draw_xy_trajectory(dimensionless=True, fmt="ro")
    ship_kt.draw_xy_trajectory(save_fig_path=save_fig_path)

    # Ship3DOF.draw_chart()
    save_fig_path = os.path.join(str(tmpdir), "test.png")
    ship_kt.draw_chart(
        "time",
        "u",
        xlabel="time [sec]",
        ylabel=r"$u$" + " [m/s]",
        save_fig_path=save_fig_path,
    )
    ship_kt.draw_chart(
        "time",
        "u",
        xlabel="time [sec]",
        ylabel=r"$u$" + " [m/s]",
        fmt="ro",
        save_fig_path=save_fig_path,
    )

    x_index_list = ["time", "u", "v", "r", "x", "y", "psi", "delta"]
    y_index_list = ["time", "u", "v", "r", "x", "y", "psi", "delta"]
    for x_index in x_index_list:
        for y_index in y_index_list:
            ship_kt.draw_chart(x_index, y_index)

    with pytest.raises(Exception):
        ship_kt.draw_chart("time", "hogehoge")
    with pytest.raises(Exception):
        ship_kt.draw_chart("hogehoge", "y")

    # Ship3DOF.draw_gif()
    ship_kt.draw_gif(fmt=None, save_fig_path=save_fig_path, obj="square")
    ship_kt.draw_gif(dimensionless=True, save_fig_path=save_fig_path)


@pytest.fixture
def kvlcc2_L7_35_turning(ship_KVLCC2_L7_model):
    """Do turning test using KVLCC2 L7 model."""
    basic_params, maneuvering_params = ship_KVLCC2_L7_model
    duration = 200  # [s]
    # steering_rate = 1.76 * 4  # [°/s]
    max_δ_rad = 35 * np.pi / 180.0  # [rad]
    n_const = 17.95  # [rpm]

    sampling = duration * 10
    time_list = np.linspace(0.00, duration, sampling)
    δ_rad_list = [0] * sampling
    for i in range(sampling):
        δ_rad_list[i] = max_δ_rad

    npm_list = np.array([n_const for i in range(sampling)])

    sol = shipmmg.mmg_3dof.simulate_mmg_3dof(
        basic_params,
        maneuvering_params,
        time_list,
        δ_rad_list,
        npm_list,
        u0=2.29 * 0.512,
        v0=0.0,
        r0=0.0,
    )
    assert sol.success

    sim_result = sol.sol(time_list)
    ship = shipmmg.mmg_3dof.ShipObj3dof(L=basic_params.L_pp, B=basic_params.B)
    ship.load_simulation_result(
        time_list,
        sim_result[0],
        sim_result[1],
        sim_result[2],
        δ=δ_rad_list,
        npm=npm_list,
    )
    return ship


def test_Ship3DOF_drawing_function_mmg3dof(kvlcc2_L7_35_turning, tmpdir):
    """Check drawing functions of Ship3DOF class by using MMG 3DOF simulation results."""
    # Ship3DOF.draw_xy_trajectory()
    save_fig_path = os.path.join(str(tmpdir), "trajectory.png")

    kvlcc2_L7_35_turning.draw_xy_trajectory(dimensionless=True)
    kvlcc2_L7_35_turning.draw_xy_trajectory(save_fig_path=save_fig_path)

    # Ship3DOF.draw_chart()
    save_fig_path = os.path.join(str(tmpdir), "param.png")

    kvlcc2_L7_35_turning.draw_chart(
        "time",
        "u",
        xlabel="time [sec]",
        ylabel=r"$u$" + " [m/s]",
        save_fig_path=save_fig_path,
    )

    x_index_list = ["time", "u", "v", "r", "x", "y", "psi", "delta", "npm"]
    y_index_list = ["time", "u", "v", "r", "x", "y", "psi", "delta", "npm"]
    for x_index in x_index_list:
        for y_index in y_index_list:
            kvlcc2_L7_35_turning.draw_chart(x_index, y_index)

    with pytest.raises(Exception):
        kvlcc2_L7_35_turning.draw_chart("time", "hogehoge")
    with pytest.raises(Exception):
        kvlcc2_L7_35_turning.draw_chart("hogehoge", "y")

    # Ship3DOF.draw_gif()
    save_fig_path = os.path.join(str(tmpdir), "test.gif")

    kvlcc2_L7_35_turning.draw_gif(save_fig_path=save_fig_path)

    kvlcc2_L7_35_turning.draw_gif(dimensionless=True, save_fig_path=save_fig_path)


def test_zigzag_test_mmg(ship_KVLCC2_L7_model, tmpdir):
    """Test zigzag test mmg simulation using KVLCC2 L7 model."""
    basic_params, maneuvering_params = ship_KVLCC2_L7_model
    target_δ_rad = 20.0 * np.pi / 180.0
    target_ψ_rad_deviation = 20.0 * np.pi / 180.0
    duration = 80
    num_of_sampling = 10000
    time_list = np.linspace(0.00, duration, num_of_sampling)
    n_const = 17.95  # [rpm]
    npm_list = np.array([n_const for i in range(num_of_sampling)])

    δ_list, u_list, v_list, r_list = shipmmg.mmg_3dof.zigzag_test_mmg_3dof(
        basic_params,
        maneuvering_params,
        target_δ_rad,
        target_ψ_rad_deviation,
        time_list,
        npm_list,
        δ_rad_rate=15.0 * np.pi / 180,
    )

    ship = shipmmg.ship_obj_3dof.ShipObj3dof(L=100, B=10)
    ship.load_simulation_result(time_list, u_list, v_list, r_list)
    ship.δ = δ_list
    ship.npm = npm_list

    save_fig_path = os.path.join(str(tmpdir), "delta_psi.png")

    fig = plt.figure()
    plt.plot(time_list, list(map(lambda δ: δ * 180 / np.pi, ship.δ)))
    plt.plot(time_list, list(map(lambda psi: psi * 180 / np.pi, ship.psi)))
    fig.savefig(save_fig_path)
    plt.close()

    (
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
    ) = shipmmg.mmg_3dof.get_sub_values_from_simulation_result(
        ship.u,
        ship.v,
        ship.r,
        ship.δ,
        ship.npm,
        basic_params,
        maneuvering_params,
        return_all_vals=True,
    )

    save_fig_path = os.path.join(str(tmpdir), "w_P.png")
    fig = plt.figure()
    plt.plot(time_list, w_P_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir), "J.png")
    fig = plt.figure()
    plt.plot(time_list, J_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir), "K_T.png")

    fig = plt.figure()
    plt.plot(time_list, K_T_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir), "U_R.png")
    fig = plt.figure()
    plt.plot(time_list, U_R_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir), "α_R.png")
    fig = plt.figure()
    plt.plot(time_list, α_R_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir), "F_N.png")
    fig = plt.figure()
    plt.plot(time_list, F_N_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir), "gamma_R.png")
    fig = plt.figure()
    plt.plot(time_list, γ_R_list)
    fig.savefig(save_fig_path)
    plt.close()
