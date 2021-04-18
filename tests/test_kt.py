#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shipmmg.kt import KTParams, simulate_kt, zigzag_test_kt
from shipmmg.ship_obj_3dof import ShipObj3dof
import numpy as np
import pytest
import os


@pytest.fixture
def sim_result():
    """Just check shimmg.kt.simulate()."""
    K = 0.155
    T = 80.5
    kt_params = KTParams(K=K, T=T)
    duration = 50
    num_of_sampling = 1000
    time_list = np.linspace(0.00, duration, num_of_sampling)
    Ts = 50.0
    δ_list = 10 * np.pi / 180 * np.sin(2.0 * np.pi / Ts * time_list)
    result = simulate_kt(kt_params, time_list, δ_list, 0.0)
    return time_list, result


@pytest.fixture
def ship_kt(sim_result):
    """
    Fixture for testing in this file.
    """
    time_list, sol = sim_result
    simulation_result = sol.sol(time_list)
    u_list = np.full(len(time_list), 20 * (1852.0 / 3600))
    v_list = np.zeros(len(time_list))
    r_list = simulation_result[0]
    ship = ShipObj3dof(L=100, B=10)
    ship.load_simulation_result(time_list, u_list, v_list, r_list)
    return ship


def test_Ship3DOF_drawing_function(ship_kt):
    """Check drawing functions of Ship3DOF class by using KT simulation results"""

    # Ship3DOF.draw_xy_trajectory()
    save_fig_path = "test.png"
    ship_kt.draw_xy_trajectory(dimensionless=True, fmt="ro")
    ship_kt.draw_xy_trajectory(save_fig_path=save_fig_path)
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)

    # Ship3DOF.draw_chart()
    save_fig_path = "test.png"
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
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)

    x_index_list = ["time", "u", "v", "r", "x", "y", "psi"]
    y_index_list = ["time", "u", "v", "r", "x", "y", "psi"]
    for x_index in x_index_list:
        for y_index in y_index_list:
            ship_kt.draw_chart(x_index, y_index)

    with pytest.raises(Exception):
        ship_kt.draw_chart("time", "hogehoge")
    with pytest.raises(Exception):
        ship_kt.draw_chart("hogehoge", "y")

    # Ship3DOF.draw_gif()
    ship_kt.draw_gif(fmt=None, save_fig_path=save_fig_path)
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)
    ship_kt.draw_gif(dimensionless=True, save_fig_path=save_fig_path)
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)


def test_zigzag_test_kt():
    K = 0.155
    T = 80.5
    kt_params = KTParams(K=K, T=T)
    target_δ_rad = 30.0 * np.pi / 180.0
    target_ψ_rad_deviation = 10.0 * np.pi / 180.0
    duration = 500
    num_of_sampling = 50000
    time_list = np.linspace(0.00, duration, num_of_sampling)
    δ_list, r_list = zigzag_test_kt(
        kt_params,
        target_δ_rad,
        target_ψ_rad_deviation,
        time_list,
    )

    u_list = np.full(len(time_list), 20 * (1852.0 / 3600))
    v_list = np.zeros(len(time_list))
    ship = ShipObj3dof(L=100, B=10)
    ship.load_simulation_result(time_list, u_list, v_list, r_list)
    ship.δ = δ_list

    save_fig_path = "test.png"
    ship.draw_xy_trajectory(save_fig_path=save_fig_path)
    ship.draw_chart(
        "time",
        "psi",
        xlabel="time [sec]",
        ylabel=r"$\psi$" + " [rad]",
        save_fig_path=save_fig_path,
    )
    ship.draw_chart(
        "time",
        "r",
        xlabel="time [sec]",
        ylabel=r"$r$" + " [rad/s]",
        save_fig_path=save_fig_path,
    )
    ship.draw_chart(
        "time",
        "delta",
        xlabel="time [sec]",
        ylabel=r"$\delta$" + " [rad]",
        save_fig_path=save_fig_path,
    )
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)
