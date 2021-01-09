#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shipmmg.kt import simulate
from shipmmg.ship_3dof import Ship3DOF
import numpy as np
import pytest
import os


@pytest.fixture
def simulate_kt():
    """Just check shimmg.kt.simulate()."""
    K = 0.155
    T = 80.5
    duration = 500
    num_of_sampling = 10000
    time_list = np.linspace(0.00, duration, num_of_sampling)
    Ts = 50.0
    delta_list = 10 * np.pi / 180 * np.sin(2.0 * np.pi / Ts * time_list)
    result = simulate(K, T, time_list, delta_list, 0.0)
    return time_list, result


@pytest.fixture
def ship_kt(simulate_kt):
    """
    Fixture for testing in this file.
    """
    time_list, simulation_result = simulate_kt
    u_list = np.full(len(time_list), 20 * (1852.0 / 3600))
    v_list = np.zeros(len(time_list))
    r_list = simulation_result.T[0]
    ship = Ship3DOF()
    ship.load_simulation_result(time_list, u_list, v_list, r_list)
    return ship


def test_Ship3DOF_drawing_function(ship_kt):
    """Check drawing functions of Ship3DOF class by using KT simulation results"""

    # Ship3DOF.draw_xy_trajectory()
    save_fig_path = "test.png"
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
