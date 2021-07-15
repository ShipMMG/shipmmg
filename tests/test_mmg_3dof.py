#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_mmg_3dof.

- pytest code of shipmmg/mmg_3dof.py
"""

import numpy as np

import pytest

from shipmmg.mmg_3dof import (
    get_sub_values_from_simulation_result,
    simulate_mmg_3dof,
    zigzag_test_mmg_3dof,
)
from shipmmg.ship_obj_3dof import ShipObj3dof


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

    sol = simulate_mmg_3dof(
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
    ship = ShipObj3dof(L=basic_params.L_pp, B=basic_params.B)
    ship.load_simulation_result(
        time_list,
        sim_result[0],
        sim_result[1],
        sim_result[2],
        δ=δ_rad_list,
        npm=npm_list,
    )
    return ship


def test_get_sub_values_from_simulation_result(
    kvlcc2_L7_35_turning, ship_KVLCC2_L7_model, tmpdir
):
    """Test get_sub_values_from_simulation_result() using KVLCC2 L7 model."""
    basic_params, maneuvering_params = ship_KVLCC2_L7_model
    (
        X_H_list,
        X_R_list,
        X_P_list,
        Y_H_list,
        Y_R_list,
        N_H_list,
        N_R_list,
    ) = get_sub_values_from_simulation_result(
        kvlcc2_L7_35_turning.u,
        kvlcc2_L7_35_turning.v,
        kvlcc2_L7_35_turning.r,
        kvlcc2_L7_35_turning.δ,
        kvlcc2_L7_35_turning.npm,
        basic_params,
        maneuvering_params,
    )
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
    ) = get_sub_values_from_simulation_result(
        kvlcc2_L7_35_turning.u,
        kvlcc2_L7_35_turning.v,
        kvlcc2_L7_35_turning.r,
        kvlcc2_L7_35_turning.δ,
        kvlcc2_L7_35_turning.npm,
        basic_params,
        maneuvering_params,
        return_all_vals=True,
    )


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

    δ_list, u_list, v_list, r_list = zigzag_test_mmg_3dof(
        basic_params,
        maneuvering_params,
        target_δ_rad,
        target_ψ_rad_deviation,
        time_list,
        npm_list,
        δ_rad_rate=15.0 * np.pi / 180,
    )
