#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shipmmg.mmg_3dof import (
    Mmg3DofBasicParams,
    Mmg3DofManeuveringParams,
    simulate_mmg_3dof,
)
from shipmmg.ship_obj_3dof import ShipObj3dof
import numpy as np
from scipy.interpolate import interp1d
import pytest
import os


@pytest.fixture
def basic_params():
    basic_params = Mmg3DofBasicParams(
        L_pp=2.19048,  # 船長Lpp[m]
        B=0.3067,  # 船幅[m]
        d=0.10286,  # 喫水[m]
        nabla=0.04495,  # 排水量[m^3]
        x_G=0.0,  # 重心位置[]
        D_p=0.0756,  # プロペラ直径[m]
        m_=0.1822,  # 質量(無次元化)[-]
        I_zG=0.01138,  # 慣性モーメント[-]
        Λ=2.1683,  # アスペクト比[-]
        A_R_Ld=0.01867,  # 船の断面に対する舵面積比[-]
        η=0.7916,  # プロペラ直径に対する舵高さ(Dp/H)
        m_x_=0.00601,  # 付加質量x(無次元)
        m_y_=0.1521,  # 付加質量y(無次元)
        J_z=0.00729,  # 付加質量Izz(無次元)
        f_α=(6.13 * 2.1683) / (2.25 + 2.1683),  # 直圧力勾配係数
        ϵ=0.90,  # プロペラ・舵位置伴流係数比
        t_R=0.441,  # 操縦抵抗減少率
        a_H=0.232,  # 舵力増加係数
        x_H=-0.711,  # 舵力増分作用位置
        γ_R=0.4115,  # 整流係数
        l_R=-0.774,  # 船長に対する舵位置
        κ=0.713,  # 修正係数
        t_P=0.20,  # 推力減少率
        w_P0=0.326,  # 有効伴流率
    )
    return basic_params


@pytest.fixture
def maneuvering_params():
    return Mmg3DofManeuveringParams(
        C_1=0.48301,
        C_2=-0.29765,
        C_3=-0.16423,
        X_0=-0.07234,
        X_ββ=-0.23840,
        X_βγ=-0.03231 + 0.1521,
        X_γγ=-0.06405,
        X_ββββ=-0.30047,
        Y_β=0.85475,
        Y_γ=0.11461 + 0.00601,
        Y_βββ=6.73201,
        Y_ββγ=-2.23689,
        Y_βγγ=3.38577,
        Y_γγγ=-0.04151,
        N_β=0.096567,
        N_γ=-0.036501,
        N_βββ=0.14090,
        N_ββγ=-0.46158,
        N_βγγ=0.01648,
        N_γγγ=-0.030404,
    )


@pytest.fixture
def ship_mmg_3dof(basic_params, maneuvering_params):
    duration = 50  # [s]
    sampling = 2000
    time_list = np.linspace(0.00, duration, sampling)
    delta_rad_list = [0] * sampling
    for i in range(sampling):
        if i >= 400:
            delta_rad_list[i] = 30.0 * np.pi / 180.0
        else:
            delta_rad_list[i] = 0
    n_const = 20.338
    npm_list = np.array([n_const for i in range(sampling)])

    # R0 (R0(u) = 0.0)
    u_list = np.linspace(0.00, 10.00, 10)
    R0_list = np.full(10, 0.0)
    R0_func = interp1d(u_list, R0_list)

    result = simulate_mmg_3dof(
        basic_params,
        maneuvering_params,
        R0_func,
        time_list,
        delta_rad_list,
        npm_list,
        u0=1.21,
    )
    ship = ShipObj3dof(L=basic_params.L_pp, B=basic_params.B)
    ship.load_simulation_result(time_list, result.T[0], result.T[1], result.T[2])
    return ship


def test_Ship3DOF_drawing_function(ship_mmg_3dof):
    """Check drawing functions of Ship3DOF class by using MMG 3DOF simulation results"""

    # Ship3DOF.draw_xy_trajectory()
    save_fig_path = "test.png"
    ship_mmg_3dof.draw_xy_trajectory(dimensionless=True)
    ship_mmg_3dof.draw_xy_trajectory(save_fig_path=save_fig_path)
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)

    # Ship3DOF.draw_chart()
    save_fig_path = "test.png"
    ship_mmg_3dof.draw_chart(
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
            ship_mmg_3dof.draw_chart(x_index, y_index)

    with pytest.raises(Exception):
        ship_mmg_3dof.draw_chart("time", "hogehoge")
    with pytest.raises(Exception):
        ship_mmg_3dof.draw_chart("hogehoge", "y")

    # Ship3DOF.draw_gif()
    ship_mmg_3dof.draw_gif(save_fig_path=save_fig_path)
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)
    ship_mmg_3dof.draw_gif(dimensionless=True, save_fig_path=save_fig_path)
    if os.path.exists(save_fig_path):
        os.remove(save_fig_path)
