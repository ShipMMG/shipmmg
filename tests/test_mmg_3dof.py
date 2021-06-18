#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shipmmg.mmg_3dof import (
    Mmg3DofBasicParams,
    Mmg3DofManeuveringParams,
    simulate_mmg_3dof,
    get_sub_values_from_simulation_result,
    zigzag_test_mmg_3dof,
)
from shipmmg.ship_obj_3dof import ShipObj3dof
import numpy as np
import pytest
import os
import matplotlib.pyplot as plt


@pytest.fixture
def ship_KVLCC2_L7_model():
    ρ = 1.025  # 海水密度

    L_pp = 7.00  # 船長Lpp[m]
    B = 1.27  # 船幅[m]
    d = 0.46  # 喫水[m]
    nabla = 3.27  # 排水量[m^3]
    x_G = 0.25  # 重心位置[m]
    # C_b = 0.810  # 方形係数[-]
    D_p = 0.216  # プロペラ直径[m]
    H_R = 0.345  # 舵高さ[m]
    A_R = 0.0539  # 舵断面積[m^2]

    t_P = 0.220  # 推力減少率
    w_P0 = 0.40  # 有効伴流率
    m_x_dash = 0.022  # 付加質量x(無次元)
    m_y_dash = 0.223  # 付加質量y(無次元)
    J_z_dash = 0.011  # 付加質量Izz(無次元)
    t_R = 0.387  # 操縦抵抗減少率
    a_H = 0.312  # 舵力増加係数
    x_H_dash = -0.464  # 舵力増分作用位置
    γ_R_minus = 0.395  # 整流係数
    γ_R_plus = 0.640  # 整流係数
    l_r_dash = -0.710  # 船長に対する舵位置
    x_P_dash = -0.690  # 船長に対するプロペラ位置
    ϵ = 1.09  # プロペラ・舵位置伴流係数比
    κ = 0.50  # 修正係数
    f_α = 2.747  # 直圧力勾配係数

    basic_params = Mmg3DofBasicParams(
        L_pp=L_pp,  # 船長Lpp[m]
        B=B,  # 船幅[m]
        d=d,  # 喫水[m]
        x_G=x_G,  # 重心位置[]
        D_p=D_p,  # プロペラ直径[m]
        m=ρ * nabla,  # 質量(無次元化)[kg]
        I_zG=ρ * nabla * ((0.25 * L_pp) ** 2),  # 慣性モーメント[-]
        A_R=A_R,  # 船の断面に対する舵面積比[-]
        η=D_p / H_R,  # プロペラ直径に対する舵高さ(Dp/H)
        m_x=(0.5 * ρ * (L_pp ** 2) * d) * m_x_dash,  # 付加質量x(無次元)
        m_y=(0.5 * ρ * (L_pp ** 2) * d) * m_y_dash,  # 付加質量y(無次元)
        J_z=(0.5 * ρ * (L_pp ** 4) * d) * J_z_dash,  # 付加質量Izz(無次元)
        f_α=f_α,
        ϵ=ϵ,  # プロペラ・舵位置伴流係数比
        t_R=t_R,  # 操縦抵抗減少率
        a_H=a_H,  # 舵力増加係数
        x_H=x_H_dash * L_pp,  # 舵力増分作用位置
        γ_R_minus=γ_R_minus,  # 整流係数
        γ_R_plus=γ_R_plus,  # 整流係数
        l_R=l_r_dash,  # 船長に対する舵位置
        κ=κ,  # 修正係数
        t_P=t_P,  # 推力減少率
        w_P0=w_P0,  # 有効伴流率
        x_P=x_P_dash,  # 船長に対するプロペラ位置
    )

    k_0 = 0.2931
    k_1 = -0.2753
    k_2 = -0.1385
    R_0_dash = 0.022
    X_vv_dash = -0.040
    X_vr_dash = 0.002
    X_rr_dash = 0.011
    X_vvvv_dash = 0.771
    Y_v_dash = -0.315
    Y_r_dash = 0.083
    Y_vvv_dash = -1.607
    Y_vvr_dash = 0.379
    Y_vrr_dash = -0.391
    Y_rrr_dash = 0.008
    N_v_dash = -0.137
    N_r_dash = -0.049
    N_vvv_dash = -0.030
    N_vvr_dash = -0.294
    N_vrr_dash = 0.055
    N_rrr_dash = -0.013
    maneuvering_params = Mmg3DofManeuveringParams(
        k_0=k_0,
        k_1=k_1,
        k_2=k_2,
        R_0_dash=R_0_dash,
        X_vv_dash=X_vv_dash,
        X_vr_dash=X_vr_dash,
        X_rr_dash=X_rr_dash,
        X_vvvv_dash=X_vvvv_dash,
        Y_v_dash=Y_v_dash,
        Y_r_dash=Y_r_dash,
        Y_vvv_dash=Y_vvv_dash,
        Y_vvr_dash=Y_vvr_dash,
        Y_vrr_dash=Y_vrr_dash,
        Y_rrr_dash=Y_rrr_dash,
        N_v_dash=N_v_dash,
        N_r_dash=N_r_dash,
        N_vvv_dash=N_vvv_dash,
        N_vvr_dash=N_vvr_dash,
        N_vrr_dash=N_vrr_dash,
        N_rrr_dash=N_rrr_dash,
    )
    return basic_params, maneuvering_params


@pytest.fixture
def kvlcc2_L7_35_turning(ship_KVLCC2_L7_model):
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
    sim_result = sol.sol(time_list)
    ship = ShipObj3dof(L=basic_params.L_pp, B=basic_params.B)
    ship.load_simulation_result(time_list, sim_result[0], sim_result[1], sim_result[2])
    ship.npm = npm_list
    ship.δ = δ_rad_list
    return ship


def test_get_sub_values_from_simulation_result(
    kvlcc2_L7_35_turning, ship_KVLCC2_L7_model, tmpdir
):
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

    save_fig_path = os.path.join(str(tmpdir),"testFN.png")

    fig = plt.figure()
    plt.plot(kvlcc2_L7_35_turning.time, F_N_list)
    fig.savefig(save_fig_path)


def test_Ship3DOF_drawing_function(kvlcc2_L7_35_turning,tmpdir):
    """Check drawing functions of Ship3DOF class by using MMG 3DOF simulation results."""
    # Ship3DOF.draw_xy_trajectory()
    save_fig_path = os.path.join(str(tmpdir),"trajectory.png") 
    

    kvlcc2_L7_35_turning.draw_xy_trajectory(dimensionless=True)
    kvlcc2_L7_35_turning.draw_xy_trajectory(save_fig_path=save_fig_path)


    # Ship3DOF.draw_chart()
    save_fig_path = os.path.join(str(tmpdir),"param.png")  

    kvlcc2_L7_35_turning.draw_chart(
        "time",
        "u",
        xlabel="time [sec]",
        ylabel=r"$u$" + " [m/s]",
        save_fig_path=save_fig_path,
    )

    x_index_list = ["time", "u", "v", "r", "x", "y", "psi"]
    y_index_list = ["time", "u", "v", "r", "x", "y", "psi"]
    for x_index in x_index_list:
        for y_index in y_index_list:
            kvlcc2_L7_35_turning.draw_chart(x_index, y_index)

    with pytest.raises(Exception):
        kvlcc2_L7_35_turning.draw_chart("time", "hogehoge")
    with pytest.raises(Exception):
        kvlcc2_L7_35_turning.draw_chart("hogehoge", "y")

    # Ship3DOF.draw_gif()
    save_fig_path = os.path.join(str(tmpdir),"test.gif")
    
    kvlcc2_L7_35_turning.draw_gif(save_fig_path=save_fig_path)
    
    kvlcc2_L7_35_turning.draw_gif(dimensionless=True, save_fig_path=save_fig_path)
    

def test_zigzag_test_mmg_before(ship_KVLCC2_L7_model,tmpdir):
    basic_params, maneuvering_params = ship_KVLCC2_L7_model
    target_δ_rad = 20.0 * np.pi / 180.0
    target_ψ_rad_deviation = -20.0 * np.pi / 180.0
    duration = 100
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
        δ_rad_rate=10.0 * np.pi / 180,
    )

    ship = ShipObj3dof(L=100, B=10)
    ship.load_simulation_result(time_list, u_list, v_list, r_list)
    ship.δ = δ_list

    save_fig_path = os.path.join(str(tmpdir),"test_psi.png")

    ship.draw_xy_trajectory(save_fig_path=save_fig_path)
    ship.draw_chart(
        "time",
        "psi",
        xlabel="time [sec]",
        ylabel=r"$\psi$" + " [rad]",
        save_fig_path=save_fig_path,
    )

    save_fig_path = os.path.join(str(tmpdir),"test_delta.png")
    ship.draw_xy_trajectory(save_fig_path=save_fig_path)
    ship.draw_chart(
        "time",
        "delta",
        xlabel="time [sec]",
        ylabel=r"$\delta$" + " [rad]",
        save_fig_path=save_fig_path,
    )

    save_fig_path = os.path.join(str(tmpdir),"test_delta_psi.png")

    ship.draw_multi_y_chart(
        "time",
        ["delta", "psi"],
        xlabel="time [sec]",
        ylabel="[rad]",
        save_fig_path=save_fig_path,
    )

    save_fig_path = os.path.join(str(tmpdir),"test_delta_psi.png")

    ship.draw_multi_x_chart(
        ["delta", "psi"],
        "time",
        ylabel="time [sec]",
        xlabel="[rad]",
        save_fig_path=save_fig_path,
    )


def test_zigzag_test_mmg(ship_KVLCC2_L7_model, tmpdir):

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

    ship = ShipObj3dof(L=100, B=10)
    ship.load_simulation_result(time_list, u_list, v_list, r_list)
    ship.δ = δ_list
    ship.npm = npm_list

    save_fig_path = os.path.join(str(tmpdir),"delta_psi.png")

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
    ) = get_sub_values_from_simulation_result(
        ship.u,
        ship.v,
        ship.r,
        ship.δ,
        ship.npm,
        basic_params,
        maneuvering_params,
        return_all_vals=True,
    )

    save_fig_path = os.path.join(str(tmpdir),"w_P.png")
    fig = plt.figure()
    plt.plot(time_list, w_P_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir),"J.png")
    fig = plt.figure()
    plt.plot(time_list, J_list)
    fig.savefig(save_fig_path)
    plt.close()


    save_fig_path = os.path.join(str(tmpdir),"K_T.png")

    fig = plt.figure()
    plt.plot(time_list, K_T_list)
    fig.savefig(save_fig_path)
    plt.close()


    save_fig_path = os.path.join(str(tmpdir),"U_R.png")
    fig = plt.figure()
    plt.plot(time_list, U_R_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir),"α_R.png")
    fig = plt.figure()
    plt.plot(time_list, α_R_list)
    fig.savefig(save_fig_path)
    plt.close()


    save_fig_path = os.path.join(str(tmpdir),"F_N.png")
    fig = plt.figure()
    plt.plot(time_list, F_N_list)
    fig.savefig(save_fig_path)
    plt.close()

    save_fig_path = os.path.join(str(tmpdir),"gamma_R.png")
    fig = plt.figure()
    plt.plot(time_list, γ_R_list)
    fig.savefig(save_fig_path)
    plt.close()

