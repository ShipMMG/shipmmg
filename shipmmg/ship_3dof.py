#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import dataclasses
import numpy as np
import matplotlib.pyplot as plt

# import traceback


@dataclasses.dataclass
class Ship3DOF:
    """Ship 3DOF class just for visualizing.

    Attributes:
        time (list[float]):
            Time list of simulation result.
        u (list[float]):
            List of axial velocity [m/s] in simulation result.
        v (list[float]):
            List of lateral velocity [m/s] in simulation result.
        r (list[float]):
            List of rate of turn [rad/s] in simulation result.
        x (list[float]):
            List of position of X axis [m] in simulation result.
        y (list[float]):
            List of position of Y axis [m/s] in simulation result.
        psi (list[float]):
            List of azimuth [rad] in simulation result.
    """

    # Simulation result
    time: List[float] = dataclasses.field(default_factory=list)
    u: List[float] = dataclasses.field(default_factory=list)
    v: List[float] = dataclasses.field(default_factory=list)
    r: List[float] = dataclasses.field(default_factory=list)
    x: List[float] = dataclasses.field(default_factory=list)
    y: List[float] = dataclasses.field(default_factory=list)
    psi: List[float] = dataclasses.field(default_factory=list)

    def load_simulation_result(
        self,
        time: List[float],
        u: List[float],
        v: List[float],
        r: List[float],
        x0: float = 0.0,
        y0: float = 0.0,
        psi0: float = 0.0,
    ):
        """Load simulation result (time, u, v, r).
        By running this, `x`, `y` and `psi` of this class are registered automatically.

        Args:
            time (list[float]):
                Time list of simulation result.
            u (list[float]):
                List of axial velocity [m/s] in simulation result.
            v (list[float]):
                List of lateral velocity [m/s] in simulation result.
            r (list[float]):
                List of rate of turn [rad/s] in simulation result.
            x0 (float, optional):
                Inital position of X axis [m].
                Defaults to 0.0.
            y (list[float]):
                Inital position of Y axis [m/s].
                Defaults to 0.0.
            psi (list[float]):
                Inital azimuth [rad].
                Defaults to 0.0.

        Examples:
            >>> time_list = np.linspace(0.00, duration, num_of_sampling)
            >>> delta_list = np.full(len(time_list), 10 * np.pi / 180)
            >>> result = simulate(0.15, 60.0, time_list, delta_list)
            >>> u_list = np.full(len(time_list), 20 * (1852.0 / 3600))
            >>> v_list = np.zeros(len(time_list))
            >>> r_list = result.T[0]
            >>> ship = Ship3DOF()
            >>> ship.load_simulation_result(time_list, u_list, v_list, r_list)
            >>> print(ship.x, ship.y, ship.psi)
        """
        x = [x0]
        y = [y0]
        psi = [psi0]
        for i, (ut, vt, rt) in enumerate(zip(u, v, r)):
            if i > 0:
                dt = time[i] - time[i - 1]
                x.append(x[-1] + (ut * np.cos(psi[-1]) - vt * np.sin(psi[-1])) * dt)
                y.append(y[-1] + (ut * np.sin(psi[-1]) + vt * np.cos(psi[-1])) * dt)
                psi.append(psi[-1] + rt * dt)

        # Register
        self.time = time
        self.u = u
        self.v = v
        self.r = r
        self.x = x
        self.y = y
        self.psi = psi

    def draw_xy_trajectory(
        self, aspect_equal=True, figsize=[6.4, 4.8], dpi=100.0, save_fig_path=None
    ) -> plt.Figure:
        """
        Draw trajectry(x,y).

        Args:
            aspect_equal (bool, optional):
                Set equal of figure aspect or not.
                Defaults to True.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
        Returns:
            matplotlib.pyplot.Figure: Figure

        Examples:
            >>> save_fig_path = "test.png"
            >>> ship.draw_xy_trajectory(save_fig_path=save_fig_path)
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.plot(self.x, self.y)
        if aspect_equal:
            plt.gca().set_aspect("equal")
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        plt.close()
        return fig

    def draw_chart(
        self,
        x_index,
        y_index,
        xlabel=None,
        ylabel=None,
        figsize=[6.4, 4.8],
        dpi=100.0,
        save_fig_path=None,
    ) -> plt.Figure:
        """
        Draw chart.

        Args:
            x_index (string):
            y_index (string):
            xlabel (string, optional):
            ylabel (string, optional):
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
        Returns:
            matplotlib.pyplot.Figure: Figure
        """
        target_x = None
        if x_index == "time":
            target_x = self.time
        elif x_index == "u":
            target_x = self.u
        elif x_index == "v":
            target_x = self.v
        elif x_index == "r":
            target_x = self.r
        elif x_index == "x":
            target_x = self.x
        elif x_index == "y":
            target_x = self.y
        elif x_index == "psi":
            target_x = self.psi
        if target_x is None:
            raise Exception(
                "`x_index` is not good. Please set `x_index` from ["
                "time"
                ", "
                " u"
                ", "
                " v"
                ", "
                " r"
                ", "
                " x"
                ", "
                " y"
                ", "
                " psi"
                "]"
            )

        target_y = None
        if y_index == "time":
            target_y = self.time
        elif y_index == "u":
            target_y = self.u
        elif y_index == "v":
            target_y = self.v
        elif y_index == "r":
            target_y = self.r
        elif y_index == "x":
            target_y = self.x
        elif y_index == "y":
            target_y = self.y
        elif y_index == "psi":
            target_y = self.psi
        if target_y is None:
            raise Exception(
                "`y_index` is not good. Please set `y_index` from ["
                "time"
                ", "
                " u"
                ", "
                " v"
                ", "
                " r"
                ", "
                " x"
                ", "
                " y"
                ", "
                " psi"
                "]"
            )
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.plot(target_x, target_y)
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        plt.close()

        return fig
