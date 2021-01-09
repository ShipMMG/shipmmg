#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .draw_obj import DrawObj


@dataclasses.dataclass
class ShipObj3dof:
    """Ship 3DOF class just for drawing.

    Attributes:
        L (float):
            ship length [m]
        B (float)
            ship breath [m]
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

    # Ship overview
    L: float
    B: float
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
            >>> ship = ShipObj3dof(L = 180, B = 20)
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
        self,
        dimensionless: bool = False,
        aspect_equal: bool = True,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        save_fig_path: str = None,
    ) -> plt.Figure:
        """
        Draw trajectry(x,y).

        Args:
            dimensionless (bool, optional):
                drawing with dimensionless by using L or not.
                Defaults to False
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
            >>> ship.draw_xy_trajectory(save_fig_path="test.png")
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)

        if dimensionless:
            plt.plot(np.array(self.x) / self.L, np.array(self.y) / self.L)
            plt.xlabel(r"$x/L$")
            plt.ylabel(r"$y/L$")
        else:
            plt.plot(self.x, self.y)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
        if aspect_equal:
            plt.gca().set_aspect("equal")
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        plt.close()
        return fig

    def draw_chart(
        self,
        x_index: str,
        y_index: str,
        xlabel: str = None,
        ylabel: str = None,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        save_fig_path: str = None,
    ) -> plt.Figure:
        """
        Draw chart.

        Args:
            x_index (string):
                Index value of X axis.
            y_index (string):
                Index value of Y axis.
            xlabel (string, optional):
                Label of X axis.
                Defaults to None.
            ylabel (string, optional):
                Label of Y axis.
                Defaults to None.
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
            >>> ship_kt.draw_chart("time", "r", xlabel="time [sec]", \
            >>> ylabel=r"$u$" + " [rad/s]",save_fig_path='test.png')
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

    def draw_gif(
        self,
        dimensionless: bool = False,
        aspect_equal: bool = True,
        frate: int = 10,
        interval: int = 100,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        save_fig_path: str = None,
    ) -> plt.Figure:
        """Draw GIF of ship trajectory
        Args:
            dimensionless (bool, optional):
                drawing with dimensionless by using L or not.
                Defaults to False
            aspect_equal (bool, optional):
                Set equal of figure aspect or not.
                Defaults to True.
            frate (int, optional):
                One of the parameter of `frames` in matplotlib.FuncAnimation().
                `frames` expresses source of data to pass func and each frame of the animation.
                `frames = int (len(time) / frate)`
                Defaults to 10.
            interval (int, optional):
                Delay between frames in milliseconds.
                Defaults to 100.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
        """

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        if dimensionless:
            draw_x = np.array(self.x) / self.L
            draw_y = np.array(self.y) / self.L
            ax.set_xlabel(r"$x/L$")
            ax.set_ylabel(r"$y/L$")
            shape = (1 / 2, self.B / (2 * self.L))
        else:
            draw_x = np.array(self.x)
            draw_y = np.array(self.y)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            shape = (self.L / 2, self.B / 2)

        plt.plot(
            draw_x,
            draw_y,
            label="trajectory",
            ls="--",
            color="k",
        )
        if aspect_equal:
            ax.set_aspect("equal")

        drawer = DrawObj(ax)

        def update_obj(i, x_list, y_list, shape_list, ψ_list, frate):
            j = int(frate * i)
            plt.title(r"$t$ = " + "{:.1f}".format(self.time[j]))

            xT = np.array(x_list).T
            _x_list_j = list(xT[j].T)
            yT = np.array(y_list).T
            _y_list_j = list(yT[j].T)
            ψT = np.array(ψ_list).T
            _ψ_list_j = list(ψT[j].T)

            return drawer.draw_square_with_angle(
                _x_list_j, _y_list_j, shape_list, _ψ_list_j
            )

        ani = FuncAnimation(
            fig,
            update_obj,
            fargs=(
                [draw_x],
                [draw_y],
                [shape],
                [self.psi],
                frate,
            ),
            interval=interval,
            frames=int(len(self.time) / frate),
        )
        gif = ani.save(save_fig_path, writer="pillow")
        plt.close()
        return gif
