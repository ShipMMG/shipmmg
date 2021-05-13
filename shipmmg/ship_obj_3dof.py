#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
        δ (list[float]):
            rudder angle list of simulation.
        npm (List[float]):
            npm list of simulation.
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
    δ: List[float] = dataclasses.field(default_factory=list)
    npm: List[float] = dataclasses.field(default_factory=list)

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
            >>> kt_params = KTParams(K=0.15, T=60.0)
            >>> result = kt.simulate_kt(kt_params, time_list, delta_list)
            >>> u_list = np.full(len(time_list), 20 * (1852.0 / 3600))
            >>> v_list = np.zeros(len(time_list))
            >>> r_list = result[0]
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
        num: int or str = None,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        fmt: str = None,
        facecolor: str = None,
        edgecolor: str = None,
        frameon: bool = True,
        FigureClass: matplotlib.figure.Figure = matplotlib.figure.Figure,
        clear: bool = False,
        tight_layout: bool = False,
        constrained_layout: bool = False,
        save_fig_path: str = None,
        **kwargs
    ) -> plt.Figure:
        """Draw trajectry(x,y).

        Args:
            dimensionless (bool, optional):
                drawing with dimensionless by using L or not.
                Defaults to False
            aspect_equal (bool, optional):
                Set equal of figure aspect or not.
                Defaults to True.
            num (int or str, optional):
                A unique identifier for the figure.
                If a figure with that identifier already exists, this figure is made active and returned.
                An integer refers to the Figure.number attribute, a string refers to the figure label.
                If there is no figure with the identifier or num is not given,
                a new figure is created, made active and returned.
                If num is an int, it will be used for the Figure.number attribute.
                Otherwise, an auto-generated integer value is used (starting at 1 and incremented for each new figure).
                If num is a string, the figure label and the window title is set to this value.
                Default to None.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0
            facecolor (str, optional):
                The background color.
            edgecolor (str, optional):
                The border color.
            frameon (bool, optional):
                If False, suppress drawing the figure frame.
                Defaults to True.
            FigureClass (subclass of matplotlib.figure.Figure, optional):
                Optionally use a custom Figure instance.
                Defaults to matplotlib.figure.Figure.
            clear (bool, optional):
                If True and the figure already exists, then it is cleared.
                Defaults to False.
            tight_layout (bool, optional):
                If False use subplotpars.
                If True adjust subplot parameters using tight_layout with default padding.
                When providing a dict containing the keys pad, w_pad, h_pad, and rect,
                the default tight_layout paddings will be overridden.
                Defaults to False.
            constrained_layout (bool, optional):
                If True use constrained layout to adjust positioning of plot elements.
                Like tight_layout, but designed to be more flexible.
                See Constrained Layout Guide for examples.
                (Note: does not work with add_subplot or subplot2grid.)
                Defaults to False.
            fmt (str, optional):
                A format string, e.g. 'ro' for red circles.
                See the Notes section for a full description of the format strings.
                Format strings are just an abbreviation for quickly setting basic line properties.
                All of these and more can also be controlled by keyword arguments.
                This argument cannot be passed as keyword.
                Defaults to None.
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
            **kwargs (matplotlib.lines.Line2D properties, optional):
                kwargs are used to specify properties
                like a line label (for auto legends), linewidth, antialiasing, marker face color.
                You can show the detailed information at `matplotlib.lines.Line2D
                 <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_

        Returns:
            matplotlib.pyplot.Figure: Figure

        Examples:
            >>> ship.draw_xy_trajectory(save_fig_path="test.png")
        """
        fig = plt.figure(
            num=num,
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            frameon=frameon,
            FigureClass=FigureClass,
            clear=clear,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout,
        )

        if dimensionless:
            if fmt is None:
                plt.plot(np.array(self.x) / self.L, np.array(self.y) / self.L, **kwargs)
            else:
                plt.plot(
                    np.array(self.x) / self.L, np.array(self.y) / self.L, fmt, **kwargs
                )
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
        num: int or str = None,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        facecolor: str = None,
        edgecolor: str = None,
        frameon: bool = True,
        FigureClass: matplotlib.figure.Figure = matplotlib.figure.Figure,
        clear: bool = False,
        tight_layout: bool = False,
        constrained_layout: bool = False,
        fmt: str = None,
        save_fig_path: str = None,
        **kwargs
    ) -> plt.Figure:
        """Draw chart.

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
            num (int or str, optional):
                A unique identifier for the figure.
                If a figure with that identifier already exists, this figure is made active and returned.
                An integer refers to the Figure.number attribute, a string refers to the figure label.
                If there is no figure with the identifier or num is not given,
                a new figure is created, made active and returned.
                If num is an int, it will be used for the Figure.number attribute.
                Otherwise, an auto-generated integer value is used (starting at 1 and incremented for each new figure).
                If num is a string, the figure label and the window title is set to this value.
                Default to None.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0.
            facecolor (str, optional):
                The background color.
            edgecolor (str, optional):
                The border color.
            frameon (bool, optional):
                If False, suppress drawing the figure frame.
                Defaults to True.
            FigureClass (subclass of matplotlib.figure.Figure, optional):
                Optionally use a custom Figure instance.
                Defaults to matplotlib.figure.Figure.
            clear (bool, optional):
                If True and the figure already exists, then it is cleared.
                Defaults to False.
            tight_layout (bool, optional):
                If False use subplotpars.
                If True adjust subplot parameters using tight_layout with default padding.
                When providing a dict containing the keys pad, w_pad, h_pad, and rect,
                the default tight_layout paddings will be overridden.
                Defaults to False.
            constrained_layout (bool, optional):
                If True use constrained layout to adjust positioning of plot elements.
                Like tight_layout, but designed to be more flexible.
                See Constrained Layout Guide for examples.
                (Note: does not work with add_subplot or subplot2grid.)
                Defaults to False.
            fmt (str, optional):
                A format string, e.g. 'ro' for red circles.
                See the Notes section for a full description of the format strings.
                Format strings are just an abbreviation for quickly setting basic line properties.
                All of these and more can also be controlled by keyword arguments.
                This argument cannot be passed as keyword.
                Defaults to None.
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
            **kwargs (matplotlib.lines.Line2D properties, optional):
                kwargs are used to specify properties
                like a line label (for auto legends), linewidth, antialiasing, marker face color.
                You can show the detailed information at `matplotlib.lines.Line2D
                 <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        Returns:
            matplotlib.pyplot.Figure: Figure

        Examples:
            >>> ship.draw_chart("time", "r", xlabel="time [sec]", \
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
        elif x_index == "delta":
            target_x = self.δ
        elif x_index == "δ":
            target_x = self.δ
        elif x_index == "npm":
            target_x = self.npm
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
                ", "
                " delta"
                ", "
                " δ"
                ", "
                " npm"
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
        elif y_index == "delta":
            target_y = self.δ
        elif y_index == "δ":
            target_y = self.δ
        elif y_index == "npm":
            target_y = self.npm
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
                ", "
                " delta"
                ", "
                " δ"
                ", "
                " npm"
                "]"
                "]"
            )
        fig = plt.figure(
            num=num,
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            frameon=frameon,
            FigureClass=FigureClass,
            clear=clear,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout,
        )
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if fmt is None:
            plt.plot(target_x, target_y, **kwargs)
        else:
            plt.plot(target_x, target_y, fmt, **kwargs)
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        plt.close()

        return fig

    def draw_multi_x_chart(
        self,
        x_index_list: List[str],
        y_index: str,
        xlabel: str = None,
        ylabel: str = None,
        num: int or str = None,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        facecolor: str = None,
        edgecolor: str = None,
        frameon: bool = True,
        FigureClass: matplotlib.figure.Figure = matplotlib.figure.Figure,
        clear: bool = False,
        tight_layout: bool = False,
        constrained_layout: bool = False,
        fmt: str = None,
        save_fig_path: str = None,
        **kwargs
    ) -> plt.Figure:
        """Draw chart of multiple Y variables.

        Args:
            x_index_list (List[string]):
                List of index value of X axis.
            y_index (string):
                Index value of Y axis.
            xlabel (string, optional):
                Label of X axis.
                Defaults to None.
            ylabel (string, optional):
                Label of Y axis.
                Defaults to None.
            num (int or str, optional):
                A unique identifier for the figure.
                If a figure with that identifier already exists, this figure is made active and returned.
                An integer refers to the Figure.number attribute, a string refers to the figure label.
                If there is no figure with the identifier or num is not given,
                a new figure is created, made active and returned.
                If num is an int, it will be used for the Figure.number attribute.
                Otherwise, an auto-generated integer value is used (starting at 1 and incremented for each new figure).
                If num is a string, the figure label and the window title is set to this value.
                Default to None.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0.
            facecolor (str, optional):
                The background color.
            edgecolor (str, optional):
                The border color.
            frameon (bool, optional):
                If False, suppress drawing the figure frame.
                Defaults to True.
            FigureClass (subclass of matplotlib.figure.Figure, optional):
                Optionally use a custom Figure instance.
                Defaults to matplotlib.figure.Figure.
            clear (bool, optional):
                If True and the figure already exists, then it is cleared.
                Defaults to False.
            tight_layout (bool, optional):
                If False use subplotpars.
                If True adjust subplot parameters using tight_layout with default padding.
                When providing a dict containing the keys pad, w_pad, h_pad, and rect,
                the default tight_layout paddings will be overridden.
                Defaults to False.
            constrained_layout (bool, optional):
                If True use constrained layout to adjust positioning of plot elements.
                Like tight_layout, but designed to be more flexible.
                See Constrained Layout Guide for examples.
                (Note: does not work with add_subplot or subplot2grid.)
                Defaults to False.
            fmt (str, optional):
                A format string, e.g. 'ro' for red circles.
                See the Notes section for a full description of the format strings.
                Format strings are just an abbreviation for quickly setting basic line properties.
                All of these and more can also be controlled by keyword arguments.
                This argument cannot be passed as keyword.
                Defaults to None.
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
            **kwargs (matplotlib.lines.Line2D properties, optional):
                kwargs are used to specify properties
                like a line label (for auto legends), linewidth, antialiasing, marker face color.
                You can show the detailed information at `matplotlib.lines.Line2D
                 <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        Returns:
            matplotlib.pyplot.Figure: Figure

        Examples:
            >>> ship.draw_chart("time", "r", xlabel="time [sec]", \
            >>> ylabel=r"$u$" + " [rad/s]",save_fig_path='test.png')
        """
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
        elif y_index == "delta":
            target_y = self.δ
        elif y_index == "δ":
            target_y = self.δ
        elif y_index == "npm":
            target_y = self.npm
        if target_y is None:
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
                ", "
                " delta"
                ", "
                " δ"
                ", "
                " npm"
                "]"
            )

        target_x_list = []
        for x_index in x_index_list:
            if x_index == "time":
                target_x_list.append(self.time)
            elif x_index == "u":
                target_x_list.append(self.u)
            elif x_index == "v":
                target_x_list.append(self.v)
            elif x_index == "r":
                target_x_list.append(self.r)
            elif x_index == "x":
                target_x_list.append(self.x)
            elif x_index == "y":
                target_x_list.append(self.y)
            elif x_index == "psi":
                target_x_list.append(self.psi)
            elif x_index == "delta":
                target_x_list.append(self.δ)
            elif x_index == "δ":
                target_x_list.append(self.δ)
            elif x_index == "npm":
                target_x_list.append(self.npm)
        if len(target_x_list) == 0:
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
                ", "
                " delta"
                ", "
                " δ"
                ", "
                " npm"
                "]"
                "]"
            )
        fig = plt.figure(
            num=num,
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            frameon=frameon,
            FigureClass=FigureClass,
            clear=clear,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout,
        )
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if fmt is None:
            for target_x in target_x_list:
                plt.plot(target_x, target_y, **kwargs)
        else:
            for target_x in target_x_list:
                plt.plot(target_x, target_y, fmt, **kwargs)
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        plt.close()

        return fig

    def draw_multi_y_chart(
        self,
        x_index: str,
        y_index_list: List[str],
        xlabel: str = None,
        ylabel: str = None,
        num: int or str = None,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        facecolor: str = None,
        edgecolor: str = None,
        frameon: bool = True,
        FigureClass: matplotlib.figure.Figure = matplotlib.figure.Figure,
        clear: bool = False,
        tight_layout: bool = False,
        constrained_layout: bool = False,
        fmt: str = None,
        save_fig_path: str = None,
        **kwargs
    ) -> plt.Figure:
        """Draw chart of multiple Y variables.

        Args:
            x_index (string):
                Index value of X axis.
            y_index_list (List[string]):
                List of index value of Y axis.
            xlabel (string, optional):
                Label of X axis.
                Defaults to None.
            ylabel (string, optional):
                Label of Y axis.
                Defaults to None.
            num (int or str, optional):
                A unique identifier for the figure.
                If a figure with that identifier already exists, this figure is made active and returned.
                An integer refers to the Figure.number attribute, a string refers to the figure label.
                If there is no figure with the identifier or num is not given,
                a new figure is created, made active and returned.
                If num is an int, it will be used for the Figure.number attribute.
                Otherwise, an auto-generated integer value is used (starting at 1 and incremented for each new figure).
                If num is a string, the figure label and the window title is set to this value.
                Default to None.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0.
            facecolor (str, optional):
                The background color.
            edgecolor (str, optional):
                The border color.
            frameon (bool, optional):
                If False, suppress drawing the figure frame.
                Defaults to True.
            FigureClass (subclass of matplotlib.figure.Figure, optional):
                Optionally use a custom Figure instance.
                Defaults to matplotlib.figure.Figure.
            clear (bool, optional):
                If True and the figure already exists, then it is cleared.
                Defaults to False.
            tight_layout (bool, optional):
                If False use subplotpars.
                If True adjust subplot parameters using tight_layout with default padding.
                When providing a dict containing the keys pad, w_pad, h_pad, and rect,
                the default tight_layout paddings will be overridden.
                Defaults to False.
            constrained_layout (bool, optional):
                If True use constrained layout to adjust positioning of plot elements.
                Like tight_layout, but designed to be more flexible.
                See Constrained Layout Guide for examples.
                (Note: does not work with add_subplot or subplot2grid.)
                Defaults to False.
            fmt (str, optional):
                A format string, e.g. 'ro' for red circles.
                See the Notes section for a full description of the format strings.
                Format strings are just an abbreviation for quickly setting basic line properties.
                All of these and more can also be controlled by keyword arguments.
                This argument cannot be passed as keyword.
                Defaults to None.
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
            **kwargs (matplotlib.lines.Line2D properties, optional):
                kwargs are used to specify properties
                like a line label (for auto legends), linewidth, antialiasing, marker face color.
                You can show the detailed information at `matplotlib.lines.Line2D
                 <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        Returns:
            matplotlib.pyplot.Figure: Figure

        Examples:
            >>> ship.draw_chart("time", "r", xlabel="time [sec]", \
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
        elif x_index == "delta":
            target_x = self.δ
        elif x_index == "δ":
            target_x = self.δ
        elif x_index == "npm":
            target_x = self.npm
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
                ", "
                " delta"
                ", "
                " δ"
                ", "
                " npm"
                "]"
            )

        target_y_list = []
        for y_index in y_index_list:
            if y_index == "time":
                target_y_list.append(self.time)
            elif y_index == "u":
                target_y_list.append(self.u)
            elif y_index == "v":
                target_y_list.append(self.v)
            elif y_index == "r":
                target_y_list.append(self.r)
            elif y_index == "x":
                target_y_list.append(self.x)
            elif y_index == "y":
                target_y_list.append(self.y)
            elif y_index == "psi":
                target_y_list.append(self.psi)
            elif y_index == "delta":
                target_y_list.append(self.δ)
            elif y_index == "δ":
                target_y_list.append(self.δ)
            elif y_index == "npm":
                target_y_list.append(self.npm)
        if len(target_y_list) == 0:
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
                ", "
                " delta"
                ", "
                " δ"
                ", "
                " npm"
                "]"
                "]"
            )
        fig = plt.figure(
            num=num,
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            frameon=frameon,
            FigureClass=FigureClass,
            clear=clear,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout,
        )
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if fmt is None:
            for target_y in target_y_list:
                plt.plot(target_x, target_y, **kwargs)
        else:
            for target_y in target_y_list:
                plt.plot(target_x, target_y, fmt, **kwargs)
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
        num: int or str = None,
        figsize: List[float] = [6.4, 4.8],
        dpi: float = 100.0,
        facecolor: str = None,
        edgecolor: str = None,
        frameon: bool = True,
        FigureClass: matplotlib.figure.Figure = matplotlib.figure.Figure,
        clear: bool = False,
        tight_layout: bool = False,
        constrained_layout: bool = False,
        fmt: str = "--k",
        save_fig_path: str = None,
        **kwargs
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
            num (int or str, optional):
                A unique identifier for the figure.
                If a figure with that identifier already exists, this figure is made active and returned.
                An integer refers to the Figure.number attribute, a string refers to the figure label.
                If there is no figure with the identifier or num is not given,
                a new figure is created, made active and returned.
                If num is an int, it will be used for the Figure.number attribute.
                Otherwise, an auto-generated integer value is used (starting at 1 and incremented for each new figure).
                If num is a string, the figure label and the window title is set to this value.
                Default to None.
            figsize ((float, float), optional):
                Width, height in inches.
                Default to [6.4, 4.8]
            dpi (float, optional):
                The resolution of the figure in dots-per-inch.
                Default to 100.0.
            facecolor (str, optional):
                The background color.
            edgecolor (str, optional):
                The border color.
            frameon (bool, optional):
                If False, suppress drawing the figure frame.
                Defaults to True.
            FigureClass (subclass of matplotlib.figure.Figure, optional):
                Optionally use a custom Figure instance.
                Defaults to matplotlib.figure.Figure.
            clear (bool, optional):
                If True and the figure already exists, then it is cleared.
                Defaults to False.
            tight_layout (bool, optional):
                If False use subplotpars.
                If True adjust subplot parameters using tight_layout with default padding.
                When providing a dict containing the keys pad, w_pad, h_pad, and rect,
                the default tight_layout paddings will be overridden.
                Defaults to False.
            constrained_layout (bool, optional):
                If True use constrained layout to adjust positioning of plot elements.
                Like tight_layout, but designed to be more flexible.
                See Constrained Layout Guide for examples.
                (Note: does not work with add_subplot or subplot2grid.)
                Defaults to False.
            fmt (str, optional):
                A format string, e.g. 'ro' for red circles.
                See the Notes section for a full description of the format strings.
                Format strings are just an abbreviation for quickly setting basic line properties.
                All of these and more can also be controlled by keyword arguments.
                This argument cannot be passed as keyword.
                Defaults to "--k".
            save_fig_path (str, optional):
                Path of saving figure.
                Defaults to None.
            **kwargs (matplotlib.lines.Line2D properties, optional):
                kwargs are used to specify properties
                like a line label (for auto legends), linewidth, antialiasing, marker face color.
                You can show the detailed information at `matplotlib.lines.Line2D
                 <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_

        Examples:
            >>> ship.draw_gif(save_fig_path='test.gif')
        """

        fig = plt.figure(
            num=num,
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            frameon=frameon,
            FigureClass=FigureClass,
            clear=clear,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout,
        )
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

        if fmt is not None:
            plt.plot(draw_x, draw_y, fmt, **kwargs)
        else:
            plt.plot(draw_x, draw_y, ls="--", color="k", **kwargs)

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

            return drawer.draw_obj_with_angle(
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
