#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""draw_obj.

* Base class for drawing object not only for shipmmg.

"""

import numpy as np


class DrawObj:
    """DrawObj class.

    General class for drawing object by using matplotlib.animation.
    Multiple ships can be drawn by using this class.
    """

    def __init__(self, ax):
        """init."""
        self.ax = ax
        self.img = []
        self.img.append(ax.plot([], [], color="b"))
        self.img.append(ax.plot([], [], color="y"))

    def draw_obj_with_angle(
        self, center_x_list, center_y_list, shape_list, angle_list, obj="ship"
    ):
        """Draw square image with angle.

        Args:
            center_x_list (List[float]): list of the center x position of the square
            center_y_list (List[float]): list of the center y position of the square
            shape_list (List[float]): list of the square's shape(length/2, width/2)
            angle_list (List[float]): list of in radians
            obj (str: optional): object type, 'ship' or 'square'
        Returns:
            Image: List of Image
        """
        for i in range(len(shape_list)):
            if obj == "square":
                square_x, square_y, angle_x, angle_y = self.__square_with_angle(
                    center_x_list[i], center_y_list[i], shape_list[i], angle_list[i]
                )
            elif obj == "ship":
                square_x, square_y, angle_x, angle_y = self.__ship_with_angle(
                    center_x_list[i], center_y_list[i], shape_list[i], angle_list[i]
                )
            self.img[i][0].set_xdata(square_x)
            self.img[i][0].set_ydata(square_y)
        return self.img

    def __rotate_pos(self, pos, angle):
        """Transform the coordinate in the angle.

        Args:
            pos (numpy.ndarray): local state, shape(data_size, 2)
            angle (float): rotate angle, in radians
        Returns:
            rotated_pos (numpy.ndarray): shape(data_size, 2)
        """
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        return np.dot(pos, rot_mat.T)

    def __square(self, center_x, center_y, shape, angle):
        """Create square.

        Args:
            center_x (float): the center x position of the square
            center_y (float): the center y position of the square
            shape (tuple): the square's shape(width/2, height/2)
            angle (float): in radians
        Returns:
            square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
            square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
        """
        # start with the up right points
        # create point in counterclockwise, local
        square_xy = np.array(
            [
                [shape[0], shape[1]],
                [-shape[0], shape[1]],
                [-shape[0], -shape[1]],
                [shape[0], -shape[1]],
                [shape[0], shape[1]],
            ]
        )
        # translate position to world
        # rotation
        trans_points = self.__rotate_pos(square_xy, angle)
        # translation
        trans_points += np.array([center_x, center_y])

        return trans_points[:, 0], trans_points[:, 1]

    def __ship(self, center_x, center_y, shape, angle):
        """Create ship.

        Args:
            center_x (float): the center x position of the ship
            center_y (float): the center y position of the ship
            shape (tuple): the ship's shape(width/2, height/2)
            angle (float): in radians
        Returns:
            ship_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
            shipe_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
        """
        # start with the up right points
        # create point in counterclockwise, local
        ship_xy = np.array(
            [
                [shape[0] * 0.75, shape[1]],
                [-shape[0], shape[1]],
                [-shape[0], -shape[1]],
                [shape[0] * 0.75, -shape[1]],
                [shape[0], 0],
                [shape[0] * 0.75, shape[1]],
            ]
        )
        # translate position to world
        # rotation
        trans_points = self.__rotate_pos(ship_xy, angle)
        # translation
        trans_points += np.array([center_x, center_y])

        return trans_points[:, 0], trans_points[:, 1]

    def __square_with_angle(self, center_x, center_y, shape, angle):
        """Create square with angle line.

        Args:
            center_x (float): the center x position of the square
            center_y (float): the center y position of the square
            shape (tuple): the square's shape(width/2, height/2)
            angle (float): in radians
        Returns:
            square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
            square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
            angle_x (numpy.ndarray): x data of square angle
            angle_y (numpy.ndarray): y data of square angle
        """
        square_x, square_y = self.__square(center_x, center_y, shape, angle)

        angle_x = np.array([center_x, center_x + np.cos(angle) * shape[0]])
        angle_y = np.array([center_y, center_y + np.sin(angle) * shape[1]])

        return square_x, square_y, angle_x, angle_y

    def __ship_with_angle(self, center_x, center_y, shape, angle):
        """Create ship with angle line.

        Args:
            center_x (float): the center x position of the ship
            center_y (float): the center y position of the ship
            shape (tuple): the ship's shape(width/2, height/2)
            angle (float): in radians
        Returns:
            ship_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
            ship_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
            angle_x (numpy.ndarray): x data of ship angle
            angle_y (numpy.ndarray): y data of ship angle
        """
        ship_x, ship_y = self.__ship(center_x, center_y, shape, angle)

        angle_x = np.array([center_x, center_x + np.cos(angle) * shape[0]])
        angle_y = np.array([center_y, center_y + np.sin(angle) * shape[1]])

        return ship_x, ship_y, angle_x, angle_y
