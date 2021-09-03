from math import cos as cos
from math import sin as sin

import numpy as np
import pymap3d as pm


def calculate_r(yaw, pitch, roll):
    """
    Forms the rotational matrix
    Parameters
    ----------
    yaw : Gimbal yaw
    pitch : Gimbal pitch
    roll : Gimbal roll

    Returns
    -------
    Rotational matrix
    """
    yaw_matrix = np.array([[cos(yaw), -sin(yaw), 0],
                           [sin(yaw), cos(yaw), 0],
                           [0, 0, 1]])
    pitch_matrix = np.array([[cos(pitch), 0, sin(pitch)],
                             [0, 1, 0],
                             [-sin(pitch), 0, cos(pitch)]])
    roll_matrix = np.array([[1, 0, 0],
                            [0, cos(roll), -sin(roll)],
                            [0, sin(roll), cos(roll)]])

    return np.matmul(np.matmul(yaw_matrix, pitch_matrix), roll_matrix)


def geodetic_to_ecef(lat, lon, alt):
    """
    Converts latitude, longitude and altitude to ECEF coordinates

    Parameters
    ----------
    lat : IMU latitude
    lon : IMU longitude
    alt : IMU altitude

    Returns
    -------
    Converted location
    """
    t_x, t_y, t_z = pm.geodetic2ecef(lat, lon, alt)
    return np.array([[t_x], [t_y], [t_z]])


class GeoReference:
    def __init__(self, scale=1/100000000, c_x=320, c_y=256, focal_length=13, pixel_size=0.017):
        """
        Assumes pin hole camera model.

        Parameters
        ----------
        scale : Scale factor
        c_x : Principle point x
        c_y : Principle point y
        focal_length : Focal length of Zenmuse XT2
        pixel_size : Pixel size of Zenmuse XT2
        """
        self.scale = scale
        self.c_x = c_x
        self.c_y = c_y
        self.fs_x = focal_length * pixel_size
        self.fs_y = focal_length * pixel_size  # Check because frame not square, but rectangle. Times by 512/640?
        self.scaled_k = self.scale * self.__k()  # Assume pin hole camera

    def __k(self):
        """
        Determines the intrinsic camera parameters

        Returns
        -------
        k : intrinsic camera parameter matrix
        """
        return np.array([[self.fs_x, 0, self.c_x],
                         [0, self.fs_y, self.c_y],
                         [0, 0, 1]])

    def calculate_xy_ned(self, x_detection, y_detection, gimbal_yaw, gimbal_pitch, gimbal_roll, lat, lon, alt):
        """
        Convert detections in pixel coordinates to NED coordinates

        Parameters
        ----------
        x_detection : Object detection x location in camera frame
        y_detection : Object detection y location in camera frame
        gimbal_yaw : Gimbal yaw in degrees
        gimbal_pitch : Gimbal pitch in degrees
        gimbal_roll : Gimbal roll in degrees
        lat : IMU latitude
        lon : IMU longitude
        alt : IMU altitude

        Returns
        -------
        x and y in NED frame
        """
        gimbal_r = calculate_r(gimbal_yaw, gimbal_pitch, gimbal_roll)
        t = geodetic_to_ecef(lat, lon, alt)
        rot_trans = np.concatenate((gimbal_r[:, :2], t), axis=1)
        scaled_rot_trans = np.matmul(self.scaled_k, rot_trans)
        detection_coords = np.array([[x_detection], [y_detection], [1]])

        return np.matmul(scaled_rot_trans, detection_coords)[:2]



