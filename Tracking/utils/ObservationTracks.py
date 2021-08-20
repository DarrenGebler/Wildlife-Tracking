from math import pow

import numpy as np
from filterpy.kalman import KalmanFilter

from Tracking.utils.Georeference import GeoReference

gate_threshold = 10


def state_transition(delta_t):
    """
    Creates a 4x4 state transition matrix.
    Parameters
    ----------
    delta_t : Time difference

    Returns
    -------
    4x4 state transition matrix
    """
    f = np.eye(4, dtype=float)
    f[0, 2] = delta_t
    f[1, 3] = delta_t
    return f


def control(delta_t):
    """
    Creates a 4x2 control matrix
    Parameters
    ----------
    delta_t : Time difference

    Returns
    -------
    4x2 control matrix
    """
    g = np.zeros((4, 2), dtype=float)
    g[0, 0] = 0.5 * pow(delta_t, 2)
    g[1, 1] = 0.5 * pow(delta_t, 2)
    g[2, 0] = delta_t
    g[3, 1] = delta_t
    return g


def observation_within_gate(predicted_position, observations):
    within_gate = []
    for observation in observations:
        if np.linalg.norm(observation - predicted_position) < gate_threshold:
            within_gate.append(observation)

    return within_gate


class Track:
    def __init__(self, x, y, yaw, pitch, roll, lat, lon, alt, delta_t):
        """
        Initiates Track Detection class when new object detected.
        Parameters
        ----------
        x : object position in image frame
        y : object position in image frame
        yaw : current yaw of gimbal
        pitch : current pitch of gimbal
        roll : current roll of gimbal
        lat : current latitude of drone
        lon : current longitude of drone
        alt : current altitude of drone
        delta_t : time of detection
        """
        self.geo_reference = GeoReference()
        self.ned_coords = self.geo_reference.calculate_xy_ned(x, y, yaw, pitch, roll, lat, lon, alt)
        self.kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
        self.__kalman_filter_init(delta_t)

    def __kalman_filter_init(self, delta_t):
        """
        Initiates Kalman filter by creating predefined matrices
        """
        self.kalman_filter.x = np.concatenate((self.ned_coords[:2], self.kalman_filter.R[:, :1]))
        self.kalman_filter.F = state_transition(delta_t)
        self.kalman_filter.B = control(delta_t)

    def residual_covariance(self):
        """
        Potential calculation required to determine distance to observation. d^2 + ln|{return}|

        Returns
        -------
        Residual covariance matrix
        """
        h = self.kalman_filter.H
        p = self.kalman_filter.P
        r = self.kalman_filter.R

        return h * p * np.transpose(h) + r

    def predict_position(self, delta_t):
        """
        Predicts objects next state position
        Parameters
        ----------
        delta_t : time difference

        Returns
        -------
        Predicted position
        """
        f = state_transition(delta_t)
        b = control(delta_t)
        self.kalman_filter.predict(F=f, B=b)

        return self.kalman_filter.x

    def update_position(self, x, y, yaw, pitch, roll, lat, lon, alt):
        """
        Updates the position of object detected.
        Parameters
        ----------
        x : object position in image frame
        y : object position in image frame
        yaw : current yaw of gimbal
        pitch : current pitch of gimbal
        roll : current roll of gimbal
        lat : current latitude of drone
        lon : current longitude of drone
        alt : current altitude of drone

        Returns
        -------
        Updated position
        """
        self.ned_coords = self.geo_reference.calculate_xy_ned(x, y, yaw, pitch, roll, lat, lon, alt)
        self.kalman_filter.update(self.ned_coords[:2])
        return self.kalman_filter.x
