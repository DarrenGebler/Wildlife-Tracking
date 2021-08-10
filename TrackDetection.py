from Georeference import GeoReference
from filterpy.kalman import KalmanFilter
import numpy as np
from math import pow


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
    Creates a 3x2 control matrix
    Parameters
    ----------
    delta_t : Time difference

    Returns
    -------
    3x2 control matrix
    """
    g = np.zeros((3, 2), dtype=float)
    g[0, 0] = 0.5 * pow(delta_t, 2)
    g[1, 1] = 0.5 * pow(delta_t, 2)
    g[2, 0] = delta_t
    g[3, 1] = delta_t
    return g


class TrackDetection:
    def __init__(self, x, y, yaw, pitch, roll, lat, lon, alt):
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
        """
        self.geo_reference = GeoReference()
        self.ned_coords = self.geo_reference.calculate_xy_ned(x, y, yaw, pitch, roll, lat, lon, alt)
        self.kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
        self.kalman_filter_init()

    def kalman_filter_init(self):
        """
        Initiates Kalman filter by creating predefined matrices
        """
        self.kalman_filter.x = self.ned_coords[:2]
        self.kalman_filter.F = state_transition(0.0)
        self.kalman_filter.B = control(0.0)

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
