import math

import numpy as np

from Tracking.utils.Georeference import GeoReference
from Tracking.utils.ObservationTracks import Track
from Tracking.HungarianAlgorithm.Hungarian import Hungarian
from scipy.optimize import linear_sum_assignment

gate_threshold = 300


def observation_within_gate(predicted_position, observations):
    """
    Calculates distance between predicted position and all observations. Adds them to a row. If not in gated threshold,
    append 100
    Parameters
    ----------
    predicted_position : Predicted position of object based on Kalman filter
    observations : all observation positions

    Returns
    -------
    Row of cost matrix
    """
    within_gate = []
    for observation in observations:
        distance = np.linalg.norm(observation - predicted_position)
        if np.linalg.norm(observation - predicted_position) < gate_threshold:
            within_gate.append(distance)
        else:
            within_gate.append(301.0)

    return within_gate


class GlobalNearestNeighbour:
    """
    Implementation of Global Nearest Neighbour algorithm.
    """

    def __init__(self):
        """
        Initialises GNN algorithm by creating empty list of tracks.
        """
        self.tracks = []
        self.geo_reference = GeoReference()

    def __add_track(self, observation, yaw, pitch, roll, lat, lon, alt, delta_t):
        """
        Adds new Track for newly identified object
        Parameters
        ----------
        observation : position of observed object
        yaw : current yaw of gimbal
        pitch : current pitch of gimbal
        roll : current roll of gimbal
        lat : current latitude of drone
        lon : current longitude of drone
        alt : current altitude of drone
        delta_t : time difference
        """
        self.tracks.append(Track(observation[0], observation[1], yaw, pitch, roll, lat, lon,
                                 alt, delta_t))

    def __observations_to_ned(self, observations, yaw, pitch, roll, lat, lon, alt):
        """
        Converts all observations to NED coordinates
        Parameters
        ----------
        observations : All observed object locations
        yaw : current yaw of gimbal
        pitch : current pitch of gimbal
        roll : current roll of gimbal
        lat : current latitude of drone
        lon : current longitude of drone
        alt : current altitude of drone

        Returns
        -------
        Converted observations
        """
        converted_observations = []
        for observation in observations:
            converted_observations.append(
                self.geo_reference.calculate_xy_ned(observation[0], observation[1], yaw, pitch, roll, lat, lon,
                                                    alt))
        return converted_observations

    def update_tracks(self, observations, yaw, pitch, roll, lat, lon, alt, delta_t):
        """
        Updates all tracks by first predicting next positions, and creating a Cost Matrix
        Parameters
        ----------
        observations : All observed object locations
        yaw : current yaw of gimbal
        pitch : current pitch of gimbal
        roll : current roll of gimbal
        lat : current latitude of drone
        lon : current longitude of drone
        alt : current altitude of drone
        delta_t : time difference
        """
        if len(self.tracks) == 0 and len(observations) != 0:
            for observation in observations:
                self.__add_track(observation, yaw, pitch, roll, lat, lon, alt, delta_t)
            return
        elif len(self.tracks) == 0 and len(observations) == 0:
            return
        elif len(self.tracks) != 0 and len(observations) == 0:
            for track in self.tracks:
                if (track.get_prior_x() == [[0.], [0.], [0.], [0.]]).all():
                    continue
                track.predict_position(delta_t)
            return

        observations = self.__observations_to_ned(observations, yaw, pitch, roll, lat, lon, alt)
        cost_matrix = np.zeros((len(self.tracks), len(observations)))

        for index, track in enumerate(self.tracks):
            position = track.predict_position(delta_t)[:2]
            cost_matrix[index] = observation_within_gate(position, observations)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] == 301.0:
                self.__add_track(observations[row], yaw, pitch, roll, lat, lon, alt, delta_t)

    def get_tracks(self):
        return self.tracks
