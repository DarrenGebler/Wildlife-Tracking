import numpy as np

from Georeference import GeoReference
from ObservationTracks import Track

gate_threshold = 10


def observation_within_gate(predicted_position, observations):
    within_gate = []
    for observation in observations:
        distance = np.linalg.norm(observation - predicted_position)
        if np.linalg.norm(observation - predicted_position) < gate_threshold:
            within_gate.append(distance)
        else:
            within_gate.append(100)

    return within_gate


class GlobalNearestNeighbour:
    def __init__(self):
        self.tracks = []
        self.geo_reference = GeoReference()

    def __add_track(self, observation, yaw, pitch, roll, lat, lon, alt, delta_t):
        self.tracks.append(Track(observation[0], observation[1], yaw, pitch, roll, lat, lon,
                                 alt, delta_t))

    def __observations_to_ned(self, observations, yaw, pitch, roll, lat, lon, alt):
        converted_observations = []
        for observation in observations:
            converted_observations.append(
                self.geo_reference.calculate_xy_ned(observation[0], observation[1], yaw, pitch, roll, lat, lon,
                                                    alt))
        return converted_observations

    def update_tracks(self, observations, yaw, pitch, roll, lat, lon, alt, delta_t):
        if len(self.tracks) == 0:
            for observation in observations:
                self.__add_track(observation, yaw, pitch, roll, lat, lon, alt, delta_t)
            return

        observations = self.__observations_to_ned(observations, yaw, pitch, roll, lat, lon, alt)
        cost_matrix = np.zeros((len(self.tracks), len(observations)))

        for index, track in enumerate(self.tracks):
            position = track.predict_position(delta_t)
            cost_matrix[index] = observation_within_gate(position, observations)
