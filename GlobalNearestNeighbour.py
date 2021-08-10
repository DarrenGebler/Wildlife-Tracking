import numpy as np


gate_threshold = 10

class GlobalNearestNeighbour:
    def __init__(self):
        pass

    def gate(self, predicted_position, observations):
        within_gate = []
        for observation in observations:
            if np.linalg.norm(observation - predicted_position) < gate_threshold:
                within_gate.append(observation)

        return within_gate


