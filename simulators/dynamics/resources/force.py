import pybullet as p
import os
import math
import numpy as np
from simulators.dynamics.resources.utils import *
from scipy.spatial.transform import Rotation

class Force:
    def __init__(self, client):
        self.urdf = " arrow.urdf"
        self.id = p.loadURDF("/home/buzi/Desktop/Princeton/RESEARCH/SAFE/safe_adaptation_dev/simulators/dynamics/resources/arrow.urdf", [0, 0, 0], useFixedBase=True, globalScaling=0.01)

    def reset(self, position, orientation):
        p.resetBasePositionAndOrientation(self.id, position, orientation)
    
    def resetAB(self, origin, heading):
        arrow_direction = heading - origin
        arrow_length = np.linalg.norm(arrow_direction)
        arrow_direction = arrow_direction / arrow_length
        axis = np.cross(np.array([0, 0, 1]), arrow_direction)
        angle = np.arccos(np.dot(np.array([0, 0, 1]), arrow_direction))
        orientation = p.getQuaternionFromAxisAngle(axis, angle)
        self.reset(origin, orientation)
