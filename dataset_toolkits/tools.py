from math import *
import numpy as np
def VaryPoint(data, axis, degree):
            xyzArray = {
                'X': np.array([[1, 0, 0],
                        [0, cos(radians(degree)), -sin(radians(degree))],
                        [0, sin(radians(degree)), cos(radians(degree))]]),
                'Y': np.array([[cos(radians(degree)), 0, sin(radians(degree))],
                        [0, 1, 0],
                        [-sin(radians(degree)), 0, cos(radians(degree))]]),
                'Z': np.array([[cos(radians(degree)), -sin(radians(degree)), 0],
                        [sin(radians(degree)), cos(radians(degree)), 0],
                        [0, 0, 1]])}
            newData = np.dot(data, xyzArray[axis])
            return newData