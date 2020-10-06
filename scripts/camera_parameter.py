import math
from math import pi
import numpy as np

PI = pi


class CameraParam:
    def __init__(self):
        self.width = None
        self.height = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def set_image_size(self, width, height):
        self.width = width
        self.height = height

    def set_intrinsic_parameter(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def k(self):
        return self.fx, 0., self.cx, 0., self.fy, self.cy, 0., 0., 1.

    @property
    def intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    @property
    def intrinsic_matrix(self):
        return np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])

    @property
    def image_size(self):
        return self.width, self.height

    @property
    def center(self):
        return self.cx, self.cy

    @property
    def focal(self):
        return self.fx, self.fy