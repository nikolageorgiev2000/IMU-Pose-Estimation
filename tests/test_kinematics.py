
from src import *

from src.math3d.kinematics import *
import unittest


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        # create synthetic data

        dp_per_axis = 10
        axes = [FT([1, 0, 0]), FT([0, 1, 0]), FT([0, 0, 1])]
        dp_count = dp_per_axis * len(axes)

        step = 2*torch.pi/dp_per_axis
        angles = torch.arange(0, dp_per_axis, 1) * step
        angle_axis = torch.cat([ein('t,a->ta', angles, a) for a in axes])
        self.rots = rotations.exp_rotmat(angle_axis)
        self.rotvecs = [step*a for a in axes]

    def test_rot_diff(self):
        w = get_angular_velocity(self.rots, FT([1]))
        print(w[0] - self.rotvecs[0])