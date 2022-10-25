
from src import *

from src.data_collection.posers import *
from src.math3d.rotations import exp_rotmat
import unittest


class PoserTests(unittest.TestCase):
    def setUp(self):
        # create synthetic data

        dp_count = 5000
        self.pds = PoseDerivs()
        self.pds.intervals = torch.ones((dp_count, 1))
        self.pds.a = torch.rand((dp_count, 3))
        self.pds.w = torch.rand((dp_count, 3)) * torch.deg2rad(FT([10]))

        def r3(): return torch.rand(3)
        nav = NavContext()
        init_pos = r3()
        init_vel = r3()
        init_rot = exp_rotmat(r3())
        init_time = FT([12345])
        init_params = (nav, init_pos, init_vel, init_rot, init_time)

        self.poser = self.pds.integrate(*init_params)

        self.imu = self.poser.generate_imu_record()

        # print(self.imu.spf.shape, self.imu.gyro.shape, self.imu.timestamps.shape)
        self.poser1 = self.imu.generate_pose(*init_params[:-1])

    def test_dead_reckoning(self):
        self.assertTrue(torch.allclose(
            self.poser.timestamps, self.poser1.timestamps))
        self.assertTrue(torch.allclose(self.poser.pos, self.poser1.pos))
        self.assertTrue(torch.allclose(self.poser.vel, self.poser1.vel))
        # print((self.poser.rot - self.poser1.rot)[-10:])
        self.assertTrue(torch.allclose(self.poser.rot, self.poser1.rot))

    def test_pds_imu(self):
        pds1 = self.poser1.differentiate()
        self.assertTrue(torch.allclose(pds1.w, self.imu.gyro[:-1]))
