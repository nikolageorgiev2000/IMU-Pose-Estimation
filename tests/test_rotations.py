
from src import *

from src.math3d import rotations
import unittest


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.vec1 = FT([1, 1, 1])
        self.vec2 = FT([[-1.5, 2, -torch.pi],
                                  [75.6, -0.00654321, 0],
                                  [1, -2, 0.0002005],
                                  [1, -2, 0.0002005], ])

        self.rotvec1 = FT([torch.pi, 0, 0])
        self.rotvec2 = rotations.normalize(
            FT([[1, 2, 3],
                          [0, 0, 0],
                          [-10000, 0.000034, 1],
                          [torch.pi/2, torch.pi/2, torch.pi/2], ]))
        self.rotmat1 = rotations.exp_rotmat(self.rotvec1)
        self.rotmat2 = rotations.exp_rotmat(self.rotvec2)

    def test_skewsymm_inv(self):
        n_hat1 = rotations.skew_symm(self.vec1)
        self.assertTrue(torch.allclose(self.vec1, rotations.inv_skew(n_hat1)))

        n_hat2 = rotations.skew_symm(self.vec2)
        self.assertTrue(torch.allclose(self.vec2, rotations.inv_skew(n_hat2)))

    def test_rotmat_log(self):
        self.assertTrue(torch.allclose(
            self.rotvec1, rotations.log_rotmat(self.rotmat1)))
        self.assertTrue(torch.allclose(
            self.rotvec2, rotations.log_rotmat(self.rotmat2)))

    def test_vector_rotation(self):
        res = FT([1,-1,-1])
        self.assertTrue(torch.allclose(self.rotmat1 @ self.vec1, res))
        self.assertTrue(torch.allclose(ein('ij,tj->ti', self.rotmat1, self.vec2), self.vec2 * res))

    def test_rotmat_inverse(self):
        self.assertTrue(torch.allclose(self.rotmat1.T @ self.rotmat1 @ self.vec1, self.vec1))
        self.assertTrue(torch.allclose(ein('ji,jk,tk->tk', self.rotmat1, self.rotmat1, self.vec2), self.vec2))