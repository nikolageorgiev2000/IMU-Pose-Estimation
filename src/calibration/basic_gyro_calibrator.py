from src import *
from src.optim import levmarq
from src.math3d.rotations import *

class Gyro_Calibrator(levmarq.LM):
    
    def residual_update(self):
        dt, rot_vecs, magnos, offset = self.data

        S = self.t_state
        a_min_b = (rot_vecs - offset)
        calibrated = ein('ij, tj -> ti', S, a_min_b)

        rot_mats = exp_rotmat(-dt*calibrated)

        v = ein('tij, tj -> ti', rot_mats, magnos[:-1])
        f = v - magnos[1:]
        # print(f)

        d_eS: FT = ein('t, tpi, tj -> tijp', dt[:,0],
                         skew_symm(magnos[:-1]), a_min_b) # Tx3x3xP

        self.t_residual = f # Tx3
        self.t_derivs = d_eS.reshape((len(dt),9,f.shape[-1]))

    def objective_update(self):
        self.t_obj = ein(
            'tp, tp', self.t_residual, self.t_residual)

    def state_update(self):
        state_mat = self.state.reshape(9)

        J = self.t_derivs  # Tx12xP
        JJT = ein('tip, tjp -> ij', J, J)  # 12x12
        G = ein('tip, tp -> i', J, self.t_residual)  # 12
        S_trust_scale = JJT * torch.eye(9)
        S_trust_scale = 0
        new_state = state_mat - \
            torch.linalg.solve(JJT + S_trust_scale *
                            self.distrust_lambda, G)  # 4x3

        self.t_state = new_state.reshape((3,3))

    def apply_model(self, data):
        Sinv = self.state
        b = self.data[:-1]
        return ein('ij, tj -> ti', Sinv, data - b)