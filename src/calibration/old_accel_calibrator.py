from src import *
from src.optim import levmarq

class Accel_Calibrator(levmarq.LM):

    def residual_update(self):
        accels, grav = self.data

        S = self.t_state[0]
        b = self.t_state[1]

        a_min_b = (accels - b)
        calibrated = ein('ij, tj -> ti', S, a_min_b)
        f = ein('ti->t', calibrated**2) - grav**2

        d_eS = 2 * ein('ti,tj ->tij', calibrated, a_min_b)
        d_eS = torch.squeeze(d_eS)  # Tx3x3
        d_eb = 2 * ein('ij,tj -> ti', -S, calibrated)
        d_eb = d_eb[..., None, :]  # Tx1x3

        self.t_residual = f
        self.t_derivs = torch.cat([d_eS, d_eb], dim=-2)  # Tx4x3

    def objective_update(self):
        self.t_obj = ein('t, t', self.t_residual, self.t_residual)

    def state_update(self):

        state_mat = torch.cat(
            [self.state[0], self.state[1][..., None, :]], dim=-2)

        J = self.t_derivs  # Tx4x3
        JJT = ein('tij, tkj -> ik', J, J)  # 4x4
        G = ein('tij, t -> ij', J, self.t_residual)  # 4x3
        # S_trust_scale = JJT * torch.eye(4)
        S_trust_scale = 0
        new_state = state_mat - \
            torch.linalg.solve(JJT + S_trust_scale *
                            self.distrust_lambda, G)  # 4x3

        self.t_state = [new_state[:3], new_state[3]]

    def apply_model(self, data):
        Sinv = self.state[0]
        b = self.state[1]
        return ein('ij, tj -> ti', Sinv, data - b)
