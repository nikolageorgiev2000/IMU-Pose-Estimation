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

        d_eS = 2 * ein('ti,tj ->tij', calibrated, a_min_b) # Tx3x3
        d_eb = 2 * ein('ij,ti -> tj', -S, calibrated) # Tx3

        self.t_residual = f
        self.t_derivs = torch.cat([d_eS.reshape(len(accels),9), d_eb], dim=-1)  # Tx12

    def objective_update(self):
        self.t_obj = ein('t, t', self.t_residual, self.t_residual)

    def state_update(self):

        state_mat = torch.cat(
            [self.state[0].reshape(9), self.state[1]])

        J = self.t_derivs  # Tx12
        JJT = ein('ti, tj -> ij', J, J)  # 12x12
        G = ein('ti, t -> i', J, self.t_residual)  # 12
        S_trust_scale = torch.eye(12)
        # S_trust_scale = 0
        new_state = state_mat - \
            torch.linalg.solve(JJT + S_trust_scale *
                            self.distrust_lambda, G)  # 4x3

        self.t_state = [new_state[:9].reshape((3,3)), new_state[9:]]

    def apply_model(self, data):
        Sinv = self.state[0]
        b = self.state[1]
        return ein('ij, tj -> ti', Sinv, data - b)
