from src import *
from src.optim import levmarq

class Accel_Magno_Calibrator(levmarq.LM):

    def residual_update(self):
        accels, grav, magnos = self.data

        Sa = self.t_state[0]
        ba = self.t_state[1]
        Sm = self.t_state[2]
        bm = self.t_state[3]
        # dip_cos = self.t_state[4]

        S = len(accels)
        T = len(magnos)

        a_min_b = (accels - ba)
        calibrated_a = ein('ij, tj -> ti', Sa, a_min_b)
        fa = ein('ti->t', calibrated_a**2) - grav**2

        d_eSa = 2 * ein('ti,tj ->tij', calibrated_a, a_min_b).reshape((S,9)) # Sx9
        d_eba = 2 * ein('ij,tj -> tj', -Sa, calibrated_a).reshape((S,3)) # Sx3

        m_min_b = (magnos - bm)
        calibrated_m = ein('ij, tj -> ti', Sm, m_min_b)
        fm = ein('ti->t', calibrated_m**2) - 1.0

        d_eSm = 2 * ein('ti,tj ->tij', calibrated_m, m_min_b).reshape((T,9)) # Tx9
        d_ebm = 2 * ein('ij,tj -> tj', -Sm, calibrated_m).reshape((T,3)) # Tx3

        self.t_residual = torch.cat((fa, fm)) # S+T
        self.t_derivs = torch.zeros((S+T,24)) # (S+T)x24
        self.t_derivs[:S,:12] = torch.cat((d_eSa,d_eba),dim=-1)
        self.t_derivs[S:,12:] = torch.cat((d_eSm,d_ebm),dim=-1)

    def objective_update(self):
        self.t_obj = ein('t, t', self.t_residual, self.t_residual)

    def state_update(self):

        state_mat = torch.cat(
            [self.state[0].reshape(9), self.state[1], self.state[2].reshape(9), self.state[3]])

        J = self.t_derivs  # (S+T)x24
        JJT = ein('ti, tj -> ij', J, J)  # 24x24
        G = ein('ti, t -> i', J, self.t_residual)  # 24
        S_trust_scale = torch.eye(24) * 0.01
        # S_trust_scale = 0
        # new_state = state_mat - \
        #     torch.linalg.solve(JJT + S_trust_scale, G)  # 24
        new_state = state_mat - torch.linalg.pinv(JJT) @ G

        self.t_state = [new_state[:9].reshape((3,3)), new_state[9:12], new_state[12:-3].reshape((3,3)), new_state[-3:]]

    def apply_model(self, data):
        Sinv = self.state[0]
        b = self.state[1]
        return ein('ij, tj -> ti', Sinv, data - b)
