from src import *
from src.optim import levmarq

class DipCalibrator(levmarq.LM):

    def residual_update(self):
        accels, grav, still_cal_magnos = self.data

        Sa = self.t_state[0]
        ba = self.t_state[1]
        dip_cos = self.t_state[2]

        S = len(accels)

        a_min_b = (accels - ba)
        calibrated_a = ein('ij, tj -> ti', Sa, a_min_b)
        fa = ein('ti->t', calibrated_a**2) - grav**2 # S

        d_eSa = 2 * ein('ti,tj ->tij', calibrated_a, a_min_b).reshape((S,9)) # Sx9
        d_eba = 2 * ein('ij,ti -> tj', -Sa, calibrated_a).reshape((S,3)) # Sx3

        weight = 0.1
        fa *= weight
        d_eSa *= weight
        d_eba *= weight

        fd = ein('ti,ti->t', calibrated_a, still_cal_magnos) - dip_cos # S

        d_fSa = ein('tj,ti->tij', a_min_b, still_cal_magnos).reshape((S,9)) # Sx9 
        d_fba = ein('ij,ti->tj',-Sa,still_cal_magnos) # Sx3
        d_fdip = -torch.ones((S,1)) # Sx1


        self.t_residual = torch.cat((fa, fd)) # S+T+S
        self.t_derivs = torch.zeros((S+S,13)) # (S+T+S)x25
        self.t_derivs[:S,:12] = torch.cat((d_eSa,d_eba),dim=-1)
        self.t_derivs[S:,:] = torch.cat((d_fSa, d_fba, d_fdip),dim=-1)

    def objective_update(self):
        self.t_obj = ein('t, t', self.t_residual, self.t_residual)

    def state_update(self):

        state_mat = torch.cat(
            [self.state[0].reshape(9), self.state[1], self.state[2]])

        J = self.t_derivs  # (S+T+S)x25
        JJT = ein('ti, tj -> ij', J, J)  # 25x25
        G = ein('ti, t -> i', J, self.t_residual)  # 25
        S_trust_scale = torch.eye(13) * self.distrust_lambda
        # S_trust_scale = 0
        new_state = state_mat - \
            torch.linalg.solve(JJT + S_trust_scale, G)  # 24
        # new_state = state_mat - 0.01 * torch.linalg.lstsq(J, self.t_residual)[0]

        self.t_state = [new_state[:9].reshape((3,3)), new_state[9:12], FT([new_state[-1]])]

    def apply_model(self, data):
        Sinv = self.state[0]
        b = self.state[1]
        return ein('ij, tj -> ti', Sinv, data - b)
