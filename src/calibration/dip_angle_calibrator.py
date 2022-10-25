from src import *
from src.optim import levmarq

class DipCalibrator(levmarq.LM):

    def residual_update(self):
        accels, grav, magnos, still_magnos = self.data

        Sa = self.t_state[0]
        ba = self.t_state[1]
        Sm = self.t_state[2]
        bm = self.t_state[3]
        dip_cos = self.t_state[4]

        S = len(accels)
        T = len(magnos)

        a_min_b = (accels - ba)
        calibrated_a = ein('ij, tj -> ti', Sa, a_min_b)
        fa = ein('ti->t', calibrated_a**2) - grav**2 # S

        d_eSa = 2 * ein('ti,tj ->tij', calibrated_a, a_min_b).reshape((S,9)) # Sx9
        d_eba = 2 * ein('ij,tj -> tj', -Sa, calibrated_a).reshape((S,3)) # Sx3

        m_min_b = (magnos - bm)
        calibrated_m = ein('ij, tj -> ti', Sm, m_min_b)
        fm = ein('ti->t', calibrated_m**2) - 1.0 # T

        d_eSm = 2 * ein('ti,tj ->tij', calibrated_m, m_min_b).reshape((T,9)) # Tx9
        d_ebm = 2 * ein('ij,tj -> tj', -Sm, calibrated_m).reshape((T,3)) # Tx3

        s_min_b = (still_magnos - bm)
        calibrated_s = ein('ij, tj -> ti', Sm, s_min_b)
        fd = ein('ti,ti->t', a_min_b, s_min_b) - dip_cos # S

        d_fSa = ein('tj,ti->tij', a_min_b, calibrated_s).reshape((S,9)) # Sx9 
        d_fba = ein('ij,ti->tj',-Sa,calibrated_s) # Sx3
        d_fSm = ein('tj,ti->tij', s_min_b, calibrated_a).reshape((S,9)) # Sx9 
        d_fbm = ein('ij,ti->tj',-Sm,calibrated_a) # Sx3
        d_fdip = -torch.ones((S,1)) # Sx1


        self.t_residual = torch.cat((fa, fm, fd)) # S+T+S
        self.t_derivs = torch.zeros((S+T+S,25)) # (S+T+S)x25
        self.t_derivs[:S,:12] = torch.cat((d_eSa,d_eba),dim=-1)
        self.t_derivs[S:S+T,12:24] = torch.cat((d_eSm,d_ebm),dim=-1)
        self.t_derivs[S+T:,:] = torch.cat((d_fSa, d_fba, d_fSm, d_fbm, d_fdip),dim=-1)

    def objective_update(self):
        self.t_obj = ein('t, t', self.t_residual, self.t_residual)

    def state_update(self):

        state_mat = torch.cat(
            [self.state[0].reshape(9), self.state[1], self.state[2].reshape(9), self.state[3], self.state[4]])

        J = self.t_derivs  # (S+T+S)x25
        JJT = ein('ti, tj -> ij', J, J)  # 25x25
        G = ein('ti, t -> i', J, self.t_residual)  # 25
        S_trust_scale = torch.eye(25)
        # S_trust_scale = 0
        # new_state = state_mat - \
        #     torch.linalg.solve(JJT + S_trust_scale, G)  # 24
        new_state = state_mat - 0.01 * torch.linalg.lstsq(J, self.t_residual)[0]

        self.t_state = [new_state[:9].reshape((3,3)), new_state[9:12], new_state[12:-4].reshape((3,3)), new_state[-4:-1], FT([new_state[-1]])]

    def apply_model(self, data):
        Sinv = self.state[0]
        b = self.state[1]
        return ein('ij, tj -> ti', Sinv, data - b)
