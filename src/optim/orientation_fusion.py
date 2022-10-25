from src.optim.gdl import *
from src.optim.flsmoother import FLSmoother

class OrientFuse():
    def __init__(self, cutoff: int, learning_rate, init_rots: FT, rlimu: IMU_Record, nav: NavContext, cov_inv_sqrts: Tuple[FT], verbose=False) -> None:
        self.cutoff = cutoff
        self.learning_rate = learning_rate
        self.init_rots = init_rots
        self.rlimu = rlimu
        self.dt = torch.diff(self.rlimu.timestamps, dim=0) / 1e6

        self.nav = nav
        self.cov_inv_sqrts = cov_inv_sqrts

        self.smoother = FLSmoother(self.cutoff, 1000, verbose)

    def run(self):
        def create_pose_state():
            return VState(torch.eye(3), FT([0, 0, 0]), lambda v, d: exp_rotmat(self.learning_rate * d) @ v, self.motion_prior)
        
        prev_vnode = VNode('pos0', create_pose_state())
        prev_vnode.state.value = self.init_rots[0]
        self.smoother.add_data([prev_vnode], [FNode('motionP', [prev_vnode], [self.init_rots[0]], self.motion_prior)])

        init_magno_factor = FNode(f'magno{0}', [prev_vnode], [self.rlimu.magno[0], self.nav.north], self.magno_updater)
        self.smoother.add_data([prev_vnode], [init_magno_factor])

        for i in range(1,len(self.init_rots)):
            ind = str(i).zfill(5)
            new_var = VNode(f'pos{ind}', create_pose_state())
            new_var.state.value = prev_vnode.get_value()

            new_factor1 = FNode(f'motion{ind}', [prev_vnode, new_var], [-(self.rlimu.gyro[:-1] * self.dt)[i-1]], self.motion_updater)
            new_factor2 = FNode(f'magno{ind}', [new_var], [self.rlimu.magno[i], self.nav.north], self.magno_updater)

            # new_factor3 = FNode(f'spf{ind}', [new_var], [self.rlimu.spf[i], self.nav.grav], self.spf_updater)
            # smoother.add_data([new_var],[new_factor3])

            self.smoother.add_data([new_var],[new_factor1, new_factor2])

            if i%1 == 0 and i >= self.smoother.max_size:
                if i>5:
                    objs = self.smoother.iterate(1000)
                else:
                    objs = self.smoother.iterate(100000)
                
                if i%1000 == 0:
                    print(objs)

            prev_vnode = new_var

    def get_rots(self):
        return torch.stack([v.get_value() for v in self.smoother.gdl.sort_variables(self.smoother.variables)])


    def motion_prior(self, params: List[FT], vals: List[FT]) -> Tuple[FT, FT, List[FT]]:
        rot0 = params[0]
        curr = vals[0]

        rot_diff = curr @ rot0.T
        residual = - log_rotmat(rot_diff)
        jacobian = [ein('pri,mpq,qr->mi', levi_civita, levi_civita, rot_diff)/2]

        c = self.cov_inv_sqrts[3]
        residual = c @ residual
        jacobian[0] = c @ jacobian[0]
        jacobian[0] = jacobian[0].T  # need to flip because MFQR uses numerator notation

        objective = torch.dot(residual, residual)

        return objective, residual, jacobian


    def motion_updater(self, params: List[FT], vals: List[FT]) -> Tuple[FT, FT, List[FT]]:
        motion_delta = params[0]
        prev, curr = vals

        rot_diff = ein('ij,kj->ik', curr, prev)
        # z - h(x)
        residual = motion_delta - log_rotmat(rot_diff)
        # H(x) (jacobian of h(x))
        jacobian = [-ein('pri,mpq,rq->mi', levi_civita, levi_civita, rot_diff)/2,
                    ein('pri,mpq,qr->mi', levi_civita, levi_civita, rot_diff)/2]

        c = self.cov_inv_sqrts[1]
        residual = c @ residual
        jacobian[0] = c @ jacobian[0]
        jacobian[1] = c @ jacobian[1]
        jacobian[0] = jacobian[0].T  # need to flip because MFQR uses numerator notation
        jacobian[1] = jacobian[1].T  # need to flip because MFQR uses numerator notation


        objective = torch.dot(residual, residual)

        return objective, residual, jacobian

    def magno_updater(self, params: List[FT], vals: List[FT]) -> Tuple[FT, FT, List[FT]]:
        magno = params[0]
        north = params[1]
        curr = vals[0]

        # z - h(x) 
        residual = magno - (curr @ north)
        # H(x) (jacobian of h(x))
        jacobian = [skew_symm(curr @ north)]

        c = self.cov_inv_sqrts[2]
        residual = c @ residual
        jacobian[0] = c @ jacobian[0]
        jacobian[0] = jacobian[0].T  # need to flip because MFQR uses numerator notation

        objective = torch.dot(residual, residual)

        return objective, residual, jacobian

    def spf_updater(self, params: List[FT], vals: List[FT]) -> Tuple[FT, FT, List[FT]]:
        spf = params[0]
        grav = params[1]
        curr = vals[0]

        # z - h(x) 
        residual = spf - (curr @ -grav)

        # H(x) (jacobian of h(x))
        jacobian = [skew_symm(curr @ -grav)]

        c = self.cov_inv_sqrts[0]
        residual = c @ residual
        jacobian[0] = c @ jacobian[0]
        jacobian[0] = jacobian[0].T  # need to flip because MFQR uses numerator notation

        objective = torch.dot(residual, residual)

        return objective, residual, jacobian