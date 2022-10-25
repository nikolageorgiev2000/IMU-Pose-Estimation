class LM(object):
    def __init__(self, init_state, data, hyperparams=[1e10, 2, 0.8, 1e10, 1e-10, 1000], verbose=False, log_freq = 100):
        self.state = init_state
        self.obj = None
        self.data = data

        self.t_state = None
        self.t_obj = None
        self.t_residual = None
        self.t_derivs = None

        self.distrust_lambda = hyperparams[0]
        self.distrust_up, self.distrust_down, self.distrust_max, self.distrust_min, self.iterations = hyperparams[
            1:]
        self.verbose = verbose
        self.log_freq = log_freq

    def fit(self):
        j = 0
        obj_target = 1e-25

        self.t_state = self.state
        self.residual_update()
        self.objective_update()
        self.obj = self.t_obj

        while True:

            if(j % self.log_freq == 0 and self.verbose):
                print((f'iter {j}: obj= {self.obj}, λ= {self.distrust_lambda}'))
            if self.obj <= obj_target or j >= self.iterations:
                if self.verbose:
                    print(
                        f"Iter {j} converged to local min (obj: {self.obj}).")
                break

            self.state_update()  # calc new t_state
            self.residual_update()  # find new error
            self.objective_update()  # get objective

            if self.t_obj < self.obj:
                self.distrust_lambda = max(
                    self.distrust_min, self.distrust_lambda*self.distrust_down)
                # record new minimums
                self.obj = self.t_obj
                self.state = self.t_state
            else:
                # reset to minimums
                self.t_obj = self.obj
                self.t_state = self.state
                self.residual_update()

                if self.distrust_lambda == self.distrust_max:
                    if self.verbose:
                        print(
                            f"Iter {j} could not minimize further (obj: {self.obj}).")
                    break
                self.distrust_lambda = min(
                    self.distrust_max, self.distrust_lambda*self.distrust_up)
            j += 1

    def residual_update(self):
        pass

    def objective_update(self):
        pass

    def state_update(self):
        pass

