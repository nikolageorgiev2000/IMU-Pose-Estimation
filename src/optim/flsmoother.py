from src.optim.gdl import *

class FLSmoother():
    def __init__(self, max_size, iter_count, verbose=False):
        self.max_size = max_size
        self.iter_count = iter_count

        self.gdl = QRGDL()
        self.variables = set()

        self.verbose = verbose

    def add_data(self, variables: Iterable[VNode], factors: Iterable[FNode]):
        self.variables.update(variables)
        self.gdl.add_data(factors)

    def iterate(self, fail_count):
        objs = [FT([float('inf')])]
        max_fails = fail_count
        counter = 0
        for i in range(self.iter_count):
            self.gdl.mfqr()
            L = self.gdl.get_loss()
            if L > objs[-1]:
                counter += 1
            else:
                counter = 0

            objs.append(L)

            if torch.abs(objs[-1]-objs[-2])/objs[-1] < 1e-5:
                break
            if objs[-1] < 1e-8:
                break
            if counter>max_fails:
                break


        self.gdl.cut_variables(max(0, len(self.gdl.variables) - self.max_size))

        if self.verbose:
            print(i, L, self.gdl.variables, torch.abs(objs[-1]-objs[-2]))
        return i, objs
