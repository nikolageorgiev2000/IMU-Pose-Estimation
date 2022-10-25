
from __future__ import annotations
from src import *
from src.data_collection.posers import *
from src.math3d.rotations import *


class VNode():
    def __init__(self, label: str, state: VState):
        self.label = label
        self.state = state

        self.sep: List[VNode] = []
        self.parent: VNode = None
        self.kernel: Set[FNode] = set()
        self.mail: Set[FNode] = set()
        self.message: FNode = FNode.create_preset(f"{self.label}_msg", [])
        self.condition: FNode = FNode.create_preset(f"{self.label}_cond", [])

    def get_value(self):
        return self.state.value

    def get_delta(self):
        return self.state.delta

    def set_delta(self, d):
        self.state.delta = d

    def factorize(self):
        # get factors of the clique
        ev_factors = sorted(self.kernel.union(
            self.mail), key=lambda f: f.label)

        # sort; prepend variable to be eliminated
        clique = self.condition.scope

        # build matrix to decompose
        rows = sum(len(f.get_residual()) for f in ev_factors)
        cols = sum(len(v.get_value()) for v in clique)
        coeff_mat = torch.zeros((rows, cols))
        c = 0
        for v in clique:
            cn = c + len(v.get_value())
            r = 0
            for f in ev_factors:
                rn = r + len(f.get_residual())
                if v in f.scope:
                    coeff_mat[r:rn, c:cn] = f.get_jacobian(v)
                r = rn
            c = cn
        residuals = torch.cat([f.get_residual() for f in ev_factors], dim=-1)
        product_mat = torch.cat((coeff_mat, residuals[:, None]), dim=-1)

        # qr decomposition
        Q, _ = torch.linalg.qr(
            product_mat[:, :len(self.get_value())], mode='complete')
        new_product_mat = Q.T @ product_mat
        eps = 1e-12
        new_product_mat[torch.abs(new_product_mat) < eps] = 0

        # add new factor; add parent coeffs of eliminated variable
        r = len(self.get_value())
        R = new_product_mat[:r, :r]
        d = new_product_mat[:r, -1]
        self.condition._jacobian = {}
        self.condition._jacobian[self] = R
        self.condition._residual = d

        if(r < len(new_product_mat)):
            c = r
            for v in clique[1:]:
                cn = c + len(v.get_value())
                self.message._jacobian[v] = new_product_mat[r:, c:cn]
                self.condition._jacobian[v] = new_product_mat[:r, c:cn]
                c = cn
            self.message._residual = new_product_mat[r:, -1]

    def maximize(self):
        parent_sum = torch.zeros_like(self.get_delta())
        for p in self.sep:
            parent_sum += self.condition.get_jacobian(p) @ p.get_delta()
        new_delta = torch.linalg.solve(self.condition.get_jacobian(
            self), (self.condition.get_residual() - parent_sum))
        self.set_delta(new_delta)

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return f"vnode_{self.label}"


class VState():
    def __init__(self, init_val: FT, default_delta: FT, update_func: Callable[[FT, FT]], prior_func):
        self.value: FT = init_val
        self.prev_value: FT = init_val
        self._default_delta: FT = default_delta
        self.delta: FT = default_delta
        self._update_func = update_func
        self._prior_func = prior_func

    def update_value(self):
        self.value = self._update_func(self.value, self.delta)

    def revert_value(self):
        self.value = self.prev_value

    def reset_delta(self):
        self.delta = self._default_delta

    def create_prior(self, v: VNode) -> FNode:
        return FNode(f'{v.label}_prior', [v], [v.get_value()], self._prior_func)


class FNode():
    def __init__(self, label: str, scope: List[VNode], params: List[FT], update_func: Callable[[
            List[FT], List[FT]], Tuple[FT, FT, List[FT]]]):
        self.label = label
        self.scope: List[VNode] = scope
        self.params: List[FT] = params

        self._objective: FT = None
        self._residual: FT = None
        self._jacobian: Dict[VNode: FT] = {}
        # update function is of the form (params, states) -> (objective, residual, list of jacobians)
        self._update_func: Callable[[
            List[FT], List[FT]], Tuple[FT, FT, List[FT]]] = update_func

    @classmethod
    def create_preset(cls, label: str, scope: List[VNode]):
        return cls(label, scope, None, None)

    def get_objective(self) -> FT:
        return self._objective

    def get_residual(self) -> FT:
        return self._residual

    def get_jacobian(self, v: VNode) -> Dict[VNode, FT]:
        return self._jacobian[v]

    def get_input_values(self) -> List[FT]:
        return [v.state.value for v in self.scope]

    def update(self):
        if self._update_func is None:
            raise Exception(f"No update function for FNode {self.label}.")
        vals = self.get_input_values()
        obj, res, jac = self._update_func(self.params, vals)
        self._objective = obj
        self._residual = res
        self._jacobian = {self.scope[i]: jac[i]
                          for i in range(len(self.scope))}

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return f"fnode_{self.label}"


class QRGDL():
    def __init__(self):
        self.variables: Set[VNode] = set()
        # does not include jtree factors (messages, conditions)
        self.factors: Set[FNode] = set()
        self.hot_facts: Set[FNode] = set()

    def add_data(self, new_factors: Iterable[FNode]):
        scope = self.get_scope(new_factors)
        old_vars = self.variables.intersection(scope)
        aff_vars = set()
        for ov in old_vars:
            c = ov
            while c is not None and c not in aff_vars:
                aff_vars.add(c)
                c = c.parent

        aff_factors = set()
        aff_factors.update(new_factors)
        for av in self.sort_variables(aff_vars, True):
            aff_factors.update(av.kernel)
            # remove messages received from affected vnodes
            if av.parent is not None:
                av.parent.mail.discard(av.message)

        self.hot_facts.update(aff_factors)

        self.factors.update(new_factors)
        self.variables.update(scope)

        self.construct_jtree()

    def sort_variables(self, vars: Iterable[VNode], reverse=False) -> List[VNode]:
        return sorted(vars, key=lambda x: x.label, reverse=reverse)

    def first_variable(self, vars: Iterable[VNode]) -> VNode:
        return min(vars, key=lambda x: x.label)

    def cut_variables(self, count: int):
        elim_order = self.sort_variables(self.variables)
        cut_vars = set(elim_order[:count])
        for v in cut_vars:
            for f in v.kernel:
                self.factors.remove(f)
            self.variables.remove(v)
            p = v.parent
            if p is not None and p not in cut_vars:
                # use parent value to create a prior
                p.mail.remove(v.message)
                prior_f = p.state.create_prior(p)
                p.kernel.add(prior_f)
                self.factors.add(prior_f)

    def construct_jtree(self):

        kernel_map: Dict[VNode, Set[FNode]] = {}
        for af in self.hot_facts:
            for v in af.scope:
                if v not in kernel_map:
                    # even if no kernel, could get a message!
                    kernel_map[v] = set()
            # add factors to kernels of first variable eliminated in scope
            kernel_map[self.first_variable(af.scope)].add(af)

        self.hot_facts.clear()

        # use active factor map to eliminate variables
        for v in self.sort_variables(kernel_map.keys()):
            # need to copy to avoid side-effects
            v.kernel = set(kernel_map.setdefault(v, set()))
            kernel_map.pop(v)

            clique = self.get_scope(v.kernel).union(self.get_scope(v.mail))
            clique.discard(v)
            v.sep = self.sort_variables(clique)
            v.message.scope = v.sep
            v.condition.scope = [v]+v.sep
            if len(v.sep) != 0:
                v.parent = v.sep[0]
                v.parent.mail.add(v.message)

        # after construction all factors should be deactivated (in jtree)
        assert len(self.hot_facts) == 0

    def mfqr(self):
        for f in self.factors:
            f.update()
        for v in self.sort_variables(self.variables):
            v.factorize()
        for v in self.sort_variables(self.variables, True):
            v.maximize()
            v.state.update_value()
            v.state.reset_delta()

    def get_scope(self, factors: Iterable[FNode]):
        scope = set()
        for f in factors:
            scope.update(f.scope)
        return scope

    def get_loss(self):
        return sum([f.get_objective() for f in self.factors])
