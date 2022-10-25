from __future__ import annotations
from src import *

class FNode(object):
    def __init__(self, name: str, index: int) -> None:
        self.name: str = name
        # assume order has been decided (SHOULD NOT CHANGE AFTER INIT)
        self.index = index
        self.var_edges: Dict[VNode, FT] = {}

        self.potential: Potential = None
        self.residual: FT = None

    def __hash__(self) -> int:
        return self.index


class VNode(object):
    def __init__(self, name: str, index: int) -> None:
        self.name: str = name
        # assume order has been decided (SHOULD NOT CHANGE AFTER INIT)
        self.index = index
        self.neighbors: Set[FNode] = set()

        self.state_index: int = None
        self.state: State = None
        self.parent_coeffs: Dict[VNode, FT] = {}
        self.R = None
        self.d = None

    def set_value(self, state: State, ind: int):
        self.state = state
        self.state_index = ind

    def get_value(self):
        return self.state.get_value()[self.state_index]

    def get_delta(self):
        return self.state.get_delta()[self.state_index]


class FactorGraph(object):
    def __init__(self) -> None:
        self.factors: List[FNode] = []
        self.variables: List[VNode] = []

    def create_variable(self, name: str) -> VNode:
        v = VNode(name, len(self.variables))
        self.variables.append(v)
        return v

    def create_factor(self, name: str) -> FNode:
        f = FNode(name, len(self.factors))
        self.factors.append(f)
        return f

    def get_variable(self, ind: int):
        return self.variables[ind]

    def get_factor(self, ind: int):
        return self.factors[ind]

    def add_edge(self, v: VNode, f: FNode, edge_value: FT):
        v.neighbors.add(f)
        f.var_edges[v] = edge_value

    def delete_edge(self, v: VNode, f: FNode):
        v.neighbors.remove(f)
        f.var_edges.pop(v)


class Potential(object):
    def __init__(self, name: str, potential_func: Callable[..., FT], J: List[Callable[..., FT]] = None) -> None:
        self.name = name
        self.residual_func = potential_func
        self.sig = inspect.signature(self.residual_func)
        self.param_order = {p: i for i, p in enumerate(
            self.sig.parameters.keys())}
        self.manual_jacobian_funcs = J
        self.residual = None
        self.jacobians = None

    def bind_vars(self, vgroups: Iterable[VGroup]):
        val_kwargs = {vg.name: vg.value
                      for vg in vgroups if vg.name in self.param_order.keys()}
        return self.sig.bind(**val_kwargs).args

    def evaluate_residual(self, vgroups: Iterable[VGroup]):
        val_args = self.bind_vars(vgroups)
        # TODO: catch exception here and raise Incompatible VNode if necessary!!!
        self.residual = self.residual_func(*val_args)

    def parameter_sort(self, vgroups: Iterable[VGroup]) -> List[VGroup]:
        vgroups = [vg for vg in vgroups if vg.name in self.param_order.keys()]
        ordered_vars = sorted(vgroups, key=lambda x: self.param_order[x.name])
        # TODO: check for None and raise Incompatible VNode exception if necessary!!!
        return ordered_vars

    def evaluate_jacobians(self, vgroups: Iterable[VGroup]):
        val_args = self.bind_vars(vgroups)
        self.jacobians = [j(*val_args) for j in self.manual_jacobian_funcs]


class State(object):
    def __init__(self, name: str, value: FT, init_deviation: FT, update_func: Callable[[FT, FT, List[Any]], FT]) -> None:
        self.name = name
        self.value: FT = value
        self.value.requires_grad = False
        self.init_deviation: FT = init_deviation
        self.delta: FT = self.init_deviation
        # update function has signature (value, deviation)
        self.update: Callable[[FT, FT, List[Any]], FT] = update_func
        self.prev_value: FT = self.value
        self.prev_delta: FT = self.delta

    def get_value(self):
        return self.value

    def get_delta(self):
        return self.delta

    def revert_state(self):
        self.value = self.prev_value
        self.delta = self.prev_delta

    def set_delta(self, d):
        self.prev_delta = self.delta
        self.delta = d

    def get_state_update(self, hyperparams: List[Any]):
        return self.update(self.value, self.delta, hyperparams)

    def apply_state_update(self, hyperparams: List[Any]):
        self.prev_value = self.value
        self.prev_delta = self.delta
        self.value = self.get_state_update(hyperparams)
        self.delta = self.init_deviation


class VGroup(object):
    def __init__(self, name: str, state: State, filter_func: Callable[[List[VNode], FT]]) -> None:
        self.name = name
        self.state: State = state
        self.filter_func = filter_func
        self.value = None
        self.vnodes = []

    def set_data(self, vnodes: List[VNode]):
        self.vnodes, self.value = self.filter_func(
            vnodes, self.state.get_value())
        if len(self.vnodes) != len(self.value):
            print(f"{self.name}: Variable count and Value count DOES NOT MATCH!")


class MFQR(object):
    def __init__(self):
        pass

    def initialize(self, potentials: List[Potential], states: List[State]) -> None:
        self.potentials: Dict[Potential, List[FNode]] = {
            p: [] for p in potentials}
        self.state_vars: Dict[State, List[VNode]] = {s: [] for s in states}
        self.vgroups: List[VGroup] = []
        self.fgraph: FactorGraph = FactorGraph()

        # create VNodes from states, 1:1 mapping
        for s in states:
            for i in range(len(s.get_value())):
                var_name = f"{s.name}_{i}"
                new_vnode = self.fgraph.create_variable(var_name)
                new_vnode.set_value(s, i)
                self.state_vars[s].append(new_vnode)

    def revert_states(self):
        for state in self.state_vars.keys():
            state.revert_state()

    def update_states(self, state_hyperparams: Dict[State, List[Any]]):
        for state in self.state_vars.keys():
            state.apply_state_update(state_hyperparams[state])

    def set_vgroup_data(self):
        for vg in self.vgroups:
            vg.set_data(self.state_vars[vg.state])

    def add_vgroup(self, vg: VGroup):
        self.vgroups.append(vg)

    def clear_factors(self):
        for f in list(self.fgraph.factors):
            self.disconnect_factor(f)
        self.potentials: Dict[Potential, List[FNode]] = {
            p: [] for p in self.potentials.keys()}
        self.fgraph.factors = []

    def evaluate_residuals(self):
        # evaluate potential residuals
        for pot in self.potentials.keys():
            pot.evaluate_residual(self.vgroups)

    def get_residual_norms(self):
        return {pot.name: float(torch.linalg.norm(pot.residual).item()) for pot in self.potentials}

    def evaluate_jacobians(self):
        # evaluate potential jacobians
        # calculate Jacobian and create VNode-FNode edges
        for pot in self.potentials.keys():
            pot.evaluate_jacobians(self.vgroups)

    def build_factors(self):

        self.clear_factors()
        self.set_vgroup_data()
        self.evaluate_residuals()
        self.evaluate_jacobians()

        # create FNodes from potential residuals, 1:1 mapping
        for pot in self.potentials.keys():
            for i in range(len(pot.residual)):
                factor_name = f"{pot.name}_{i}"
                f = self.fgraph.create_factor(factor_name)
                self.potentials[pot].append(f)
                f.residual = pot.residual[i]
                f.var_edges = {}

        # calculate Jacobian and create VNode-FNode edges
        for pot in self.potentials.keys():
            param_vgs = pot.parameter_sort(self.vgroups)
            for i in range(len(param_vgs)):
                vnodes = param_vgs[i].vnodes
                for j in range(len(vnodes)):
                    v = vnodes[j]
                    f = self.potentials[pot][j]
                    jacobian = pot.jacobians[i][j]
                    self.fgraph.add_edge(v, f, jacobian)

    def eliminate_variable(self, ev: VNode):

        if len(ev.neighbors) == 0:
            print(f"VNode {ev} has no factor neighbors!")

        # get factors of the clique
        ev_factors = sorted(ev.neighbors, key=lambda f: f.index)
        # build separator
        sepset = set.union(*(set(f.var_edges.keys()) for f in ev_factors))
        sepset.remove(ev)

        # sort; prepend variable to be eliminated
        clique = sorted(sepset, key=lambda v: v.index)
        clique.insert(0, ev)

        # build matrix to decompose
        rows = sum(len(f.residual) for f in ev_factors)
        cols = sum(len(v.get_value()) for v in clique)
        coeff_mat = torch.zeros((rows, cols))
        c = 0
        for v in clique:
            cn = c + len(v.get_value())
            r = 0
            for f in ev_factors:
                rn = r + len(f.residual)
                if v in f.var_edges:
                    coeff_mat[r:rn, c:cn] = f.var_edges[v]
                r = rn
            c = cn
        residuals = torch.cat([f.residual for f in ev_factors], dim=-1)
        product_mat = torch.cat((coeff_mat, residuals[:, None]), dim=-1)

        # qr decomposition
        Q, _ = torch.linalg.qr(
            product_mat[:, :len(ev.get_value())], mode='complete')
        new_product_mat = Q.T @ product_mat
        eps = 1e-12
        new_product_mat[torch.abs(new_product_mat) < eps] = 0
        # display(product_mat, new_product_mat)

        # add new factor; add parent coeffs of eliminated variable
        r = len(ev.get_value())
        ev.R = new_product_mat[:r, :r]
        ev.d = new_product_mat[:r, -1]
        if(r < len(new_product_mat)):
            new_edges = {}
            c = r
            new_f = self.fgraph.create_factor("new_f")
            for v in clique[1:]:
                cn = c + len(v.get_value())
                new_edges[v] = new_product_mat[r:, c:cn]
                ev.parent_coeffs[v] = new_product_mat[:r, c:cn]
                c = cn
            # display(ev.R, ev.d, ev.parent_coeffs)
            new_residuals = new_product_mat[r:, -1]
            self.connect_precomputed_factor(new_f, new_edges, new_residuals)

        # disconnect old factors
        for f in ev_factors:
            self.disconnect_factor(f)
        # display([f.var_edges for f in self.factors])

    def connect_precomputed_factor(self, f: FNode, edge_weights: Dict[VNode, FT], residual: FT):
        # use pre-computed weights and residual
        f.residual = residual
        for v in edge_weights.keys():
            self.fgraph.add_edge(v, f, edge_weights[v])

    def backsubstitute(self):
        new_deltas: Dict[State, FT] = {
            s: s.init_deviation.clone() for s in self.state_vars.keys()}
        for v in self.fgraph.variables[::-1]:
            parent_sum = torch.zeros_like(v.get_delta())
            for p, T in v.parent_coeffs.items():
                parent_sum += T @ new_deltas[p.state][p.state_index]
            d = torch.linalg.inv(v.R) @ (v.d - parent_sum)
            # display(v.d, v.d.shape, parent_sum, parent_sum.shape)
            new_deltas[v.state][v.state_index] = d
        for s, d in new_deltas.items():
            s.set_delta(d)

    def disconnect_factor(self, f: FNode) -> None:
        '''Disconnect factor node from the graph by removing it from its neighbors.'''
        for v in set(f.var_edges.keys()):
            self.fgraph.delete_edge(v, f)
        f.var_edges = {}
