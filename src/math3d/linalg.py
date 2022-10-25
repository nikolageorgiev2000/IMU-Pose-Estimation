from src import *
from torch.nn.functional import pad
from rotations import *

def eig_sqrt(mat: FT) -> FT:
    if mat.ndim == 1:
        return mat ** 0.5
    lambdas, axes = torch.linalg.eig(mat)
    return axes.T @ torch.diag(lambdas) ** 0.5 @ axes


def covariance(x: FT) -> FT:
    x_mean_diff = x - torch.mean(x, dim=0)
    return ein('...i,...j->ij', x_mean_diff, x_mean_diff) / x.shape[0]


def moving_covariance_threshold(data: FT, radius: int, deviations: float = 3.0) -> FT:
    cov_vols = torch.empty(0)
    for idx in range(radius, len(data)-radius):
        data_window = data[idx-radius: idx+radius+1]
        volume = torch.det(covariance(data_window)) ** 0.5
        cov_vols = torch.cat((cov_vols, FT([volume])))
    min_det = torch.min(cov_vols)
    static_inds = cov_vols <= (deviations**data.shape[-1])*min_det
    padding = FT([False]).repeat(radius)
    return torch.cat((padding, static_inds, padding))


def normalize(vec: FT) -> FT:
    normed = vec.div(torch.linalg.norm(vec, dim=-1)[..., None])
    # normed[torch.isnan(normed) | torch.isinf(normed)] = 0
    return torch.nan_to_num(normed)


def moving_average(data: FT, radius: int) -> FT:
    signal = data
    padded_sig = pad(signal.T, (radius, radius),
                     mode='replicate').T  # pad time-series rows

    window = 2*radius + 1
    avg = sum([padded_sig[i:len(padded_sig)-window+i+1]
              for i in range(window)]) / window

    return avg

def linear_interpolate(t, x, y):
    return x + (y-x)*t


def SO3_interpolate(t, x, y):
    return exp_rotmat(t * log_rotmat(y @ x.T)) @ x


def interpolate_at(query, gtime, gdata, interpolater: Callable):
    # assume times are sorted
    g_ind = 0
    idata = torch.empty((0, *gdata.shape[1:]))
    for q in query:
        while True:
            if g_ind%100==0:
                print(g_ind)
            if g_ind >= len(gdata)-1:
                raise Exception(f"query {q} continues past ground truth end {gtime[-1]}")
            tprev = gtime[g_ind]
            tnext = gtime[g_ind+1]
            # print(q, tnext, tprev)
            if q < tnext:
                if q >= tprev:
                    t = (q - tprev)/(tnext-tprev)
                    # print(t)
                    d = interpolater(t, gdata[g_ind], gdata[g_ind+1])
                    idata = torch.cat((idata, d[None,:]))
                    break
                else:
                    continue
                    # raise Exception("query starts before ground truth!")
            else:
                g_ind += 1

    return idata