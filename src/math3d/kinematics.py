from src import *

from . import rotations


def get_linear_acceleration(spf, rots_to_frame, grav):
    # s + R_bn @ g_n
    return spf + torch.einsum("tij, j->ti", rots_to_frame, grav)


def get_specific_force(a, rots_to_frame, grav):
    # a - R_bn @ g_n
    return a - torch.einsum("tij, j->ti", rots_to_frame, grav)


def integrate_linear_acceleration(accel, intervals, init_vel, init_pos):
    v_deltas = intervals * accel
    v_offsets = torch.cumsum(v_deltas, dim=0)
    v = torch.cat((init_vel[None, :], init_vel + v_offsets))

    p_deltas = intervals * v[:-1] + intervals**2 * accel / 2
    p_offsets = torch.cumsum(p_deltas, dim=0)
    p = torch.cat((init_pos[None, :], init_pos + p_offsets))

    return v, p


def diff_linear_velocity(s, intervals):
    deltas = s[1:] - s[:-1]
    return torch.reciprocal(intervals) * deltas


def get_angular_velocity(rots, intervals):
    # calculate R_bn @ R*_nb

    rot_deltas = torch.einsum("tij, tkj->tik", rots[:-1], rots[1:])
    rotvecs = rotations.log_rotmat(rot_deltas)
    return torch.reciprocal(intervals) * rotvecs


def integrate_angular_velocity(w, intervals, init_rot):
    # calculate rot(w_b).T @ R_bn

    rot_deltas = rotations.exp_rotmat(intervals * w)
    rots = init_rot[None, :, :]
    for idx in range(rot_deltas.shape[0]):
        rots = torch.cat((rots, (rot_deltas[idx].T @ rots[-1])[None, :, :]))
    return rots


def comp_filter(init_rot, rotvecs, magnos, nav, alpha):
    samples = len(rotvecs)
    g_deltas = rotations.exp_rotmat(rotvecs)
    rots = init_rot[None, ...]
    for i in range(samples):
        g_rot = g_deltas[i].T @ rots[-1]
        new_m = g_rot @ nav.north
        axis = rotations.normalize(torch.cross(new_m, magnos[i+1]))
        angle = torch.acos(torch.clamp(torch.dot(new_m, magnos[i+1]), -1, 1))
        new_rot = rotations.exp_rotmat(alpha * axis * angle) @ g_rot
        rots = torch.cat((rots, new_rot[None, ...]))
        if torch.any(torch.isnan(rots[-1])):
            raise Exception('Nan rot! Check your inputs.')
    return rots
