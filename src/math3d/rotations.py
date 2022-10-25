from torch import norm, normal
from src import *
from src.math3d.linalg import normalize

levi_civita = torch.zeros((3, 3, 3))
levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1
levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1
levi_civita = FT(levi_civita)


def skew_symm(vec):
    check_vec(vec)
    return torch.einsum('ijk, ...k -> ...ij', -levi_civita, vec)


def inv_skew(skew_mat):
    return torch.einsum('ijk, ...jk -> ...i', -levi_civita, skew_mat) / 2


def exp_rotmat(rotvec):
    check_vec(rotvec)
    a = torch.linalg.norm(rotvec, dim=-1)[..., None, None]
    n = normalize(rotvec)
    skew_n = skew_symm(n)
    mat = torch.eye(3) + torch.sin(a)*skew_n + (1 - torch.cos(a))*torch.einsum('...ij,...jk', skew_n, skew_n)
    return mat


def log_rotmat(rotmat):
    tr = torch.einsum('...ii', rotmat)[..., None, None]
    a = torch.arccos((tr-1)/2)
    a_coeff = torch.nan_to_num((a/torch.sin(a)/2))
    mat_log = a_coeff*(rotmat-torch.einsum('...ij->...ji', rotmat))
    return inv_skew(mat_log)


def log_rotmat2(R):
    # Use rotmat instead. Rotation angle limited (-pi/2, pi/2)
    A = (R - torch.einsum('...ij->...ji', R))/2
    tr = ein('ii',A@A)
    a = (-tr/2) ** 0.5
    print(A, tr, a)
    vec = torch.nan_to_num(torch.asin(a)/a) * A
    return inv_skew(vec)

def log_rotmat3(rotmat):
    tr = torch.einsum('...ii', rotmat)[..., None, None]
    a = torch.arccos((tr-1)/2)
    sin_a = (1 - tr**2)**0.5
    a_coeff = torch.nan_to_num((a/sin_a/2))
    mat_log = a_coeff*(rotmat-torch.einsum('...ij->...ji', rotmat))
    return inv_skew(mat_log)

def exp_quat(vec):
    check_vec(vec)
    v_norm = torch.linalg.norm(vec, dim=-1)[..., None]
    # handle divide by 0 with nan_to_num
    return torch.cat((torch.nan_to_num(vec/v_norm) * torch.sin(v_norm), torch.cos(v_norm)), dim=-1)


def log_quat(q):
    '''Takes unit quaternion. Returns rotation vector.'''
    check_quat(q)
    check_unit(q)
    v, w = decomp_quat(q)
    v_norm = torch.linalg.norm(v, dim=-1)[..., None]
    # handle divide by 0 with nan_to_num
    return torch.nan_to_num(torch.arccos(w) / v_norm) * v


def mul_quat(q0, q1):
    # add new axes in case they are vectors
    premul_mat = premul_quat(q0)
    # broadcasting (...,4,4) @ (...,4,1) = (...,4,1) to (...,4)
    return (premul_mat @ q1[..., None])[..., 0]


def premul_quat(p):
    return matrix_isomorphism_quat(p, pre=True)


def postmul_quat(p):
    return matrix_isomorphism_quat(p, pre=False)


def matrix_isomorphism_quat(p, pre=True):
    '''Create 4x4 matrix isomorphism from quaternions `p`. Set `pre` true for pre-multiplication, false for post.'''
    check_quat(p)
    iso = torch.empty((*p.shape, 4))
    v, w = decomp_quat(p)

    iso[..., 0, 1:] = -v
    iso[..., 1:, 0] = v
    iso[..., 1:, 1:] = skew_symm(v) if pre else -skew_symm(v)
    # torch.einsum('...ii->...i', iso)[:] = w.repeat((4,1))
    iso[..., range(4), range(4)] = w

    # move 1st column to end, matching [vector scalar] input order
    iso[..., [0, 1, 2, 3]] = iso[..., [1, 2, 3, 0]]
    # move 1st row to end, matching [vector scalar] output order
    iso[..., [0, 1, 2, 3], :] = iso[..., [1, 2, 3, 0], :]

    return iso


def decomp_quat(q):
    check_quat(q)
    return q[..., :3], q[..., 3, None]


def conj_quat(q):
    check_quat(q)
    qc = torch.clone(q)
    qc[..., :3] *= -1
    return qc


def check_quat(q):
    if q.shape[-1] != 4:
        raise Exception(f"Wrong shape {q.shape}, Quat should be (...,4)")


def check_unit(q):
    q_norm = torch.linalg.norm(q, dim=-1)[..., None]
    if not torch.allclose(q_norm, torch.ones_like(q_norm)):
        raise Exception(f"Unit quaternion required! Input was: {q}")


def check_vec(vec):
    if vec.shape[-1] != 3:
        raise Exception(f"Wrong shape {vec.shape}, vector should be (...,3)")


def rotate_from_quat(vec, q):
    pure_vec = torch.cat((vec, torch.zeros((*vec.shape[:-1], 1))), dim=-1)
    # print(vec.shape, pure_vec.shape, q.shape)

    rotated_vec1 = postmul_quat(conj_quat(q)) @ pure_vec
    rotated_vec2 = premul_quat(q) @ rotated_vec1
    # print(pure_vec, rotated_vec)
    return decomp_quat(rotated_vec2)[0]


def get_rotvec(v0, v1):
    theta = angle_to(v0, v1)
    return theta * normalize(torch.cross(v0, v1))

def angle_to(v0, v1):
    '''Takes two vectors. Returns angle between them using the dot-product.'''
    cos_theta = torch.sum(v0 * v1, axis=-1) / torch.linalg.norm(v0,
                                                          axis=-1) / torch.linalg.norm(v1, axis=-1)
    cos_theta = torch.nan_to_num(cos_theta)[..., None]
    # limit to [-1,1] range (otherwise nan) ; not sure if necessary
    cos_theta = torch.clip(cos_theta, -1, 1)
    return torch.arccos(cos_theta)
