from src import *
from src.math3d.linalg import eig_sqrt

def fit_ellipsoid(data) -> Tuple[FT, FT]:
    '''Returns center and inverse transform of least squares ellipsoid.'''

    J = torch.cat(
        (data**2, *((2*data[:, i]*data[:, (i+1) % data.shape[1]])[:, None] for i in [0, 2, 1]), 2*data), dim=1)
    #  Ax^2 + By^2 + Cz^2 +  2Dxy +  2Exz +  2Fyz +  2Gx +  2Hy +  2Iz = 1
    JtJ = J.T @ J
    # perform least squares
    print(JtJ.dtype, J.dtype)
    coeffs = torch.linalg.solve(JtJ, J.T @ torch.ones(data.shape[0]))

    X = FT(
        [
            [coeffs[0], coeffs[3], coeffs[4]],
            [coeffs[3], coeffs[1], coeffs[5]],
            [coeffs[4], coeffs[5], coeffs[2]],
        ])
    p = coeffs[6:9]
    b = - torch.linalg.lstsq(X,p)[0]
    b = torch.squeeze(b)

    centered_data = data - b
    # J = np.concatenate(
    #     (centered_data**2, *((2*centered_data[:, i]*centered_data[:, (i+1) % centered_data.shape[1]])[:, np.newaxis] for i in [0, 2, 1])), axis=1)
    # #  Ax^2 + By^2 + Cz^2 +  2Dxy +  2Exz +  2Fyz = 1
    # JtJ = J.T @ J
    # # perform least squares
    # coeffs = np.linalg.solve(JtJ, J.T @ np.ones(data.shape[0]))
    # SinvT_Sinv = FT([
    #     [coeffs[0], coeffs[3], coeffs[4]],
    #     [coeffs[3], coeffs[1], coeffs[5]],
    #     [coeffs[4], coeffs[5], coeffs[2]],
    # ])
    # Sinv = eig_sqrt(SinvT_Sinv).float()

    J = centered_data**2
    #  Ax^2 + By^2 + Cz^2 +  2Dxy +  2Exz +  2Fyz = 1
    JtJ = J.T @ J
    # perform least squares
    coeffs = torch.linalg.solve(JtJ, J.T @ torch.ones(data.shape[0]))
    Sinv = coeffs**0.5 * torch.eye(3)
    
    return Sinv, b