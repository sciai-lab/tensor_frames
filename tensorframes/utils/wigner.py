import os

import torch

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def _z_rot_mat(angle, l):
    """
    Compute the wigner d matrix for a rotation around the z axis with a given angle for an angular momentum.

    Args:
        angle (torch.Tensor): The rotation angle.
        l (int): The angular momentum.

    Returns:
        torch.Tensor: The rotation matrix.

    """
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


def euler_angles_yxy(matrix):
    """
    Calculate the Euler angles using the yxy convention to match the wigner d from e3nn.

    Args:
        matrix (torch.Tensor): The input rotation matrix of shape (..., 3, 3).

    Returns:
        torch.Tensor: The Euler angles alpha, beta, and gamma, each of shape (...).

    References:
        - https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    alpha = torch.arctan2(matrix[..., 0, 1], matrix[..., 2, 1])
    beta = torch.arccos(matrix[..., 1, 1])
    gamma = torch.arctan2(matrix[..., 1, 0], -matrix[..., 1, 2])
    return alpha, beta, gamma


def wigner_D_with_J(l, J, alpha, beta, gamma):
    """
    Calculate the Wigner D matrix using the precomputed J matrix and Euler angles.
    Taken from https://github.com/atomicarchitects/equiformer_v2/blob/main/nets/equiformer_v2/wigner.py
 
    The paper which presents this approach: 
    https://iopscience.iop.org/article/10.1088/1751-8113/40/7/011/pdf
    
    The originial implementation by Taco Cohen:
    https://github.com/AMLab-Amsterdam/lie_learn/tree/master/lie_learn/representations/SO3/pinchon_hoggan

    Args:
        l (int): The order of the Wigner D matrix.
        J (torch.Tensor): The J matrix.
        alpha (torch.Tensor): The rotation angle around the y-axis. shape: (...,)
        beta (torch.Tensor): The rotation angle around the x-axis. shape like alpha.
        gamma (torch.Tensor): The rotation angle around the y-axis. shape like alpha.

    Returns:
        torch.Tensor: The resulting Wigner D matrix of shape (..., 2l+1, 2l+1)

    .. note::
        The Euler angles are in the yxy convention. But in the paper and other theoretical works one uses the zyz convention. E3nn is special in that regard.
    """
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def wigner_D_from_matrix(l, matrix, J=None):
    """
    Calculate the Wigner D-matrix for a given angular momentum `l` and rotation matrix `matrix`.

    Args:
        l (int): The angular momentum quantum number.
        matrix (torch.Tensor): The rotation matrix. shape (..., 3, 3)
        J (torch.Tensor, optional): The J matrix. If not provided, it will be looked up based on the angular momentum `l`.

    Returns:
        torch.Tensor: The resulting Wigner D matrix of shape (..., 2l+1, 2l+1)

    """
    if J is None:
        J = _Jd[l].to(matrix.dtype).to(matrix.device)
    alpha, beta, gamma = euler_angles_yxy(matrix)
    return wigner_D_with_J(l, J, alpha, beta, gamma)


if __name__ == "__main__":
    # test that wigner for the identity matrix is the identity matrix
    for l in range(4):
        print("l:", l)
        print(wigner_D_from_matrix(l, torch.eye(3)) - torch.eye(2 * l + 1))
        assert torch.allclose(wigner_D_from_matrix(l, torch.eye(3)), torch.eye(2 * l + 1), atol=1e-5)

    # test that wigner is 1 for l = 0:
    assert torch.allclose(wigner_D_from_matrix(0, torch.rand(10, 3, 3)), torch.ones(10, 1, 1))

    # test that wigner for l=1 is the same as the rotation matrix:
    from e3nn.o3 import rand_matrix
    matrix = rand_matrix(10)    
    assert torch.allclose(wigner_D_from_matrix(1, matrix), matrix)

    # test that wigner is a representation:
    matrix1 = rand_matrix(10)
    matrix2 = rand_matrix(10)
    assert torch.allclose(wigner_D_from_matrix(1, torch.bmm(matrix1, matrix2)), torch.bmm(wigner_D_from_matrix(1, matrix1), wigner_D_from_matrix(1, matrix2)))

    # test that wigner is an orthogonal matrix:
    wigner = wigner_D_from_matrix(3, rand_matrix(10))
    assert torch.allclose(wigner @ wigner.transpose(-1, -2), torch.eye(2 * 3 + 1))

    # test that wigner is an orthogonal representation:
    matrix = rand_matrix(10)
    assert torch.allclose(wigner_D_from_matrix(4, matrix).transpose(-1, -2), wigner_D_from_matrix(4, matrix.transpose(-1, -2)))