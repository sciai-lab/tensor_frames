import os
from typing import Union

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
def _z_rot_mat(angle: torch.Tensor, l: int) -> torch.Tensor:
    """Compute the wigner d matrix for a rotation around the z axis with a given angle for an
    angular momentum.

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


def euler_angles_yxy(
    matrix: torch.Tensor, handle_special_cases: bool = False, eps: float = 1e-9
) -> torch.Tensor:
    """Calculate the Euler angles using the yxy convention to match the wigner d from e3nn.

    Args:
        matrix (torch.Tensor): The input rotation matrix of shape (..., 3, 3).

    Returns:
        torch.Tensor: The Euler angles alpha, beta, and gamma, each of shape (...).

    References:
        - https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    angles = torch.zeros(matrix.shape[:-1], dtype=matrix.dtype, device=matrix.device)
    angles[..., 0] = torch.arctan2(matrix[..., 0, 1], matrix[..., 2, 1])
    angles[..., 1] = torch.arccos(matrix[..., 1, 1])
    angles[..., 2] = torch.arctan2(matrix[..., 1, 0], -matrix[..., 1, 2])

    if handle_special_cases:
        # hard code diagonal special cases (these do not happen much in practice):
        mask_0 = (matrix - torch.eye(3)).abs().max(-1).values.max(-1).values < eps
        mask_1 = (
            matrix - torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        ).abs().max(-1).values.max(-1).values < eps
        mask_2 = (
            matrix - torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        ).abs().max(-1).values.max(-1).values < eps
        mask_3 = (
            matrix - torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        ).abs().max(-1).values.max(-1).values < eps
        angles[mask_0] = 0.0
        angles[mask_1] = torch.tensor([0.0, torch.pi, 0.0])
        angles[mask_2] = torch.tensor([0.0, 0.0, torch.pi])
        angles[mask_3] = torch.tensor([0.0, torch.pi, torch.pi])

    return angles[..., 0], angles[..., 1], angles[..., 2]


def wigner_D_with_J(
    l: int, J: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Wigner D matrix using the precomputed J matrix and Euler angles.
    Taken from https://github.com/atomicarchitects/equiformer_v2/blob/main/nets/equiformer_v2/wigner.py

    The paper which presents this approach:
    https://iopscience.iop.org/article/10.1088/1751-8113/40/7/011/pdf

    The original implementation by Taco Cohen:
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


def wigner_D_from_matrix(
    l: int,
    matrix: torch.Tensor,
    J: Union[None, torch.Tensor] = None,
    handle_special_cases: bool = False,
) -> torch.Tensor:
    """Calculate the Wigner D-matrix for a given angular momentum `l` and rotation matrix `matrix`.

    Args:
        l (int): The angular momentum quantum number.
        matrix (torch.Tensor): The rotation matrix. shape (..., 3, 3)
        J (torch.Tensor, optional): The J matrix. If not provided, it will be looked up based on the angular momentum `l`.

    Returns:
        torch.Tensor: The resulting Wigner D matrix of shape (..., 2l+1, 2l+1)
    """
    if J is None:
        J = _Jd[l].to(matrix.dtype).to(matrix.device)
    alpha, beta, gamma = euler_angles_yxy(matrix, handle_special_cases=handle_special_cases)
    return wigner_D_with_J(l, J, alpha, beta, gamma)


if __name__ == "__main__":
    from e3nn.o3 import matrix_to_angles, rand_matrix

    # test angle implementation:
    R = rand_matrix(1000)

    # check that is works even for noise:
    R += 1e-7 * torch.randn_like(R)

    angles = matrix_to_angles(R)
    my_angles = euler_angles_yxy(R)

    for angle, my_angle in zip(angles, my_angles):
        assert torch.allclose(angle, my_angle, atol=1e-5)

    # test angle special cases:
    special_Rs = [torch.eye(3)]
    for i in range(3):
        R = -torch.eye(3)
        R[i, i] = 1
        special_Rs.append(R)
    special_Rs = torch.stack(special_Rs)

    angles = matrix_to_angles(special_Rs)
    my_angles = euler_angles_yxy(special_Rs, handle_special_cases=True)

    for angle, my_angle in zip(angles, my_angles):
        assert torch.allclose(angle, my_angle)

    # test that wigner for the identity matrix is the identity matrix
    for l in range(4):
        assert torch.allclose(
            wigner_D_from_matrix(l, torch.eye(3), handle_special_cases=True),
            torch.eye(2 * l + 1),
            atol=1e-6,
        )

    # test that wigner is 1 for l = 0:
    assert torch.allclose(wigner_D_from_matrix(0, torch.rand(10, 3, 3)), torch.ones(10, 1, 1))

    # test that wigner for l=1 is the same as the rotation matrix:

    matrix = rand_matrix(10)
    assert torch.allclose(wigner_D_from_matrix(1, matrix), matrix)

    # test that wigner is a representation:
    matrix1 = rand_matrix(10)
    matrix2 = rand_matrix(10)
    assert torch.allclose(
        wigner_D_from_matrix(1, torch.bmm(matrix1, matrix2)),
        torch.bmm(wigner_D_from_matrix(1, matrix1), wigner_D_from_matrix(1, matrix2)),
        atol=1e-6,
    )

    # test that wigner is an orthogonal matrix:
    wigner = wigner_D_from_matrix(3, rand_matrix(10))
    assert torch.allclose(wigner @ wigner.transpose(-1, -2), torch.eye(2 * 3 + 1), atol=1e-6)

    # test that wigner is an orthogonal representation:
    matrix = rand_matrix(10)
    assert torch.allclose(
        wigner_D_from_matrix(4, matrix).transpose(-1, -2),
        wigner_D_from_matrix(4, matrix.transpose(-1, -2)),
        atol=1e-6,
    )

    print("All tests passed!")

    # compare our implementation with e3nn:
    import time

    R = rand_matrix(10000)
    start = time.time()
    matrix_to_angles(R)
    diff = time.time() - start
    print(f"e3nn implementation took {diff:.3f}s")

    start = time.time()
    euler_angles_yxy(R)
    diff = time.time() - start
    print(f"Our implementation took {diff:.3f}s")
