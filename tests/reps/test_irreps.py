import torch
from e3nn import o3

from tensorframes.lframes.classical_lframes import RandomLFrames
from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.irreps import Irreps, IrrepsTransform
from tensorframes.reps.tensorreps import TensorReps


def test_irreps():
    basis_change = RandomLFrames()(pos=torch.zeros(100, 3))

    # test that 0n transforms correctly:
    irrep = Irreps("5x0")
    coeffs = torch.randn(100, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(transformed_coeffs, coeffs)

    # test that 0p transforms correctly:
    irrep = Irreps("5x0p")
    coeffs = torch.randn(100, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(transformed_coeffs, coeffs * basis_change.det[:, None], atol=1e-7)

    # test that 1n transforms correctly:
    irrep = Irreps("5x1")
    coeffs = torch.randn(100, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(
        transformed_coeffs.reshape(100, 5, 3),
        torch.matmul(coeffs.reshape(100, 5, 3), basis_change.matrices.transpose(-1, -2)),
        atol=1e-7,
    )

    # test that on vectors tensorreps and irreps agree:
    tensor_reps = TensorReps("5x1")
    tensor_reps_transform = tensor_reps.get_transform_class()
    tensor_transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(tensor_transformed_coeffs, transformed_coeffs, atol=1e-7)

    # test that 1p transforms correctly:
    irrep = Irreps("5x1p")
    coeffs = torch.randn(100, irrep.dim)
    irreps_transform = IrrepsTransform(irrep)
    transformed_coeffs = irreps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(
        transformed_coeffs.reshape(100, 5, 3),
        torch.matmul(coeffs.reshape(100, 5, 3), basis_change.matrices.transpose(-1, -2))
        * basis_change.det[:, None, None],
        atol=1e-7,
    )

    # test that on vectors tensorreps and irreps agree:
    tensor_reps = TensorReps("5x1p")
    tensor_reps_transform = tensor_reps.get_transform_class()
    tensor_transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(tensor_transformed_coeffs, transformed_coeffs, atol=1e-7)

    # test consistency of higher order irreps:
    in_reps = Irreps("16x0n + 8x1n + 2x1p + 4x2n + 2x3p")
    x = torch.randn(100, in_reps.dim)

    feature_transform = in_reps.get_transform_class()

    # Perform the forward pass
    x_local = feature_transform(coeffs=x.clone(), basis_change=basis_change)

    # add small invariance test:
    global_rot = o3.rand_matrix(1).repeat(100, 1, 1)
    x_trafo = feature_transform(coeffs=x, basis_change=LFrames(global_rot))
    lframes_trafo = []
    for i in range(100):
        lframes_trafo.append(basis_change.matrices[i] @ global_rot[0].T)
    lframes_trafo = LFrames(torch.stack(lframes_trafo))
    x_trafo_local = feature_transform(coeffs=x_trafo.clone(), basis_change=lframes_trafo)
    diff = x_local - x_trafo_local
    print(diff.abs().max(), "max diff", diff.abs().mean(), "mean diff")
    assert torch.allclose(x_local, x_trafo_local, atol=1e-4)
