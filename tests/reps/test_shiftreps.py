import torch
from e3nn import o3

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.shiftreps import ShiftReps, ShiftRepsTransform


def test_shiftreps():
    # rep_1 = "5x0p+3x1+3x2+5x3"
    # rep_2 = "5x0"

    # tensor_reps_1 = ShiftReps(rep_1)
    # tensor_reps_2 = ShiftReps(rep_2)

    # coeffs = torch.randn(10, tensor_reps_1.dim)
    # shift_reps_transform = ShiftRepsTransform(tensor_reps_1)
    num_nodes = 10
    random_rot = o3.rand_matrix(num_nodes)
    shift = torch.randn(num_nodes, 3)
    lframes = LFrames(random_rot, shift=shift)

    # random_rot = o3.rand_matrix(20)
    # flip_mask = torch.randint(0, 2, (20,), dtype=torch.bool)
    # random_rot[flip_mask] *= -1
    # basis_change = ChangeOfLFrames(LFrames(random_rot[:10]), LFrames(random_rot[10:]))

    # test that 0n transforms correctly:
    irrep = ShiftReps("5x0")
    assert irrep.dim == 5
    coeffs = torch.randn(num_nodes, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep)
    transformed_coeffs = shift_reps_transform(coeffs.clone(), lframes)
    assert torch.allclose(transformed_coeffs, coeffs)

    # test that 1n transforms correctly:
    mult = 5
    irrep = ShiftReps(f"{mult}x1")
    assert irrep.dim == mult * 4
    coeffs = torch.randn(num_nodes, irrep.dim)
    print("coeffs.shape", coeffs.shape, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep)
    transformed_coeffs = shift_reps_transform(coeffs.clone(), lframes)

    print("coeffs.shape", coeffs.shape, transformed_coeffs.shape)
    print(coeffs - transformed_coeffs)

    # what I would expect the transformation to be:
    manually_transformed_coeffs = coeffs.reshape(num_nodes, mult, 4)
    print("random_rot.shape", random_rot.shape, manually_transformed_coeffs.shape)
    manually_transformed_coeffs[..., :-1] = (
        torch.einsum("kij, kmj -> kmi", random_rot, manually_transformed_coeffs[..., :-1])
        - torch.einsum("kij, kj -> ki", random_rot, shift)[:, None, :]
    )
    print("manually_transformed_coeffs.shape", manually_transformed_coeffs.shape)

    assert torch.allclose(
        transformed_coeffs,
        manually_transformed_coeffs.reshape(num_nodes, -1),
        atol=1e-7,
    )

    # check that back and forth trafo yields the same as before:

    # # test that 1p transforms correctly:
    # irrep = ShiftReps("5x1p")
    # coeffs = torch.randn(10, irrep.dim)
    # tensor_reps_transform = ShiftRepsTransform(irrep)
    # transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    # assert torch.allclose(
    #     transformed_coeffs.reshape(10, 5, 3),
    #     torch.matmul(coeffs.reshape(10, 5, 3), basis_change.matrices.transpose(-1, -2))
    #     * basis_change.det[:, None, None],
    #     atol=1e-7,
    # )

    # # test that 2n transforms correctly:
    # irrep = ShiftReps("5x2")
    # coeffs = torch.randn(10, irrep.dim)
    # tensor_reps_transform = ShiftRepsTransform(irrep)
    # transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    # naive_trafo = torch.einsum(
    #     "kij, klm, kcjm -> kcil",
    #     basis_change.matrices,
    #     basis_change.matrices,
    #     coeffs.reshape(10, 5, 3, 3),
    # )
    # assert torch.allclose(transformed_coeffs.reshape(10, 5, 3, 3), naive_trafo, atol=1e-7)

    # # test that 2p transforms correctly:
    # irrep = ShiftReps("5x2p")
    # coeffs = torch.randn(10, irrep.dim)
    # tensor_reps_transform = ShiftRepsTransform(irrep)
    # transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    # naive_trafo = torch.einsum(
    #     "kij, klm, kcjm -> kcil",
    #     basis_change.matrices,
    #     basis_change.matrices,
    #     coeffs.reshape(10, 5, 3, 3),
    # )
    # naive_trafo *= basis_change.det[:, None, None, None]
    # assert torch.allclose(transformed_coeffs.reshape(10, 5, 3, 3), naive_trafo, atol=1e-7)


if __name__ == "__main__":
    test_shiftreps()
