import torch
from e3nn import o3

from tensorframes.lframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.reps.tensorreps import TensorRep, TensorReps, TensorRepsTransform


def test_tensorreps():
    rep_1 = "5x0p+3x1+3x2+5x3"
    rep_2 = "5x0"

    tensor_reps_1 = TensorReps(rep_1)
    tensor_reps_2 = TensorReps(rep_2)
    tensor_reps_3 = TensorReps(
        [(5, TensorRep(0, 1)), (3, TensorRep(0, 1)), (2, TensorRep(0, 1)), (1, TensorRep(0, 1))]
    )

    num_nodes = 10
    random_rot = o3.rand_matrix(num_nodes)

    coeffs = torch.randn(num_nodes, tensor_reps_1.dim)
    tensor_reps_transform = TensorRepsTransform(tensor_reps_1)
    lframes = LFrames(random_rot)

    random_rot = o3.rand_matrix(20)
    flip_mask = torch.randint(0, 2, (20,), dtype=torch.bool)
    random_rot[flip_mask] *= -1
    basis_change = ChangeOfLFrames(
        LFrames(random_rot[:num_nodes]), LFrames(random_rot[num_nodes:])
    )

    # test that 0n transforms correctly:
    irrep = TensorReps("5x0")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(transformed_coeffs, coeffs)

    # test that 0p transforms correctly:
    irrep = TensorReps("5x0p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)

    assert torch.allclose(transformed_coeffs, coeffs * basis_change.det[:, None], atol=1e-7)

    # test that 1n transforms correctly:
    irrep = TensorReps("5x1")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(
        transformed_coeffs.reshape(num_nodes, 5, 3),
        torch.matmul(coeffs.reshape(num_nodes, 5, 3), basis_change.matrices.transpose(-1, -2)),
        atol=1e-7,
    )

    # test that 1p transforms correctly:
    irrep = TensorReps("5x1p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    assert torch.allclose(
        transformed_coeffs.reshape(num_nodes, 5, 3),
        torch.matmul(coeffs.reshape(num_nodes, 5, 3), basis_change.matrices.transpose(-1, -2))
        * basis_change.det[:, None, None],
        atol=1e-7,
    )

    # test that 2n transforms correctly:
    irrep = TensorReps("5x2")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    naive_trafo = torch.einsum(
        "kij, klm, kcjm -> kcil",
        basis_change.matrices,
        basis_change.matrices,
        coeffs.reshape(num_nodes, 5, 3, 3),
    )
    assert torch.allclose(transformed_coeffs.reshape(num_nodes, 5, 3, 3), naive_trafo, atol=1e-7)

    # test that 2p transforms correctly:
    irrep = TensorReps("5x2p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), basis_change)
    naive_trafo = torch.einsum(
        "kij, klm, kcjm -> kcil",
        basis_change.matrices,
        basis_change.matrices,
        coeffs.reshape(num_nodes, 5, 3, 3),
    )
    naive_trafo *= basis_change.det[:, None, None, None]
    assert torch.allclose(transformed_coeffs.reshape(num_nodes, 5, 3, 3), naive_trafo, atol=1e-7)

    # test back and forth trafo yields the same as before:
    irrep = TensorReps("5x0+3x1p+2x2+7x5p+9x3+1x4p+2x4+7x0p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)

    transformed_coeffs = tensor_reps_transform(coeffs, lframes)
    back_and_forth_coeffs = tensor_reps_transform(transformed_coeffs, lframes.inverse_lframes())

    assert torch.allclose(coeffs, back_and_forth_coeffs, atol=1e-6)

    # test trivial frame to frame transition:
    irrep = TensorReps("5x0+3x1p+2x2+7x5p+9x3+1x4p+2x4+7x0p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)

    change_of_lframes = ChangeOfLFrames(lframes, lframes)
    back_and_forth_coeffs = tensor_reps_transform(coeffs, change_of_lframes)

    assert torch.allclose(coeffs, back_and_forth_coeffs, atol=1e-6)

    # testing compatibility of different tensorreps transforms:
    irrep = TensorReps("5x0+3x1p+2x2+7x5p+9x3+1x4p+2x4+7x0p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform1 = TensorRepsTransform(irrep, use_parallel=True, avoid_einsum=False)
    tensor_reps_transform2 = TensorRepsTransform(irrep, use_parallel=True, avoid_einsum=True)
    tensor_reps_transform3 = TensorRepsTransform(irrep, use_parallel=False, avoid_einsum=True)
    transformed_coeffs1 = tensor_reps_transform1(coeffs, lframes)
    transformed_coeffs2 = tensor_reps_transform2(coeffs, lframes)
    transformed_coeffs3 = tensor_reps_transform3(coeffs, lframes)
    assert torch.allclose(transformed_coeffs1, transformed_coeffs2, atol=1e-6)
    assert torch.allclose(transformed_coeffs1, transformed_coeffs3, atol=1e-6)

    # testing the compatibility if things are sorted:
    irrep = TensorReps("5x0+3x1p+4x1p+5x1n+2x2+9x3+1x4p+2x4+7x5p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    coeffs_clone = coeffs.clone()
    tensor_reps_transform1 = TensorRepsTransform(irrep, use_parallel=True, avoid_einsum=False)
    tensor_reps_transform2 = TensorRepsTransform(irrep, use_parallel=True, avoid_einsum=True)
    tensor_reps_transform3 = TensorRepsTransform(irrep, use_parallel=False, avoid_einsum=True)
    transformed_coeffs1 = tensor_reps_transform1(coeffs, lframes)
    transformed_coeffs2 = tensor_reps_transform2(coeffs, lframes)
    transformed_coeffs3 = tensor_reps_transform3(coeffs, lframes)
    # check that coeffs are not changed:
    assert torch.allclose(coeffs_clone, coeffs, atol=1e-6)

    assert torch.allclose(transformed_coeffs1, transformed_coeffs2, atol=1e-6)
    assert torch.allclose(transformed_coeffs1, transformed_coeffs3, atol=1e-6)

    # second test for non-sorted reps and sorted reps:
    irrep = TensorReps("5x0+3x1p+4x1p+5x1n+2x2+9x3+1x4p+2x4+7x5p")
    coeffs = torch.randn(num_nodes, irrep.dim)
    tensor_reps_transform = TensorRepsTransform(irrep)
    transformed_coeffs = tensor_reps_transform(coeffs, lframes)
    transformed_coeffs_repeat = torch.cat([transformed_coeffs] * 2, dim=1)

    # now use the duplicated reps:
    irrep2 = irrep + irrep
    coeffs2 = torch.cat([coeffs] * 2, dim=1)
    tensor_reps_transform = TensorRepsTransform(irrep2)
    transformed_coeffs2 = tensor_reps_transform(coeffs2, lframes)

    assert torch.allclose(transformed_coeffs_repeat, transformed_coeffs2, atol=1e-6)


if __name__ == "__main__":
    test_tensorreps()
    print("All tests passed!")
