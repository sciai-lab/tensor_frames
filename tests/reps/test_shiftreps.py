import torch
from e3nn import o3

from tensorframes.lframes.lframes import ChangeOfLFrames, LFrames
from tensorframes.reps.shiftreps import ShiftReps, ShiftRepsTransform


def test_shiftreps():
    num_nodes = 10
    random_rot = o3.rand_matrix(num_nodes)
    flip_mask = torch.randint(0, 2, (num_nodes,), dtype=torch.bool)
    random_rot[flip_mask] *= -1
    shift = torch.randn(num_nodes, 3)
    lframes = LFrames(random_rot, shift=shift)

    # test that 0n transforms correctly:
    irrep = ShiftReps("5x0")
    assert irrep.dim == 5
    coeffs = torch.randn(num_nodes, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep)
    transformed_coeffs = shift_reps_transform(coeffs, lframes)
    assert torch.allclose(transformed_coeffs, coeffs)

    # test that 1n transforms correctly:
    mult = 1
    irrep = ShiftReps(f"{mult}x1")
    assert irrep.dim == mult * 4
    coeffs = torch.randn(num_nodes, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep)
    coeffs_clone = coeffs.clone()
    transformed_coeffs = shift_reps_transform(coeffs, lframes)
    assert torch.allclose(coeffs, coeffs_clone, atol=1e-6)

    # what I would expect the transformation to be:
    manually_transformed_coeffs = coeffs.reshape(num_nodes, mult, 4)
    manually_transformed_coeffs[..., :-1] = (
        torch.einsum("kij, kmj -> kmi", random_rot, manually_transformed_coeffs[..., :-1])
        - torch.einsum("kij, kj -> ki", random_rot, shift)[:, None, :]
    )

    assert torch.allclose(
        transformed_coeffs,
        manually_transformed_coeffs.reshape(num_nodes, -1),
        atol=1e-6,
    )

    # test that back and forth trafo yields the same as before for sorted reps:
    irrep = ShiftReps("3x0+3x1+2x2+3x2+4x3+4x4")
    coeffs = torch.randn(num_nodes, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep, use_parallel=True)
    print("is sorted?", shift_reps_transform.is_sorted)

    transformed_coeffs = shift_reps_transform(coeffs, lframes)
    back_and_forth_coeffs = shift_reps_transform(transformed_coeffs, lframes.inverse_lframes())

    print("max abs diff:", torch.max(torch.abs(coeffs - back_and_forth_coeffs)))
    assert torch.allclose(coeffs, back_and_forth_coeffs, atol=1e-4)

    # test that back and forth trafo yields the same as before for non-sorted reps:
    irrep = ShiftReps("5x0+3x1+2x2+7x5+9x3+1x4+2x4+7x0")
    # irrep = ShiftReps("2x2+1x1")
    coeffs = torch.randn(num_nodes, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep, use_parallel=True)
    print("is sorted?", shift_reps_transform.is_sorted)

    transformed_coeffs = shift_reps_transform(coeffs, lframes)
    back_and_forth_coeffs = shift_reps_transform(transformed_coeffs, lframes.inverse_lframes())

    print("max abs diff:", torch.max(torch.abs(coeffs - back_and_forth_coeffs)))
    assert torch.allclose(coeffs, back_and_forth_coeffs, atol=1e-4)

    # check trivial frame to frame transition:
    irrep = ShiftReps("3x0+3x1+2x2+3x2+4x3+4x4")
    coeffs = torch.randn(num_nodes, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep)

    change_of_lframes = ChangeOfLFrames(lframes, lframes)
    back_and_forth_coeffs = shift_reps_transform(coeffs, change_of_lframes)

    assert torch.allclose(coeffs, back_and_forth_coeffs, atol=1e-4)

    # testing compatibility of different shiftreps transforms if sorted:
    irrep = ShiftReps("3x0+3x1+2x2+3x2+4x3+4x4")
    coeffs = torch.randn(num_nodes, irrep.dim)
    coeffs_clone = coeffs.clone()
    shift_reps_transform1 = ShiftRepsTransform(irrep, use_parallel=True, avoid_einsum=False)
    shift_reps_transform2 = ShiftRepsTransform(irrep, use_parallel=True, avoid_einsum=True)
    shift_reps_transform3 = ShiftRepsTransform(irrep, use_parallel=False, avoid_einsum=True)
    transformed_coeffs1 = shift_reps_transform1(coeffs, lframes)
    transformed_coeffs2 = shift_reps_transform2(coeffs, lframes)
    transformed_coeffs3 = shift_reps_transform3(coeffs, lframes)
    assert torch.allclose(transformed_coeffs1, transformed_coeffs2, atol=1e-4)
    assert torch.allclose(transformed_coeffs1, transformed_coeffs3, atol=1e-4)
    # check that coeffs are not changed:
    assert torch.allclose(coeffs_clone, coeffs, atol=1e-7)

    # testing the compatibility if reps are not sorted:
    irrep = ShiftReps("5x0+3x1+2x2+7x5+9x3+1x4+2x4+7x0")
    coeffs = torch.randn(num_nodes, irrep.dim)
    coeffs_clone = coeffs.clone()
    shift_reps_transform1 = ShiftRepsTransform(irrep, use_parallel=True, avoid_einsum=False)
    shift_reps_transform2 = ShiftRepsTransform(irrep, use_parallel=True, avoid_einsum=True)
    shift_reps_transform3 = ShiftRepsTransform(irrep, use_parallel=False, avoid_einsum=True)
    transformed_coeffs1 = shift_reps_transform1(coeffs, lframes)
    transformed_coeffs2 = shift_reps_transform2(coeffs, lframes)
    transformed_coeffs3 = shift_reps_transform3(coeffs, lframes)
    # check that coeffs are not changed:
    assert torch.allclose(coeffs_clone, coeffs, atol=1e-7)
    assert torch.allclose(transformed_coeffs1, transformed_coeffs2, atol=1e-4)
    assert torch.allclose(transformed_coeffs1, transformed_coeffs3, atol=1e-4)

    # second test for non-sorted reps and sorted reps:
    irrep = ShiftReps("5x0+3x1+4x1+5x1+2x2+9x3+1x4+2x4+7x5")
    coeffs = torch.randn(num_nodes, irrep.dim)
    shift_reps_transform = ShiftRepsTransform(irrep)
    transformed_coeffs = shift_reps_transform(coeffs, lframes)
    transformed_coeffs_repeat = torch.cat([transformed_coeffs] * 2, dim=1)

    # now use the duplicated reps:
    irrep2 = irrep + irrep
    print("irrep dim:", irrep.dim, "irrep2 dim:", irrep2.dim)
    coeffs2 = torch.cat([coeffs] * 2, dim=1)
    shift_reps_transform = ShiftRepsTransform(irrep2)
    transformed_coeffs2 = shift_reps_transform(coeffs2, lframes)

    assert torch.allclose(transformed_coeffs_repeat, transformed_coeffs2, atol=1e-6)


if __name__ == "__main__":
    test_shiftreps()
    print("All tests passed!")
