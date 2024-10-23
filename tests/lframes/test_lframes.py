import torch

from tensorframes.lframes.classical_lframes import RandomLFrames
from tensorframes.lframes.lframes import ChangeOfLFrames, LFrames


def test_lframes():
    # create lframes object:
    num_lframes = 10
    lframes1 = RandomLFrames()(pos=torch.zeros(num_lframes, 3))
    lframes2 = RandomLFrames()(pos=torch.zeros(num_lframes, 3))

    # test change of lframes:
    change = ChangeOfLFrames(lframes_start=lframes1, lframes_end=lframes1)

    # matrices should be the identity:
    assert torch.allclose(change.matrices, torch.eye(3).repeat(num_lframes, 1, 1), atol=1e-6)

    # check that wigner D matrices are the identity:
    assert torch.allclose(change.wigner_D(2), torch.eye(5).repeat(num_lframes, 1, 1), atol=1e-6)

    # test inverse of change of lframes
    change = ChangeOfLFrames(lframes_start=lframes1, lframes_end=lframes2)
    change_inv = change.inverse_lframes()

    assert torch.allclose(
        torch.bmm(change.matrices, change_inv.matrices),
        torch.eye(3).repeat(num_lframes, 1, 1),
        atol=1e-6,
    )

    # test the inverse of lframes:
    lframes = RandomLFrames()(pos=torch.zeros(num_lframes, 3))
    lframes.wigner_D(2)
    lframes_inv = lframes.inverse_lframes()

    assert torch.allclose(
        torch.bmm(lframes.matrices, lframes_inv.matrices),
        torch.eye(3).repeat(num_lframes, 1, 1),
        atol=1e-6,
    )

    # test that the angles are close to a multiple of 2*pi:
    lframes_inv_no_cache = LFrames(lframes_inv.matrices)
    angles_diff = lframes_inv_no_cache.angles - lframes_inv.angles
    assert torch.allclose(
        angles_diff, torch.round(angles_diff / (2 * torch.pi)) * (2 * torch.pi), atol=1e-6
    )

    # test the wigner:
    assert torch.allclose(lframes_inv_no_cache.wigner_D(2), lframes_inv.wigner_D(2), atol=1e-6)
