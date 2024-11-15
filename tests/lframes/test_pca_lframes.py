import torch

from tensorframes.lframes.classical_lframes import PCALFrames
from tensorframes.reps.tensorreps import TensorReps


def test_pca_lframes():
    """Tests pca based lframes."""
    num_points = 100
    in_reps = TensorReps("12x0")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.rand(num_points, in_reps.dim)
    pos = torch.rand(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    pca_lframes = PCALFrames(r=0.1, max_num_neighbors=16)
    lframes = pca_lframes(pos=pos, idx=None, batch=batch)

    dets = torch.linalg.det(lframes.matrices)
    print("dets: ", dets.mean(), dets.max(), dets.min())
    assert torch.allclose(torch.linalg.det(lframes.matrices).abs(), torch.ones(num_points))
    idents = torch.bmm(lframes.matrices, lframes.matrices.transpose(1, 2))
    print("diff from identity: ", (idents - torch.eye(3).expand(num_points, -1, -1)).abs().max())
    assert torch.allclose(
        torch.bmm(lframes.matrices, lframes.matrices.transpose(1, 2)),
        torch.eye(3).expand(num_points, -1, -1),
        atol=1e-5,
    )


if __name__ == "__main__":
    test_pca_lframes()
