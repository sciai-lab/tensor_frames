import numpy as np
import torch
from sklearn.decomposition import PCA

from tensorframes.lframes.classical_lframes import PCALFrames, RandomGlobalLFrames
from tensorframes.reps.tensorreps import TensorReps


def test_pca_lframes():
    """Tests pca based lframes."""
    num_points = 100
    pos = torch.rand(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    pca_lframes = PCALFrames(r=1, max_num_neighbors=16)
    lframes = pca_lframes(pos=pos, idx=None, batch=batch)

    dets = torch.linalg.det(lframes.matrices)
    print("dets: ", dets.mean(), dets.max(), dets.min())
    print("dets counts: ", torch.unique(torch.round(dets, decimals=2), return_counts=True))
    assert torch.allclose(torch.linalg.det(lframes.matrices).abs(), torch.ones(num_points))
    idents = torch.bmm(lframes.matrices, lframes.matrices.transpose(1, 2))
    print("diff from identity: ", (idents - torch.eye(3).expand(num_points, -1, -1)).abs().max())
    assert torch.allclose(
        torch.bmm(lframes.matrices, lframes.matrices.transpose(1, 2)),
        torch.eye(3).expand(num_points, -1, -1),
        atol=1e-5,
    )

    # create a test case to check against PCA:
    num_points = 5000
    A = np.random.randn(3, 3)
    pos = torch.from_numpy(
        np.random.multivariate_normal(mean=np.zeros(3), cov=A @ A.T, size=num_points)
    ).float()
    pos[0] = 0.0  # the mean

    pca_lframes = PCALFrames(r=100, max_num_neighbors=num_points)
    lframes = pca_lframes(pos=pos, idx=None, batch=None)

    print("lframes: ", lframes.matrices[0])

    # check that the first lframe is the PCA frame:
    pca = PCA(n_components=3)
    pca.fit(pos.numpy())
    print("pca components: ", pca.components_)

    # divide them and see if they are the same up to a sign:
    ratio = torch.from_numpy(pca.components_).flip(dims=(0,)) / lframes.matrices[0]
    print("ratio: ", ratio)
    assert torch.allclose(ratio.abs(), torch.ones(3), atol=1e-3)

    # ckeck that lframes are equivariant:
    num_points = 100
    pos = torch.rand(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    lframes_learner = PCALFrames(r=1, max_num_neighbors=num_points)
    lframes1 = lframes_learner(pos=pos, batch=batch)

    # check that x is invariant and lframes are equivariant:
    random_trafo = RandomGlobalLFrames()(pos=pos)
    pos_rot = TensorReps("1x1").get_transform_class()(coeffs=pos, basis_change=random_trafo)
    lframes2 = lframes_learner(pos=pos_rot, batch=batch)

    # check that lframes are equivariant:
    lframes_matrices1 = torch.bmm(lframes1.matrices, random_trafo.matrices.transpose(1, 2))
    diff = (lframes2.matrices - lframes_matrices1).abs()
    print("frames max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(lframes2.matrices, lframes_matrices1, atol=1e-3)


if __name__ == "__main__":
    test_pca_lframes()
