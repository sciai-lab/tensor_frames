import numpy as np
import torch
from sklearn.decomposition import PCA

from tensorframes.lframes.classical_lframes import PCALFrames


def test_pca_lframes():
    """Tests pca based lframes."""
    num_points = 100
    pos = torch.rand(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    pca_lframes = PCALFrames(r=1, max_num_neighbors=16)
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


if __name__ == "__main__":
    test_pca_lframes()
