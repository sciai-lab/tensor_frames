import torch

from tensorframes.lframes.gram_schmidt import gram_schmidt


def test_gram_schmidt():
    """Test function for the gram_schmidt function.

    This function creates a random set of vectors and tests the gram_schmidt function for both
    three vectors and two vectors. It checks the shape of the resulting matrix and verifies that
    the resulting matrix is orthogonal.
    """
    # create a random set of vectors
    N = 20
    x_axis = torch.randn(N, 3)
    y_axis = torch.randn(N, 3)
    z_axis = torch.randn(N, 3)

    # now test the gram schmidt function for three vectors
    lframes = gram_schmidt(x_axis, y_axis, z_axis)

    # check shape of resulting matrix
    assert lframes.shape == (
        N,
        3,
        3,
    ), "The resulting matrix has the wrong shape, should be (N, 3, 3)"

    # check that the resulting matrix is orthogonal
    for i in range(N):
        assert torch.allclose(
            torch.einsum("ij,jl->il", lframes[i], lframes[i].T), torch.eye(3), atol=1e-5
        ), "The resulting matrix is not orthogonal"

    # now test the gram schmidt function for two vectors
    lframes = gram_schmidt(x_axis, y_axis)

    # check shape of resulting matrix
    assert lframes.shape == (
        N,
        3,
        3,
    ), "The resulting matrix has the wrong shape, should be (N, 3, 3)"

    # check that the resulting matrix is orthogonal
    for i in range(N):
        assert torch.allclose(
            torch.einsum("ij,jl->il", lframes[i], lframes[i].T), torch.eye(3), atol=1e-5
        ), "The resulting matrix is not orthogonal"
