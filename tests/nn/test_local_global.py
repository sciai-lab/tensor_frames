import torch

from tensorframes.lframes.classical_lframes import RandomLFrames
from tensorframes.nn.local_global import FromGlobalToLocalFrame, FromLocalToGlobalFrame
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


def test_local_global_trafo_irreps():
    """Test the local to global and global to local transformation by concatenating the two
    transformations in the case of Irreps."""

    num_nodes = 5

    lframes = RandomLFrames()(torch.randn(num_nodes, 3))

    # test with Irreps
    irreps = Irreps("3x0n + 2x1n + 3x1p + 1x2n")

    x = torch.randn(num_nodes, irreps.dim)

    trafo_global_to_local = FromGlobalToLocalFrame(irreps)
    trafo_local_to_global = FromLocalToGlobalFrame(irreps)

    x_local = trafo_global_to_local(x, lframes)
    x_global = trafo_local_to_global(x_local, lframes)

    assert torch.allclose(x, x_global, atol=1e-6)


def test_local_global_trafo_tensorreps():
    """Test the local to global and global to local transformation by concatenating the two
    transformations in the case of TensorReps."""

    num_nodes = 5

    lframes = RandomLFrames()(torch.randn(num_nodes, 3))

    tensorreps = TensorReps("3x0n + 2x1n + 3x1p + 1x2n")

    x = torch.randn(num_nodes, tensorreps.dim)

    trafo_global_to_local = FromGlobalToLocalFrame(tensorreps)
    trafo_local_to_global = FromLocalToGlobalFrame(tensorreps)

    x_local = trafo_global_to_local(x, lframes)
    x_global = trafo_local_to_global(x_local, lframes)

    assert torch.allclose(x, x_global, atol=1e-6)
