import torch
from e3nn.o3 import rand_matrix

from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.mlp_conv import MLPConv
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


def test_gcn_conv_layer():
    # Test the MLPConv layer
    # First test the MLPConv layer with tensor_reps

    # Define the input and output representations
    in_reps = TensorReps("16x0n + 8x1n + 2x1p + 4x2n")

    # Initialize the GCNConv layer
    mlp_conv = MLPConv(in_reps=in_reps, hidden_channels=[32, 32], out_channels=16)

    # create some dummy data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(10, in_reps.dim)
    lframes = LFrames(rand_matrix(10))

    # Perform the forward pass
    output = mlp_conv(edge_index, x, lframes)

    # add small invariance test:
    global_rot = rand_matrix(1).repeat(10, 1, 1)
    feature_transform = in_reps.get_transform_class()
    x_trafo = feature_transform(coeffs=x, lframes=LFrames(global_rot))
    lframes_trafo = []
    for i in range(10):
        lframes_trafo.append(lframes[i] @ global_rot[i].T)
    lframes_trafo = LFrames(torch.stack(lframes_trafo))
    output_trafo = mlp_conv(edge_index, x_trafo, lframes_trafo)

    assert torch.allclose(output, output_trafo)

    # Second test the MLPConv layer with tensor_reps

    # Define the input and output representations
    in_reps = Irreps("16x0n + 8x1n + 2x1p + 4x2n")

    # Initialize the GCNConv layer
    mlp_conv = MLPConv(in_reps=in_reps, hidden_channels=[32, 32], out_channels=16)

    # create some dummy data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(10, in_reps.dim)
    lframes = LFrames(rand_matrix(10))

    # Perform the forward pass
    output = mlp_conv(edge_index, x, lframes)

    # add small invariance test:
    global_rot = rand_matrix(1).repeat(10, 1, 1)
    feature_transform = in_reps.get_transform_class()
    x_trafo = feature_transform(coeffs=x, lframes=LFrames(global_rot))
    lframes_trafo = []
    for i in range(10):
        lframes_trafo.append(lframes[i] @ global_rot[i].T)
    lframes_trafo = LFrames(torch.stack(lframes_trafo))
    output_trafo = mlp_conv(edge_index, x_trafo, lframes_trafo)

    assert torch.allclose(output, output_trafo)
