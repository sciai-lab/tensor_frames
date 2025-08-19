import torch

from tensor_frames.lframes.lframes import LFrames
from tensor_frames.nn.gcn_conv import GCNConv
from tensor_frames.reps.irreps import Irreps
from tensor_frames.reps.tensorreps import TensorReps


def test_gcn_conv_layer():
    # Test the GCNConv layer
    # First test the GCNVonv layer with tensor_reps

    # Define the input and output representations
    in_reps = TensorReps("16x0n + 8x1n + 2x1p + 4x2n")
    out_reps = TensorReps("16x0n + 8x1n + 2x1p + 4x2n")

    # Initialize the GCNConv layer
    gcn_conv = GCNConv(in_reps, out_reps)

    # create some dummy data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, in_reps.dim)
    lframes = LFrames(torch.eye(3).unsqueeze(0).repeat(3, 1, 1))

    # Perform the forward pass
    output = gcn_conv(edge_index, x, lframes)

    # do the same with irreps
    in_reps = Irreps("16x0n + 8x1n + 2x1p + 4x2n")
    out_reps = Irreps("16x0n + 8x1n + 2x1p + 4x2n")

    # Initialize the GCNConv layer
    gcn_conv = GCNConv(in_reps, out_reps)

    # create some dummy data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, in_reps.dim)
    lframes = LFrames(torch.eye(3).unsqueeze(0).repeat(3, 1, 1))

    # Perform the forward pass
    output = gcn_conv(edge_index, x, lframes)
