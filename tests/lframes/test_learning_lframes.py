import torch
from e3nn.o3 import rand_matrix

from tensorframes.lframes.learning_lframes import WrappedLearnedLFrames
from tensorframes.lframes.lframes import LFrames
from tensorframes.nn.embedding.radial import TrivialRadialEmbedding
from tensorframes.reps.irreps import Irreps
from tensorframes.reps.tensorreps import TensorReps


def test_edge_conv_layer():
    num_points = 100
    in_reps = TensorReps("12x0")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.randn(num_points, in_reps.dim)
    pos = torch.randn(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    lframes_learner = WrappedLearnedLFrames(
        in_reps=in_reps,
        hidden_channels=[16, 16],
        radial_module=TrivialRadialEmbedding(),
        max_num_neighbors=16,
        max_radius=1.0,
    )
    x_trafo, lframes = lframes_learner(x=x.clone(), pos=pos, batch=batch)

    # check that in scalar case: x=x_trafo
    assert torch.allclose(x, x_trafo)

    num_points = 100
    in_reps = TensorReps("5x0")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.randn(num_points, in_reps.dim)
    pos = torch.randn(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    lframes_learner = WrappedLearnedLFrames(
        in_reps=in_reps,
        hidden_channels=[16, 16],
        radial_module=TrivialRadialEmbedding(),
        max_radius=100,
        exceptional_choice="zero",
        use_double_cross_product=False,
    )
    x_trafo1, lframes1 = lframes_learner(x=x.clone(), pos=pos, batch=batch)

    # check that x is invariant and lframes are equivariant:
    random_rot = rand_matrix(1).repeat(100, 1, 1)
    random_rot = -random_rot
    x_rot = in_reps_trafo(coeffs=x.clone(), basis_change=LFrames(random_rot))
    pos_rot = TensorReps("1x1").get_transform_class()(coeffs=pos, basis_change=LFrames(random_rot))
    x_trafo2, lframes2 = lframes_learner(x=x_rot, pos=pos_rot, batch=batch)

    diff = (x_trafo1 - x_trafo2).abs()
    print("tensorreps: x max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(x_trafo1, x_trafo2, atol=1e-4)

    # check that lframes are equivariant:
    lframes_matrices1 = torch.bmm(lframes1.matrices, random_rot.transpose(1, 2))
    diff = (lframes2.matrices - lframes_matrices1).abs()
    print("tensorreps: frames max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(lframes2.matrices, lframes_matrices1, atol=1e-4)

    num_points = 100
    in_reps = TensorReps("5x0")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.randn(num_points, in_reps.dim)
    pos = torch.randn(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    lframes_learner = WrappedLearnedLFrames(
        in_reps=in_reps,
        hidden_channels=[16, 16],
        radial_module=TrivialRadialEmbedding(),
        max_radius=100,
        exceptional_choice="zero",
        use_double_cross_product=True,
    )
    x_trafo1, lframes1 = lframes_learner(x=x.clone(), pos=pos, batch=batch)

    # check that x is invariant and lframes are equivariant:
    random_rot = rand_matrix(1).repeat(100, 1, 1)
    random_rot = -random_rot
    x_rot = in_reps_trafo(coeffs=x.clone(), basis_change=LFrames(random_rot))
    pos_rot = TensorReps("1x1").get_transform_class()(coeffs=pos, basis_change=LFrames(random_rot))
    x_trafo2, lframes2 = lframes_learner(x=x_rot, pos=pos_rot, batch=batch)

    diff = (x_trafo1 - x_trafo2).abs()
    print("tensorreps (double cross): x max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(x_trafo1, x_trafo2, atol=1e-4)

    # check that lframes are equivariant:
    lframes_matrices1 = torch.bmm(lframes1.matrices, random_rot.transpose(1, 2))
    diff = (lframes2.matrices - lframes_matrices1).abs()
    print("tensorreps (double cross): frames max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(lframes2.matrices, lframes_matrices1, atol=1e-4)

    num_points = 100
    in_reps = Irreps("5x0")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.randn(num_points, in_reps.dim)
    pos = torch.randn(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    lframes_learner = WrappedLearnedLFrames(
        in_reps=in_reps,
        hidden_channels=[16, 16],
        radial_module=TrivialRadialEmbedding(),
        max_radius=100,
        exceptional_choice="zero",
        use_double_cross_product=False,
    )
    x_trafo1, lframes1 = lframes_learner(x=x.clone(), pos=pos, batch=batch)

    # check that x is invariant and lframes are equivariant:
    random_rot = rand_matrix(1).repeat(100, 1, 1)
    random_rot = -random_rot
    x_rot = in_reps_trafo(coeffs=x.clone(), basis_change=LFrames(random_rot))
    pos_rot = Irreps("1x1").get_transform_class()(coeffs=pos, basis_change=LFrames(random_rot))
    x_trafo2, lframes2 = lframes_learner(x=x_rot, pos=pos_rot, batch=batch)

    diff = (x_trafo1 - x_trafo2).abs()
    print("irreps: x max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(x_trafo1, x_trafo2, atol=1e-4)

    # check that lframes are equivariant:
    lframes_matrices1 = torch.bmm(lframes1.matrices, random_rot.transpose(1, 2))
    diff = (lframes2.matrices - lframes_matrices1).abs()
    print("irreps: frames max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(lframes2.matrices, lframes_matrices1, atol=1e-4)
