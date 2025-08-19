import torch

from tensor_frames.lframes.classical_lframes import RandomGlobalLFrames
from tensor_frames.lframes.learning_lframes import WrappedLearnedLFrames
from tensor_frames.nn.embedding.radial import TrivialRadialEmbedding
from tensor_frames.reps.irreps import Irreps
from tensor_frames.reps.tensorreps import TensorReps


def test_wrapped_learned_lframes():
    num_points = 100
    in_reps = TensorReps("12x0")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.rand(num_points, in_reps.dim)
    pos = torch.rand(num_points, 3)
    batch = torch.tensor([1, 2]).repeat_interleave(num_points // 2)

    lframes_learner = WrappedLearnedLFrames(
        in_reps=in_reps,
        hidden_channels=[16, 16],
        radial_module=TrivialRadialEmbedding(),
        max_num_neighbors=16,
        max_radius=100,
        exceptional_choice="zero",
    )
    x_trafo, lframes = lframes_learner(x=x.clone(), pos=pos, batch=batch)

    # check that in scalar case: x=x_trafo
    assert torch.allclose(x, x_trafo)

    num_points = 100
    in_reps = TensorReps("5x0")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.rand(num_points, in_reps.dim)
    pos = torch.rand(num_points, 3)
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
    random_trafo = RandomGlobalLFrames()(pos=pos)
    x_rot = in_reps_trafo(coeffs=x.clone(), basis_change=random_trafo)
    pos_rot = TensorReps("1x1").get_transform_class()(coeffs=pos, basis_change=random_trafo)
    x_trafo2, lframes2 = lframes_learner(x=x_rot, pos=pos_rot, batch=batch)

    diff = (x_trafo1 - x_trafo2).abs()
    print("tensorreps: x max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(x_trafo1, x_trafo2, atol=1e-3)

    # check that lframes are equivariant:
    lframes_matrices1 = torch.bmm(lframes1.matrices, random_trafo.matrices.transpose(1, 2))
    diff = (lframes2.matrices - lframes_matrices1).abs()
    print("tensorreps: frames max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(lframes2.matrices, lframes_matrices1, atol=1e-3)

    num_points = 100
    in_reps = TensorReps("5x1")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.rand(num_points, in_reps.dim)
    pos = torch.rand(num_points, 3)
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
    random_trafo = RandomGlobalLFrames(flip_probability=0.5)(pos=pos)
    x_rot = in_reps_trafo(coeffs=x.clone(), basis_change=random_trafo)
    pos_rot = TensorReps("1x1").get_transform_class()(coeffs=pos, basis_change=random_trafo)
    x_trafo2, lframes2 = lframes_learner(x=x_rot, pos=pos_rot, batch=batch)

    diff = (x_trafo1 - x_trafo2).abs()
    print("tensorreps (double cross): x max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(x_trafo1, x_trafo2, atol=1e-3)

    # check that lframes are equivariant:
    lframes_matrices1 = torch.bmm(lframes1.matrices, random_trafo.matrices.transpose(1, 2))
    diff = (lframes2.matrices - lframes_matrices1).abs()
    print("tensorreps (double cross): frames max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(lframes2.matrices, lframes_matrices1, atol=1e-3)

    num_points = 100
    in_reps = Irreps("5x1")
    in_reps_trafo = in_reps.get_transform_class()
    x = torch.rand(num_points, in_reps.dim)
    pos = torch.rand(num_points, 3)
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
    random_trafo = RandomGlobalLFrames()(pos=pos)
    x_rot = in_reps_trafo(coeffs=x.clone(), basis_change=random_trafo)
    pos_rot = TensorReps("1x1").get_transform_class()(coeffs=pos, basis_change=random_trafo)
    x_trafo2, lframes2 = lframes_learner(x=x_rot, pos=pos_rot, batch=batch)

    diff = (x_trafo1 - x_trafo2).abs()
    print("irreps: x max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(x_trafo1, x_trafo2, atol=1e-3)

    # check that lframes are equivariant:
    lframes_matrices1 = torch.bmm(lframes1.matrices, random_trafo.matrices.transpose(1, 2))
    diff = (lframes2.matrices - lframes_matrices1).abs()
    print("irreps: frames max diff", diff.max(), "mean diff", diff.mean())
    assert torch.allclose(lframes2.matrices, lframes_matrices1, atol=1e-3)


if __name__ == "__main__":
    test_wrapped_learned_lframes()
