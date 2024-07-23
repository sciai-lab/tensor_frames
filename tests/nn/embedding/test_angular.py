import torch
from e3nn.o3 import rand_matrix

from tensorframes.lframes import LFrames
from tensorframes.nn.embedding.angular import (
    SphericalHarmonicsEmbedding,
    TrivialAngularEmbedding,
    compute_edge_vec,
)


def test_angular():
    pos = torch.rand(10, 3)
    edge_index = torch.randint(0, 10, (2, 20))
    lframes = rand_matrix(10)
    parity_mask = torch.randint(0, 2, (10,), dtype=torch.bool)
    lframes[parity_mask, :, 0] *= -1
    lframes = LFrames(matrices=lframes)

    edge_vec = compute_edge_vec(pos, edge_index, lframes)

    # test trivial angular embedding
    trivial = TrivialAngularEmbedding(normalize=True)
    assert torch.allclose(
        trivial(edge_vec=edge_vec),
        edge_vec / (torch.norm(edge_vec, dim=-1, keepdim=True) + 1e-9),
        atol=1e-7,
    )
    assert torch.allclose(
        trivial(pos=pos, edge_index=edge_index, lframes=lframes),
        trivial(edge_vec=edge_vec),
        atol=1e-7,
    )
    assert torch.allclose(
        trivial(pos=(pos, pos), edge_index=edge_index, lframes=(lframes, lframes)),
        trivial(edge_vec=edge_vec),
        atol=1e-7,
    )

    # test spherical harmonics embedding
    sp_embedding = SphericalHarmonicsEmbedding(lmax=2)
    assert torch.allclose(
        sp_embedding(edge_vec=edge_vec),
        sp_embedding(pos=pos, edge_index=edge_index, lframes=lframes),
    )
