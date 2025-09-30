import torch
from e3nn.o3 import rand_matrix

from tensor_frames.lframes import LFrames
from tensor_frames.nn.embedding.radial import (
    BesselEmbedding,
    GaussianEmbedding,
    compute_edge_vec,
)
from tensor_frames.nn.envelope import EnvelopePoly


def test_compute_edge_vec():
    pos = torch.rand(10, 3)
    edge_index = torch.randint(0, 10, (2, 20))
    lframes = rand_matrix(10)
    parity_mask = torch.randint(0, 2, (10,), dtype=torch.bool)
    lframes[parity_mask, :, 0] *= -1
    lframes = LFrames(matrices=lframes)

    # edge_vec without lframes:
    edge_vec = compute_edge_vec(pos, edge_index)

    # compare against manual computation:
    edge_vec_ = pos[edge_index[0]] - pos[edge_index[1]]

    assert torch.allclose(edge_vec, edge_vec_)

    # with tuple pos:
    subset_index = torch.randint(0, 10, (5,))
    pos_tuple = (pos, pos[subset_index])
    edge_index_ = torch.zeros((2, 5), dtype=torch.long)
    edge_index_[0] = edge_index_[0][:5]
    edge_index_[1] = torch.randperm(5)

    edge_vec = compute_edge_vec(pos_tuple, edge_index_)

    # compare against manual computation:
    edge_vec_ = pos_tuple[0][edge_index_[0]] - pos_tuple[1][edge_index_[1]]

    assert torch.allclose(edge_vec, edge_vec_)

    # with lframes:
    edge_vec = compute_edge_vec(pos, edge_index, lframes)

    # compare against manual computation:
    edge_vec_ = pos[edge_index[0]] - pos[edge_index[1]]
    edge_vec_ = torch.einsum("ik,ijk->ij", edge_vec_, lframes.matrices[edge_index[1]])
    assert torch.allclose(edge_vec, edge_vec_)


def test_radial():
    pos = torch.rand(10, 3)
    edge_index = torch.randint(0, 10, (2, 20))
    lframes = rand_matrix(10)
    parity_mask = torch.randint(0, 2, (10,), dtype=torch.bool)
    lframes[parity_mask, :, 0] *= -1
    lframes = LFrames(matrices=lframes)

    edge_vec1 = compute_edge_vec(pos, edge_index)
    edge_vec2 = compute_edge_vec(pos, edge_index, lframes)

    edge_vec2_back = []
    lframes_i = lframes.index_select(edge_index[1]).matrices.reshape(-1, 3, 3)
    for i in range(20):
        edge_vec2_back.append(lframes_i[i].T @ edge_vec2[i])  # rotate back
    edge_vec2_back = torch.stack(edge_vec2_back)
    assert torch.allclose(edge_vec1, edge_vec2_back)

    # test gaussian radial embedding
    gauss = GaussianEmbedding()
    assert torch.allclose(gauss(edge_vec=edge_vec1), gauss(edge_vec=edge_vec2))
    assert torch.allclose(
        gauss(pos=pos, edge_index=edge_index), gauss(pos=(pos, pos), edge_index=edge_index)
    )
    assert torch.allclose(gauss(pos=pos, edge_index=edge_index), gauss(edge_vec=edge_vec1))

    # test bessel radial embedding
    bessel = BesselEmbedding(num_frequencies=10, cutoff=10.0, envelope=EnvelopePoly(2))
    assert torch.allclose(bessel(edge_vec=edge_vec1), bessel(edge_vec=edge_vec2), atol=1e-6)
    assert torch.allclose(
        bessel(pos=pos, edge_index=edge_index),
        bessel(pos=(pos, pos), edge_index=edge_index),
        atol=1e-6,
    )
    assert torch.allclose(
        bessel(pos=pos, edge_index=edge_index), bessel(edge_vec=edge_vec1), atol=1e-6
    )
