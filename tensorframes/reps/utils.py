from typing import Union

import torch

from tensorframes.reps.irreps import Irrep, Irreps
from tensorframes.reps.tensorreps import TensorRep, TensorReps


def extract_even_scalar_mask_from_reps(tensor_reps: Union[TensorReps, Irreps]) -> torch.Tensor:
    scalar_mask = torch.zeros(tensor_reps.dim, dtype=torch.bool)
    start_ind = 0
    for rep_i in tensor_reps:
        end_ind = start_ind + rep_i.dim
        if rep_i.rep == TensorRep(order=0, p=1):
            scalar_mask[start_ind:end_ind] = True
        elif rep_i.rep == Irrep(angular_momentum=0, p=1):
            scalar_mask[start_ind:end_ind] = True
        start_ind = end_ind

    return scalar_mask
