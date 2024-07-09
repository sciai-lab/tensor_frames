from typing import Union

import torch
from torch.nn import Module

from tensorframes.lframes import ChangeOfLFrames
from tensorframes.reps.irreps import Irreps, IrrepsTransform
from tensorframes.reps.tensorreps import TensorReps, TensorRepsTransform


class TransfromCoeffs(Module):
    """Transform the coefficients of a representation."""

    def __init__(self, reps: Union[Irreps, TensorReps], **transform_kwargs) -> None:
        """Initialize the TransformCoeffs class.

        Args:
            reps (Union[Irreps, TensorReps]): The representation object to transform.
        """
        super().__init__()
        self.reps = reps
        if isinstance(reps, Irreps):
            self.transform_coeffs = IrrepsTransform(irreps=reps)
        elif isinstance(reps, TensorReps):
            self.transform_coeffs = TensorRepsTransform(tensorreps=reps, **transform_kwargs)
        else:
            raise ValueError(f"Unknown representation type {type(reps)}.")

    def forward(self, coeffs: torch.Tensor, basis_change: ChangeOfLFrames) -> torch.Tensor:
        """Transform the coefficients of the representation.

        Args:
            coeffs (torch.Tensor): The coefficients to transform.
            transform (Union[IrrepsTransform, TensorRepsTransform]): The transformation object.

        Returns:
            torch.Tensor: The transformed coefficients.
        """
        return self.transform_coeffs(coeffs, basis_change)
