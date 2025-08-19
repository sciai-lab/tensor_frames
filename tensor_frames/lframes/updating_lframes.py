import warnings
from typing import Tuple, Union

import numpy as np
import torch
from e3nn.o3 import angles_to_matrix

from tensor_frames.lframes import LFrames
from tensor_frames.lframes.gram_schmidt import double_cross_orthogonalize, gram_schmidt
from tensor_frames.nn.mlp import MLPWrapped
from tensor_frames.reps import Irreps, TensorReps
from tensor_frames.utils.quaternions import quaternions_to_matrix


class GramSchmidtUpdateLFrames(torch.nn.Module):
    """Module for updating LFrames using Gram-Schmidt orthogonalization."""

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        hidden_channels: list,
        fix_gravitational_axis: bool = False,
        gravitational_axis_index: int = 1,
        exceptional_choice: str = "random",
        use_double_cross_product: bool = False,
        **mlp_kwargs,
    ):
        """Initialize the GramSchmidtUpdateLFrames module.

        Args:
            in_reps (Union[TensorReps, Irreps]): List of input representations.
            hidden_channels (list): List of hidden channel sizes for the MLP.
            fix_gravitational_axis (bool, optional): Whether to fix the gravitational axis. Defaults to False.
            gravitational_axis_index (int, optional): Index of the gravitational axis. Defaults to 1.
            exceptional_choice (str, optional): Choice of exceptional index. Defaults to "random".
            use_double_cross_product (bool, optional): Whether to use the double cross product to predict the vectors. Defaults to False.
            **mlp_kwargs: Additional keyword arguments for the MLPWrapped module.
        """
        super().__init__()
        self.in_reps = in_reps
        if fix_gravitational_axis:
            gravitational_axis = torch.zeros(3)
            gravitational_axis[gravitational_axis_index] = 1.0
            self.register_buffer("gravitational_axis", gravitational_axis)
            out_dim = 3

            # find the even permutation where index_order[gravitational_axis_index] is 0:
            index_order = [0, 1, 2]
            for i in range(3):
                current_index_order = np.roll(index_order, i)
                if current_index_order[gravitational_axis_index] == 0:
                    self.index_order = current_index_order.tolist()

        else:
            self.gravitational_axis = None
            self.index_order = None
            out_dim = 6
        self.mlp = MLPWrapped(
            in_channels=self.in_reps.dim,
            hidden_channels=hidden_channels + [out_dim],
            **mlp_kwargs,
        )

        self.coeffs_transform = self.in_reps.get_transform_class()
        self.exceptional_choice = exceptional_choice
        self.use_double_cross_product = use_double_cross_product

    def forward(
        self, x: torch.Tensor, lframes: LFrames, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, LFrames]:
        """Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.
            lframes (LFrames): LFrames object.
            batch (torch.Tensor): Batch tensor.

        Returns:
            Tuple[torch.Tensor, LFrames]: Tuple containing the updated input tensor and LFrames object.
        """
        out = self.mlp(x, batch=batch)
        if self.gravitational_axis is None:
            vec_1 = out[:, :3]
            vec_2 = out[:, 3:]
        else:
            vec_1 = self.gravitational_axis[None, :].repeat(out.shape[0], 1)
            vec_2 = out

        if self.use_double_cross_product:
            rot_matr = double_cross_orthogonalize(
                vec_1, vec_2, exceptional_choice=self.exceptional_choice
            )
        else:
            rot_matr = gram_schmidt(
                vec_1,
                vec_2,
                exceptional_choice=self.exceptional_choice,
            )

        if self.index_order is not None:
            rot_matr = rot_matr[:, self.index_order]
        new_lframes = LFrames(torch.einsum("ijk, ikn -> ijn", rot_matr, lframes.matrices))
        new_x = self.coeffs_transform(x, LFrames(rot_matr))

        return new_x, new_lframes


class AngleUpdateLFrames(torch.nn.Module):
    """Module for updating LFrames using angles."""

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        hidden_channels: list,
        use_atan2: bool = False,
        **mlp_kwargs,
    ):
        """Initialize the AngleUpdateLFrames module.

        Args:
            in_reps (Union[TensorReps, Irreps]): List of input representations.
            hidden_channels (list): List of hidden channel sizes for the MLP.
            use_atan (bool, optional): Whether to use atan2 function to predict the angles. Defaults to False.
            **mlp_kwargs: Additional keyword arguments for the MLPWrapped module.
        """
        super().__init__()
        self.in_reps = in_reps

        self.use_atan2 = use_atan2

        self.mlp = MLPWrapped(
            in_channels=self.in_reps.dim,
            hidden_channels=hidden_channels + [6] if use_atan2 else [3],
            **mlp_kwargs,
        )

        self.coeffs_transform = self.in_reps.get_transform_class()

    def forward(
        self, x: torch.Tensor, lframes: LFrames, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, LFrames]:
        """Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.
            lframes (LFrames): LFrames object.
            batch (torch.Tensor): Batch tensor.

        Returns:
            Tuple[torch.Tensor, LFrames]: Tuple containing the updated input tensor and LFrames object.
        """
        out = self.mlp(x, batch=batch)
        if self.use_atan2:
            out = out.view(-1, 3, 2)

            denominator = torch.where(out[..., 0].abs() < self.eps, self.eps, out[..., 0])
            angles = torch.arctan2(out[..., 1], denominator)
        else:
            angles = 2 * torch.atan(out)

        rot_matr = angles_to_matrix(angles[:, 0], angles[:, 1], angles[:, 2])

        new_lframes = LFrames(torch.einsum("ijk, ikn -> ijn", rot_matr, lframes.matrices))
        new_x = self.coeffs_transform(x, LFrames(rot_matr))

        return new_x, new_lframes


class QuaternionsUpdateLFrames(torch.nn.Module):
    """Module for updating LFrames using quaternions."""

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        hidden_channels: list,
        init_zero_angle: bool = False,
        eps: float = 1e-6,
        **mlp_kwargs,
    ):
        """Initialize the QuaternionsUpdateLFrames module.

        Args:
            in_reps (Union[TensorReps, Irreps]): List of input representations.
            hidden_channels (list): List of hidden channel sizes for the MLP.
            init_zero_angle (bool, optional): Whether to initialize angle weights to zero. Defaults to False.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
            **mlp_kwargs: Additional keyword arguments for the MLPWrapped module.
        """
        super().__init__()
        self.in_reps = in_reps
        self.eps = eps

        self.mlp = MLPWrapped(
            in_channels=self.in_reps.dim,
            hidden_channels=hidden_channels + [5],
            **mlp_kwargs,
        )
        self.coeffs_transform = self.in_reps.get_transform_class()

        if init_zero_angle:
            warnings.warn(
                "Make sure that the activation function is NOT ReLU, When using init_zero_angle = True."
            )
            self.set_angle_weights_to_zero()

    def set_angle_weights_to_zero(self):
        """Sets the relevant weights and biases to zero to achieve that the first output channel
        predicts zeros initially."""
        with torch.no_grad():
            if self.mlp.use_torchvision:
                # torchvision mlp
                self.mlp.mlp[-2].weight.data[1].zero_()
                self.mlp.mlp[-2].bias.data[1].zero_()
            else:
                # torch_geometric mlp
                self.mlp._modules["lins"][-1].weight.data[1].zero_()
                self.mlp._modules["lins"][-1].bias.data[1].zero_()

    def forward(
        self, x: torch.Tensor, lframes: LFrames, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, LFrames]:
        """Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.
            lframes (LFrames): LFrames object.
            batch (torch.Tensor): Batch tensor.

        Returns:
            Tuple[torch.Tensor, LFrames]: Tuple containing the updated input tensor and LFrames object.
        """
        out = self.mlp(x, batch=batch)
        denominator = torch.where(out[..., 0].abs() < self.eps, self.eps, out[..., 0])
        angle = torch.arctan2(out[..., 1], denominator)
        axis = torch.nn.functional.normalize(out[..., 2:], p=2, dim=-1)
        rot_matr = quaternions_to_matrix(
            torch.cat(
                [torch.cos(angle / 2).unsqueeze(-1), torch.sin(angle / 2).unsqueeze(-1) * axis],
                dim=-1,
            )
        )

        new_lframes = LFrames(torch.einsum("ijk, ikn -> ijn", rot_matr, lframes.matrices))
        new_x = self.coeffs_transform(x, LFrames(rot_matr))

        return new_x, new_lframes
