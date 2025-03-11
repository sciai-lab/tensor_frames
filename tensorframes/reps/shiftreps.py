import torch
from torch import Tensor

from tensorframes.lframes.lframes import LFrames
from tensorframes.reps.tensorreps import TensorReps, TensorRepsTransform


class ShiftReps(TensorReps):
    """Represents a collection of shifting tensor representations."""

    def __init__(self, tensor_reps, spatial_dim=3):
        super().__init__(tensor_reps, spatial_dim)

    @property
    def dim(self) -> int:
        """
        int: The total dimension of the tensor reps.
        """
        if self._dim is None:
            self._dim = sum(
                (self.spatial_dim + 1) ** mul_ir.rep.order for mul_ir in self
            )  # just use one feature that is overwritten
        return self._dim

    def get_transform_class(self, use_parallel: bool = True, avoid_einsum: bool = False):
        """Returns the tensor reps transform class.

        Args:
            use_parallel (bool, optional): Whether to use parallel computation for the transformation. Defaults to True.
            avoid_einsum (bool, optional): Whether to avoid using `torch.einsum` for the transformation. Defaults to False.

        Returns:
            TensorRepsTransform: The tensor reps transform class.
        """
        return ShiftRepsTransform(self, use_parallel, avoid_einsum)


class ShiftRepsTransform(TensorRepsTransform):
    """A module for transforming tensor representations.

    Args:
        tensor_reps (TensorReps): The tensor representations to be transformed.
        use_parallel (bool, optional): Whether to use parallel computation for the transformation. Defaults to True.
        avoid_einsum (bool, optional): Whether to avoid using `torch.einsum` for the transformation. Defaults to False.
    """

    def __init__(
        self, tensor_reps: TensorReps, use_parallel: bool = True, avoid_einsum: bool = False
    ):
        """Initialize a shifting TensorReps transform object.

        Args:
            tensor_reps (TensorReps): The tensor representations.
            use_parallel (bool, optional): Whether to use parallel computation. Defaults to True.
            avoid_einsum (bool, optional): Whether to avoid using einsum. Defaults to False.
        """
        super().__init__(tensor_reps, use_parallel, avoid_einsum)

    def transform_coeffs_parallel(
        self,
        coeffs: Tensor,
        basis_change: LFrames,
        avoid_einsum: bool = False,
        inplace: bool = False,
    ) -> Tensor:
        """Transforms the coefficients using more parallel computation.

        Args:
            coeffs (Tensor): The input coefficients to be transformed. Of shape `(N, dim)`, where `N` is the batch size and `dim` is the total dimension of the tensor reps.
            basis_change (LFrames): The basis change object representing the transformation. With matrices attribute of shape `(N, 3, 3)`.
            avoid_einsum (bool, optional): Whether to avoid using einsum for the transformation. Defaults to False.
            inplace (bool, optional): Whether to perform the transformation inplace. Defaults to False.

        Returns:
            Tensor: The transformed coefficients.
        """

        if self.tensor_reps.dim == 0:
            return coeffs

        if isinstance(basis_change, torch.Tensor):
            basis_change = LFrames(basis_change)

        N = coeffs.shape[0]

        # append the shift to the basis change matrices
        trafo_matrices = torch.zeros(
            basis_change.matrices.shape[0],
            basis_change.matrices.shape[1] + 1,
            basis_change.matrices.shape[2] + 1,
        )
        trafo_matrices[:, :-1, :-1] = basis_change.matrices
        trafo_matrices[:, -1, -1] = 1.0

        assert basis_change.shift is not None, "Shift must be provided for shift reps trafo"
        trafo_matrices[:, -1, :-1] = basis_change.local_shift

        trafo_matrices_t = trafo_matrices.transpose(-1, -2)

        if inplace:
            output_coeffs = coeffs
        else:
            output_coeffs = coeffs.clone()

        largest_tensor = torch.tensor([], device=coeffs.device)
        for i, l in enumerate(self.sorted_n):
            if self.is_sorted:
                start_idx, end_idx = self.start_end_indices[i]
                smaller_tensor = coeffs[:, start_idx:end_idx].view(
                    N, -1, *(l * (self.spatial_dim + 1,))
                )
            else:
                n_mask = self.n_masks[i]
                smaller_tensor = coeffs[:, n_mask].view(N, -1, *(l * (self.spatial_dim + 1,)))

            if i == 0:
                # highest n
                largest_tensor = smaller_tensor
            else:
                largest_tensor = torch.cat([smaller_tensor, largest_tensor], dim=1)

            if avoid_einsum:
                # apply transformation at axis 2 (0 is batch, 1 is channel)
                # move the axis 2 to the last index transform it and move it back
                # this is to avoid einsum (maybe there is a better way)
                largest_tensor = largest_tensor.moveaxis(2, -1)
                largest_shape = largest_tensor.shape

                # overwrite the last entry of the last dimension with the shift
                largest_tensor = largest_tensor.reshape(N, -1, self.spatial_dim + 1)
                additional_scalar = largest_tensor[:, :, -1].clone()
                largest_tensor[:, :, -1] = 1.0

                largest_tensor = torch.matmul(largest_tensor, trafo_matrices_t)

                # reinsert again the additional scalar
                largest_tensor[:, :, -1] = additional_scalar

                largest_tensor = largest_tensor.reshape(*largest_shape)
                largest_tensor = largest_tensor.moveaxis(-1, 2)
            else:
                print(trafo_matrices.shape, largest_tensor.shape)

                additional_scalar = largest_tensor[..., -1].clone()
                largest_tensor[..., -1] = 1.0

                largest_tensor = torch.einsum("ijk,ilk...->ilj...", trafo_matrices, largest_tensor)

                # reinsert again the additional scalar
                largest_tensor[..., -1] = additional_scalar

            # no need to transform again along this axis so flatten it for now into channels:
            largest_tensor = largest_tensor.flatten(start_dim=1, end_dim=2)

        # all computations are now done in largest_tensor and have to be unpacked into coeffs:
        if self.is_sorted:
            output_coeffs[:, self.scalar_dim :] = largest_tensor
        else:
            n_mask_rev = self.n_masks[::-1]
            n_muls_rev = self.n_muls[::-1]
            for i, n in enumerate(self.sorted_n[::-1]):
                if n == 0:
                    # scalars
                    continue

                # this could be faster if things where sorted. then largest would just be coeffs
                n_mask = n_mask_rev[i]
                l_mul = n_muls_rev[i]
                smaller_tensor = largest_tensor[:, : l_mul * 3**n]
                output_coeffs[:, n_mask] = smaller_tensor
                largest_tensor = largest_tensor[:, l_mul * 3**n :]

        # # apply parity:
        # # get the determinants of the rotation matrices:
        # is_det_neg = basis_change.det < 0
        # output_coeffs[is_det_neg] = output_coeffs[is_det_neg] * self.pseudo_tensor

        return output_coeffs

    def transform_coeffs(self, coeffs: Tensor, basis_change: LFrames) -> Tensor:
        """Transforms the coefficients using less parallel computation.

        Args:
            coeffs (Tensor): The input coefficients to be transformed. Of shape `(N, dim)`, where `N` is the batch size and `dim` is the total dimension of the tensor reps.
            basis_change (LFrames): The basis change object representing the transformation. With matrices attribute of shape `(N, 3, 3)`.

        Returns:
            Tensor: The transformed coefficients.
        """

        # append the shift to the basis change matrices
        trafo_matrices = torch.zeros(
            basis_change.matrices.shape[0],
            basis_change.matrices.shape[1] + 1,
            basis_change.matrices.shape[2] + 1,
        )
        trafo_matrices[:, :-1, :-1] = basis_change.matrices
        trafo_matrices[:, -1, -1] = 1.0
        trafo_matrices[:, -1, :-1] = basis_change.local_shift

        current_index = 0
        length = 0

        output = torch.zeros_like(coeffs)

        for mul_reps in self.tensor_reps:
            mul, rep = mul_reps

            rep_n = rep.order

            length = mul_reps.dim

            left_index = current_index
            right_index = current_index + length

            if rep_n == 0:
                if rep.p == -1:
                    det_sign = basis_change.det.sign()
                    output[:, left_index:right_index] = torch.einsum(
                        "i,ij->ij", det_sign, coeffs[:, left_index:right_index]
                    )
                else:
                    output[:, left_index:right_index] = coeffs[:, left_index:right_index]
                current_index += length
                continue

            einsum_str = self._get_einsum_string(rep_n)

            tensor = coeffs[:, left_index:right_index].reshape(
                coeffs.shape[0], mul, *([self.spatial_dim + 1] * rep_n)
            )

            # overwrite the last entry of the last dimension with the shift
            additional_scalar = tensor[:, :, *([-1] * rep_n)].clone()
            tensor[:, :, *([-1] * rep_n)] = 1.0

            trafo_tensors = torch.einsum(einsum_str, *([trafo_matrices] * rep_n), tensor)

            # reinsert again the additional scalar
            trafo_tensors[:, :, *([-1] * rep_n)] = additional_scalar

            output[:, left_index:right_index] = trafo_tensors.flatten(start_dim=1)

            if rep.p == -1 and False:
                # Do pseudo points make any sense? For now I decide no.
                assert False, "Pseudo points are not implemented yet."
                det_sign = basis_change.det.sign()
                output[:, left_index:right_index] = torch.einsum(
                    "i,ij->ij", det_sign, trafo_tensors.flatten(start_dim=1)
                )
            else:
                output[:, left_index:right_index] = trafo_tensors.flatten(start_dim=1)

            current_index += length

        return output

    def forward(
        self,
        coeffs: Tensor | None,
        basis_change: LFrames,
        inplace: bool = False,
        shift: Tensor | None = None,
    ) -> Tensor:
        """Applies the forward transformation to the input coefficients.

        Args:
            coeffs (Tensor): The input coefficients to be transformed. Of shape `(N, dim)`, where `N` is the batch size and `dim` is the total dimension of the tensor reps.
            basis_change (LFrames): The basis change object representing the transformation. With matrices attribute of shape `(N, 3, 3)`.
            inplace (bool, optional): Whether to perform the transformation inplace. Only relevant for parallel trafo. Defaults to False.

        Returns:
            Tensor: The transformed coefficients.
        """

        if coeffs is None:
            assert self.tensor_reps.dim == 0, "No coeffs are provided for non-trivial transform"
            return None

        if self.use_parallel:
            return self.transform_coeffs_parallel(
                coeffs, basis_change, self.avoid_einsum, inplace=inplace
            )
        else:
            return self.transform_coeffs(coeffs, basis_change, shift=shift)
