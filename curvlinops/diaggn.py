"""Linear operator for diagonal approximation of the Gauss-Newton Hessian.

The diagonal GN approximation provides a computationally efficient alternative to 
more complex approaches like KFAC, by using only the diagonal elements of the 
Gauss-Newton matrix. This significantly reduces computation and memory requirements 
at the cost of some approximation quality.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from enum import EnumMeta
from math import sqrt
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from einops import rearrange
from torch import Generator, Tensor, device, dtype, zeros_like, allclose
from torch.autograd import grad
from torch.nn import (
    MSELoss,
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    Module,
    Parameter,
)
from tqdm import tqdm

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.kfac_utils import loss_hessian_matrix_sqrt
from curvlinops.kfac import FisherType


class MetaEnum(EnumMeta):
    """Metaclass for the Enum class for desired behavior of the `in` operator."""

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True



class DiagGNLinearOperator(PyTorchLinearOperator):
    """Linear operator for diagonal Gauss-Newton Hessian approximation.

    This operator stores only the diagonal elements of the GN Hessian, which
    significantly reduces computational and memory requirements compared to
    full-matrix or Kronecker-factored approximations.

    Attributes:
        _SUPPORTED_LOSSES: Tuple of supported loss functions.
        _SUPPORTED_GN_TYPE: Enum of supported GN approximation types.
        SELF_ADJOINT: Whether the operator is self-adjoint. `True` for diagonal matrices.
    """

    _SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)
    _SUPPORTED_FISHER_TYPE: FisherType = FisherType
    SELF_ADJOINT: bool = True  # Diagonal matrices are self-adjoint

    def __init__(
        self,
        model_func: Module,
        loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        seed: int = 2147483647,
        fisher_type: str = FisherType.MC,
        mc_samples: int = 1,
        num_per_example_loss_terms: Optional[int] = None,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
        matrix_dtype: Optional[dtype] = None,
        matrix_device: Optional[device] = None,
    ):
        """Initialize the diagonal Gauss-Newton approximation.

        Args:
            model_func: The neural network.
            loss_func: The loss function.
            params: Parameters to compute the GN Hessian for.
            data: Data loader for the GN Hessian computation.
            progressbar: Whether to show a progress bar. Defaults to ``False``.
            check_deterministic: Whether to check model determinism. Defaults to ``True``.
            seed: Random seed for MC sampling. Defaults to ``2147483647``.
            fisher_type: Type of Fisher approximation to use. Defaults to ``FisherType.MC``.
            mc_samples: Number of MC samples per data point. Defaults to ``1``.
            num_per_example_loss_terms: Number of per-example loss terms, e.g., the
                number of tokens in a sequence. If ``None``, inferred from data.
            num_data: Optional number of data points. If None, will be inferred.
            batch_size_fn: Function to get batch size from inputs.
            matrix_dtype: The dtype of the diagonal elements. Defaults to the dtype of
                the models parameters.
            matrix_device: The device of the diagonal elements. Defaults to the device
                of the model's parameters.
        """
        if not isinstance(loss_func, self._SUPPORTED_LOSSES):
            raise ValueError(
                f"Invalid loss: {loss_func}. Supported: {self._SUPPORTED_LOSSES}."
            )
        if fisher_type not in self._SUPPORTED_FISHER_TYPE:
            raise ValueError(
                f"Invalid fisher_type: {fisher_type}. "
                f"Supported: {self._SUPPORTED_FISHER_TYPE}."
            )
        if fisher_type != FisherType.MC and mc_samples != 1:
            raise ValueError(
                f"Invalid mc_samples: {mc_samples}. "
                "Only mc_samples=1 is supported for `fisher_type != FisherType.MC`."
            )

        # Initialize with shapes derived from params
        in_shape = [tuple(p.shape) for p in params]
        out_shape = [tuple(p.shape) for p in params]
        super().__init__(in_shape, out_shape)

        self._params = params
        self._model_func = model_func
        self._loss_func = loss_func
        self._data = data
        self._progressbar = progressbar
        self._device = self._infer_device()
        self._batch_size_fn = (
            (lambda X: X.shape[0]) if batch_size_fn is None else batch_size_fn
        )
        self._seed = seed
        self._generator: Union[None, Generator] = None
        self._fisher_type = fisher_type
        self._mc_samples = mc_samples
        self._diag_elements: Dict[int, Tensor] = {}  # Will store diagonal elements for each parameter
        self._matrix_dtype = matrix_dtype
        self._matrix_device = matrix_device

        # Determine the number of data points
        self._N_data = (
            sum(
                self._batch_size_fn(X)
                for (X, _) in self._loop_over_data(desc="_N_data")
            )
            if num_data is None
            else num_data
        )
        
        # Properties of the diagonal approximation
        self._trace = None
        self._det = None
        self._logdet = None
        self._frobenius_norm = None
        
        self._set_num_per_example_loss_terms(num_per_example_loss_terms)
        
        if check_deterministic:
            self._check_deterministic()

    def _set_num_per_example_loss_terms(
        self, num_per_example_loss_terms: Optional[int]
    ):
        """Set the number of per-example loss terms.

        Args:
            num_per_example_loss_terms: Number of per-example loss terms. If ``None``,
                it is inferred from the data at the cost of one traversal through the
                data loader.

        Raises:
            ValueError: If the number of loss terms is not divisible by the number of
                data points.
        """
        if num_per_example_loss_terms is None:
            # Determine the number of per-example loss terms
            num_loss_terms = sum(
                (
                    y.numel()
                    if isinstance(self._loss_func, CrossEntropyLoss)
                    else y.shape[:-1].numel()
                )
                for (_, y) in self._loop_over_data(desc="_num_per_example_loss_terms")
            )
            if num_loss_terms % self._N_data != 0:
                raise ValueError(
                    "The number of loss terms must be divisible by the number of data "
                    f"points; num_loss_terms={num_loss_terms}, N_data={self._N_data}."
                )
            self._num_per_example_loss_terms = num_loss_terms // self._N_data
        else:
            self._num_per_example_loss_terms = num_per_example_loss_terms

    def _reset_matrix_properties(self):
        """Reset matrix properties."""
        self._trace = None
        self._det = None
        self._logdet = None
        self._frobenius_norm = None

    def compute_diagonal_elements(self):
        """Compute and cache diagonal elements of the GN approximation."""
        self._reset_matrix_properties()
        
        # Initialize diagonal elements for each parameter
        for i, p in enumerate(self._params):
            self._diag_elements[i] = zeros_like(p, dtype=self._matrix_dtype, device=self._matrix_device)
        
        # Initialize generator for MC sampling
        if self._generator is None or self._generator.device != self._device:
            self._generator = Generator(device=self._device)
        self._generator.manual_seed(self._seed)
        
        # Loop over data
        for X, y in self._loop_over_data(desc="DiagGN matrices"):
            output = self._model_func(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_diagonal_elements_batch(output, y)

    def _loop_over_data(
        self, desc: Optional[str] = None, add_device_to_desc: bool = True
    ) -> Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Args:
            desc: Description for the progress bar. Will be ignored if progressbar is
                disabled.
            add_device_to_desc: Whether to add the device to the description.
                Default: ``True``.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = self._data

        if self._progressbar:
            desc = f"{self.__class__.__name__}{'' if desc is None else f'.{desc}'}"
            if add_device_to_desc:
                desc = f"{desc} (on {str(self._device)})"
            data_iter = tqdm(data_iter, desc=desc)

        for X, y in data_iter:
            # Assume everything is handled by the model
            # if `X` is a custom data format
            if isinstance(X, Tensor):
                X = X.to(self._device)
            y = y.to(self._device)
            yield (X, y)

    def _rearrange_for_larger_than_2d_output(
        self, output: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Rearrange the output and target if output is >2d.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.
            y: The labels :math:`\{\mathbf{y}_n\}_{n=1}^N`.

        Returns:
            The rearranged output and target.
        """
        if isinstance(self._loss_func, CrossEntropyLoss):
            output = rearrange(output, "batch c ... -> (batch ...) c")
            y = rearrange(y, "batch ... -> (batch ...)")
        else:
            output = rearrange(output, "batch ... c -> (batch ...) c")
            y = rearrange(y, "batch ... c -> (batch ...) c")
        return output, y

    def _maybe_adjust_loss_scale(self, loss: Tensor, output: Tensor) -> Tensor:
        """Adjust the scale of the loss tensor if necessary.

        The ``BCEWithLogitsLoss`` and ``MSELoss`` also average over the output dimension
        in addition to the batch dimension. We adjust the scale of the loss to correct
        for this.

        Args:
            loss: The loss tensor to adjust.
            output: The model's output.

        Returns:
            The scaled loss tensor.
        """
        if (
            isinstance(self._loss_func, (BCEWithLogitsLoss, MSELoss))
            and self._loss_func.reduction == "mean"
        ):
            # ``BCEWithLogitsLoss`` and ``MSELoss`` also average over non-batch
            # dimensions. We have to scale the loss to incorporate this scaling.
            _, C = output.shape
            loss *= sqrt(C)
        return loss

    def _compute_diagonal_elements_batch(self, output: Tensor, y: Tensor):
        """Compute the diagonal elements for the current batch.

        Args:
            output: The model's prediction.
            y: The true labels.
        """
        if output.ndim != 2 or y.ndim not in {1, 2}:
            raise ValueError(
                "Only 2d output and 1d/2d target are supported. "
                f"Got {output.ndim=} and {y.ndim=}."
            )

        if self._fisher_type == FisherType.TYPE2:
            # Compute per-sample Hessian square root, then concatenate over samples.
            # Result has shape `(batch_size, num_classes, num_classes)`
            hessian_sqrts = []
            for out, target in zip(output.split(1), y.split(1)):
                hessian_sqrt = loss_hessian_matrix_sqrt(out.detach(), target, self._loss_func)
                hessian_sqrts.append(hessian_sqrt)

            # Fix scaling caused by the batch dimension
            num_loss_terms = output.shape[0]
            reduction = self._loss_func.reduction
            scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[reduction]
            
            # For each column of the matrix square root we need to backpropagate,
            # but we can iterate over the output dimension one by one
            num_cols = hessian_sqrts[0].shape[1]
            for c in range(num_cols):
                # Process each column of the Hessian square root for all samples
                batched_columns = [
                    sqrt.select(-1, c) * scale for sqrt in hessian_sqrts
                ]
                
                # Compute the gradient for this column across all samples
                for sample_idx, batched_column in enumerate(batched_columns):
                    out = output[sample_idx:sample_idx+1]
                    
                    # For each parameter, compute gradient and square it (for diagonal)
                    grads = grad(
                        (out * batched_column).sum(),
                        self._params,
                        retain_graph=True,
                    )
                    
                    # Accumulate squared gradients to diagonal elements
                    for i, g in enumerate(grads):
                        if g is not None:
                            self._diag_elements[i].add_(g.square())

        elif self._fisher_type == FisherType.MC:
            for mc in range(self._mc_samples):
                y_sampled = self.draw_label(output)
                loss = self._loss_func(output, y_sampled)
                loss = self._maybe_adjust_loss_scale(loss, output)
                
                # Compute gradients
                grads = grad(loss, self._params, retain_graph=mc != self._mc_samples - 1)
                
                # Accumulate squared gradients to diagonal elements
                for i, g in enumerate(grads):
                    if g is not None:
                        # Scale by 1/mc_samples for proper averaging
                        self._diag_elements[i].add_(g.square() / self._mc_samples)

        elif self._fisher_type == FisherType.EMPIRICAL:
            loss = self._loss_func(output, y)
            loss = self._maybe_adjust_loss_scale(loss, output)
            
            # Compute gradients
            grads = grad(loss, self._params)
            
            # Accumulate squared gradients to diagonal elements
            for i, g in enumerate(grads):
                if g is not None:
                    self._diag_elements[i].add_(g.square())
                
        else:
            raise ValueError(
                f"Invalid fisher_type: {self._fisher_type}. "
                + f"Supported: {self._SUPPORTED_FISHER_TYPE}."
            )

    def _get_normalization_factor(
        self, X: Union[MutableMapping, Tensor], y: Tensor
    ) -> float:
        """Return the correction factor for correct normalization over the data set.

        Args:
            X: Input to the DNN.
            y: Ground truth.

        Returns:
            Normalization factor
        """
        return {"sum": 1.0, "mean": self._batch_size_fn(X) / self._N_data}[
            self._loss_func.reduction
        ]

    def draw_label(self, output: Tensor) -> Tensor:
        r"""Draw a sample from the model's predictive distribution.

        The model's distribution is implied by the (negative log likelihood) loss
        function. For instance, ``MSELoss`` implies a Gaussian distribution with
        constant variance, and ``CrossEntropyLoss`` implies a categorical distribution.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.

        Returns:
            A sample
            :math:`\{\mathbf{y}_n\}_{n=1}^N` drawn from the model's predictive
            distribution :math:`p(\mathbf{y} \mid \mathbf{x}, \mathbf{\theta})`. Has
            the same shape as the labels that would be fed into the loss function
            together with ``output``.

        Raises:
            ValueError: If the output is not 2d.
            NotImplementedError: If the loss function is not supported.
        """
        if output.ndim != 2:
            raise ValueError("Only a 2d output is supported.")

        if isinstance(self._loss_func, MSELoss):
            std = sqrt(0.5)
            perturbation = std * Tensor(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator,
            ).normal_()
            return output.clone().detach() + perturbation

        elif isinstance(self._loss_func, CrossEntropyLoss):
            probs = output.softmax(dim=1)
            labels = probs.multinomial(
                num_samples=1, generator=self._generator
            ).squeeze(-1)
            return labels

        elif isinstance(self._loss_func, BCEWithLogitsLoss):
            probs = output.sigmoid()
            labels = probs.bernoulli(generator=self._generator)
            return labels

        else:
            raise NotImplementedError

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            M: Matrix for multiplication in tensor list format. Assume the linear
                operator's input tensor product space consists of shapes ``[*N1],
                [*N2], ...``. Then, ``M`` is a list of tensors with shapes
                ``[*N1, K], [*N2, K], ...`` with ``K`` the number of columns.

        Returns:
            Matrix-multiplication result in tensor list format.
        """
        # If diagonal elements not computed yet, compute them
        if not self._diag_elements:
            self.compute_diagonal_elements()
        
        # Element-wise multiplication of M with diagonal elements
        # Each element in M is multiplied by the corresponding diagonal element
        return [m * self._diag_elements[i][..., None] for i, m in enumerate(M)]

    def _check_deterministic(self):
        """Check that the linear operator is deterministic.

        Raises:
            RuntimeError: If non-deterministic behavior is detected.
        """
        # Create random vector
        v = [p.new_zeros(p.shape).normal_() for p in self._params]
        
        # Apply operator twice
        Av1 = self @ v
        Av2 = self @ v
        
        # Check if results are the same
        for av1, av2 in zip(Av1, Av2):
            if not allclose(av1, av2, rtol=1e-5, atol=1e-8):
                raise RuntimeError("DiagGN operator is not deterministic")

    @property
    def trace(self) -> Tensor:
        """Trace of the diagonal GN approximation.

        Will call ``compute_diagonal_elements`` if it has not been called before and
        will cache the trace until ``compute_diagonal_elements`` is called again.

        Returns:
            Trace of the diagonal GN approximation.
        """
        if self._trace is not None:
            return self._trace

        if not self._diag_elements:
            self.compute_diagonal_elements()

        self._trace = sum(diag.sum() for diag in self._diag_elements.values())
        return self._trace

    @property
    def det(self) -> Tensor:
        """Determinant of the diagonal GN approximation.

        Will call ``compute_diagonal_elements`` if it has not been called before and
        will cache the determinant until ``compute_diagonal_elements`` is called again.

        Returns:
            Determinant of the diagonal GN approximation.
        """
        if self._det is not None:
            return self._det

        if not self._diag_elements:
            self.compute_diagonal_elements()

        self._det = 1.0
        for diag in self._diag_elements.values():
            self._det *= diag.prod()
        return self._det

    @property
    def logdet(self) -> Tensor:
        """Log determinant of the diagonal GN approximation.

        More numerically stable than the ``det`` property.
        Will call ``compute_diagonal_elements`` if it has not been called before and
        will cache the log determinant until ``compute_diagonal_elements`` is called again.

        Returns:
            Log determinant of the diagonal GN approximation.
        """
        if self._logdet is not None:
            return self._logdet

        if not self._diag_elements:
            self.compute_diagonal_elements()

        self._logdet = 0.0
        for diag in self._diag_elements.values():
            self._logdet += diag.log().sum()
        return self._logdet

    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the diagonal GN approximation.

        Will call ``compute_diagonal_elements`` if it has not been called before and
        will cache the Frobenius norm until ``compute_diagonal_elements`` is called again.

        Returns:
            Frobenius norm of the diagonal GN approximation.
        """
        if self._frobenius_norm is not None:
            return self._frobenius_norm

        if not self._diag_elements:
            self.compute_diagonal_elements()

        self._frobenius_norm = sum(diag.square().sum() for diag in self._diag_elements.values()).sqrt()
        return self._frobenius_norm

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the DiagGN linear operator.

        Returns:
            State dictionary.
        """
        loss_type = {
            MSELoss: "MSELoss",
            CrossEntropyLoss: "CrossEntropyLoss",
            BCEWithLogitsLoss: "BCEWithLogitsLoss",
        }[type(self._loss_func)]
        return {
            # Model and loss function
            "model_func_state_dict": self._model_func.state_dict(),
            "loss_type": loss_type,
            "loss_reduction": self._loss_func.reduction,
            # Attributes
            "progressbar": self._progressbar,
            "seed": self._seed,
            "fisher_type": self._fisher_type,
            "mc_samples": self._mc_samples,
            "num_per_example_loss_terms": self._num_per_example_loss_terms,
            "num_data": self._N_data,
            # Diagonal elements (if computed)
            "diag_elements": self._diag_elements,
            # Properties (not necessarily computed)
            "trace": self._trace,
            "det": self._det,
            "logdet": self._logdet,
            "frobenius_norm": self._frobenius_norm,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state of the DiagGN linear operator.

        Args:
            state_dict: State dictionary.

        Raises:
            ValueError: If the loss function does not match the state dict.
            ValueError: If the loss function reduction does not match the state dict.
        """
        self._model_func.load_state_dict(state_dict["model_func_state_dict"])
        # Verify that the loss function and its reduction match the state dict
        loss_func_type = {
            "MSELoss": MSELoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
        }[state_dict["loss_type"]]
        if not isinstance(self._loss_func, loss_func_type):
            raise ValueError(
                f"Loss function mismatch: {loss_func_type} != {type(self._loss_func)}."
            )
        if state_dict["loss_reduction"] != self._loss_func.reduction:
            raise ValueError(
                "Loss function reduction mismatch: "
                f"{state_dict['loss_reduction']} != {self._loss_func.reduction}."
            )

        # Set attributes
        self._progressbar = state_dict["progressbar"]
        self._seed = state_dict["seed"]
        self._fisher_type = state_dict["fisher_type"]
        self._mc_samples = state_dict["mc_samples"]
        self._num_per_example_loss_terms = state_dict["num_per_example_loss_terms"]
        self._N_data = state_dict["num_data"]

        # Set diagonal elements (if computed)
        self._diag_elements = state_dict["diag_elements"]

        # Set properties (not necessarily computed)
        self._trace = state_dict["trace"]
        self._det = state_dict["det"]
        self._logdet = state_dict["logdet"]
        self._frobenius_norm = state_dict["frobenius_norm"]

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any],
        model_func: Module,
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        check_deterministic: bool = True,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
    ) -> DiagGNLinearOperator:
        """Load a DiagGN linear operator from a state dictionary.

        Args:
            state_dict: State dictionary.
            model_func: The model function.
            params: The model's parameters that DiagGN is computed for.
            data: A data loader containing the data for computation.
            check_deterministic: Whether to check that the linear operator is
                deterministic. Defaults to ``True``.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified.

        Returns:
            Linear operator of diagonal GN approximation.
        """
        loss_func = {
            "MSELoss": MSELoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
        }[state_dict["loss_type"]](reduction=state_dict["loss_reduction"])
        diag_gn = cls(
            model_func,
            loss_func,
            params,
            data,
            batch_size_fn=batch_size_fn,
            check_deterministic=False,
            progressbar=state_dict["progressbar"],
            seed=state_dict["seed"],
            fisher_type=state_dict["fisher_type"],
            mc_samples=state_dict["mc_samples"],
            num_per_example_loss_terms=state_dict["num_per_example_loss_terms"],
            num_data=state_dict["num_data"],
        )
        diag_gn.load_state_dict(state_dict)

        # Potentially call `check_deterministic` after the state dict is loaded
        if check_deterministic:
            diag_gn._check_deterministic()

        return diag_gn

    def _infer_device(self) -> device:
        """Infer the device of the parameters.

        Returns:
            The device on which the parameters reside.

        Raises:
            RuntimeError: If parameters are on different devices.
        """
        devices = {p.device for p in self._params}
        if len(devices) != 1:
            raise RuntimeError(f"Parameters on different devices: {devices}")
        return devices.pop()

    def _infer_dtype(self) -> dtype:
        """Infer the data type of the parameters.

        Returns:
            The data type of the parameters.

        Raises:
            RuntimeError: If parameters have different data types.
        """
        dtypes = {p.dtype for p in self._params}
        if len(dtypes) != 1:
            raise RuntimeError(f"Parameters have different data types: {dtypes}")
        return dtypes.pop()


class DiagGNInverseLinearOperator(PyTorchLinearOperator):
    """Inverse linear operator for the diagonal Gauss-Newton Hessian approximation.
    
    This implements the inverse of the DiagGNLinearOperator with optional damping.
    The inverse of a diagonal matrix is simply the reciprocal of each diagonal element,
    with damping added for numerical stability.
    
    Attributes:
        SELF_ADJOINT: Whether the operator is self-adjoint. ``True`` for diagonal matrices.
    """
    
    SELF_ADJOINT: bool = True  # Diagonal matrices are self-adjoint
    
    def __init__(
        self,
        diag_gn_linop: DiagGNLinearOperator,
        damping: float = 0.0,
        damping_type: str = "add",
    ):
        """Initialize the inverse diagonal GN operator with damping.
        
        Args:
            diag_gn_linop: The diagonal GN linear operator to invert.
            damping: Damping parameter for numerical stability. Defaults to ``0.0``.
            damping_type: Type of damping to apply. Options are:
                - "add": Add damping to diagonal elements (D + λI)^-1
                - "scale": Scale diagonal elements and add damping ((1-λ)D + λI)^-1, 
                  where the diagonal scaling factor is (1-λ)
                Defaults to ``"add"``.
                
        Raises:
            ValueError: If ``damping_type`` is not one of the supported types.
            ValueError: If ``damping`` is not a positive float.
        """
        if damping_type not in ["add", "scale"]:
            raise ValueError(
                f"Invalid damping_type: {damping_type}. Supported: ['add', 'scale']."
            )
        if damping < 0.0:
            raise ValueError(f"Damping must be non-negative. Got: {damping}.")
        
        # Initialize with same shapes as the diag_gn_linop
        super().__init__(diag_gn_linop._in_shape, diag_gn_linop._out_shape)
        
        # Save references to attributes from the original operator
        self._diag_gn = diag_gn_linop
        self._params = diag_gn_linop._params
        self._damping = damping
        self._damping_type = damping_type
        
        # Compute the inverse diagonal elements
        self._inverse_diag_elements = {}
        self._compute_inverse_diag_elements()
    
    def _compute_inverse_diag_elements(self):
        """Compute the inverse of the diagonal elements with damping applied."""
        # Ensure diagonal elements are computed in the original operator
        if not self._diag_gn._diag_elements:
            self._diag_gn.compute_diagonal_elements()
        
        # Apply damping and compute inverse for each parameter
        for param_idx, diag in self._diag_gn._diag_elements.items():
            if self._damping_type == "add":
                # (D + λI)^-1
                self._inverse_diag_elements[param_idx] = 1.0 / (diag + self._damping)
            else:  # damping_type == "scale"
                # ((1-λ)D + λI)^-1
                identity = diag.new_ones(diag.shape)
                damped_diag = (1.0 - self._damping) * diag + self._damping * identity
                self._inverse_diag_elements[param_idx] = 1.0 / damped_diag
    
    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            M: List of tensors to multiply.

        Returns:
            Result of multiplication with the inverse diagonal GN.
        """
        # Element-wise multiplication of M with inverse diagonal elements
        return [m * self._inverse_diag_elements[i][..., None] for i, m in enumerate(M)]
    
    @property
    def trace(self) -> Tensor:
        """Trace of the inverse diagonal GN approximation.
        
        Returns:
            Trace of the inverse matrix.
        """
        return sum(diag.sum() for diag in self._inverse_diag_elements.values())
    
    @property
    def det(self) -> Tensor:
        """Determinant of the inverse diagonal GN approximation.
        
        Returns:
            Determinant of the inverse matrix.
        """
        result = 1.0
        for diag in self._inverse_diag_elements.values():
            result *= diag.prod()
        return result
    
    @property
    def logdet(self) -> Tensor:
        """Log determinant of the inverse diagonal GN approximation.
        
        Returns:
            Log determinant of the inverse matrix.
        """
        result = 0.0
        for diag in self._inverse_diag_elements.values():
            result += diag.log().sum()
        return result
    
    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the inverse diagonal GN approximation.
        
        Returns:
            Frobenius norm of the inverse matrix.
        """
        return sum(diag.square().sum() for diag in self._inverse_diag_elements.values()).sqrt()
    
    def update_damping(self, damping: float):
        """Update the damping parameter and recompute the inverse.
        
        Args:
            damping: New damping parameter value.
            
        Raises:
            ValueError: If damping is not a positive float.
        """
        if damping < 0.0:
            raise ValueError(f"Damping must be non-negative. Got: {damping}.")
        
        self._damping = damping
        self._compute_inverse_diag_elements()
        
    def _infer_device(self) -> device:
        """Infer the device of the operator.

        Returns:
            The device on which the parameters reside.
        """
        return self._diag_gn._infer_device()

    def _infer_dtype(self) -> dtype:
        """Infer the data type of the operator.

        Returns:
            The data type of the parameters.
        """
        return self._diag_gn._infer_dtype()
