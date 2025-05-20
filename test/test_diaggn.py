"""Tests for the diagonal Gauss-Newton Hessian approximation."""

import pytest
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from curvlinops.diaggn import DiagGNLinearOperator, DiagGNInverseLinearOperator, FisherType


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


@pytest.fixture
def setup_model_and_data():
    """Setup a simple model and data for testing."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a simple model
    model = SimpleModel()
    
    # Generate some random data
    batch_size = 8
    x = torch.randn(batch_size, 5)
    y = torch.randint(0, 2, (batch_size,))
    
    # Create DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Loss function
    loss_func = nn.CrossEntropyLoss()
    
    return model, dataloader, loss_func


def test_diaggn_linear_operator(setup_model_and_data):
    """Test the DiagGNLinearOperator."""
    model, dataloader, loss_func = setup_model_and_data
    params = list(model.parameters())
    
    # Create the DiagGN linear operator
    diaggn = DiagGNLinearOperator(
        model_func=model,
        loss_func=loss_func,
        params=params,
        data=dataloader,
        fisher_type=FisherType.EMPIRICAL,
    )
    
    # Test compute_diagonal_elements
    diaggn.compute_diagonal_elements()
    assert len(diaggn._diag_elements) == len(params)
    
    # Test shape properties
    assert diaggn.shape[0] == diaggn.shape[1]  # Square matrix
    assert diaggn.shape[0] == sum(p.numel() for p in params)  # Correct size
    
    # Test trace
    trace = diaggn.trace
    assert trace > 0
    
    # Test matmat operation
    # Create a random vector in parameter space
    v = [torch.randn_like(p) for p in params]
    
    # Apply the operator
    Av = diaggn @ v
    
    # Check shapes
    assert len(Av) == len(v)
    for i in range(len(v)):
        assert Av[i].shape == v[i].shape
        
    # Test element-wise multiplication property of diagonal matrices
    for i, (a, diag) in enumerate(zip(Av, [diaggn._diag_elements[i] for i in range(len(params))])):
        torch.testing.assert_close(a, diag * v[i])
    
    # Test device and dtype inference
    assert diaggn._infer_device() == params[0].device
    assert diaggn._infer_dtype() == params[0].dtype


def test_diaggn_inverse_linear_operator(setup_model_and_data):
    """Test the DiagGNInverseLinearOperator."""
    model, dataloader, loss_func = setup_model_and_data
    params = list(model.parameters())
    
    # Create the DiagGN linear operator
    diaggn = DiagGNLinearOperator(
        model_func=model,
        loss_func=loss_func,
        params=params,
        data=dataloader,
        fisher_type=FisherType.EMPIRICAL,
    )
    
    # Create the inverse with damping
    damping = 1e-3
    diaggn_inv = DiagGNInverseLinearOperator(
        diag_gn_linop=diaggn,
        damping=damping,
        damping_type="add",
    )
    
    # Test compute_inverse_diag_elements
    assert len(diaggn_inv._inverse_diag_elements) == len(params)
    
    # Verify shape properties
    assert diaggn_inv.shape == diaggn.shape
    
    # Verify inverse property with damping
    v = [torch.randn_like(p) for p in params]
    Av = diaggn @ v
    A_inv_Av = diaggn_inv @ Av
    
    # Check if A_inv_Av ≈ v when damping is small
    if damping < 1e-5:
        for i in range(len(v)):
            # This should be approximately true: A_inv_Av[i] ≈ v[i]
            torch.testing.assert_close(A_inv_Av[i], v[i], rtol=1e-4, atol=1e-4)
    
    # Test update_damping
    new_damping = 0.1
    diaggn_inv.update_damping(new_damping)
    assert diaggn_inv._damping == new_damping
    
    # Test damping types
    diaggn_inv_scale = DiagGNInverseLinearOperator(
        diag_gn_linop=diaggn,
        damping=damping,
        damping_type="scale",
    )
    assert diaggn_inv_scale._damping_type == "scale"
    
    # Apply both operators to v
    Av_add = diaggn_inv @ v
    Av_scale = diaggn_inv_scale @ v
    
    # They should be different
    for i in range(len(v)):
        assert not torch.allclose(Av_add[i], Av_scale[i])
    
    # Test device and dtype inheritance 
    assert diaggn_inv._infer_device() == diaggn._infer_device()
    assert diaggn_inv._infer_dtype() == diaggn._infer_dtype()


def test_diaggn_types(setup_model_and_data):
    """Test different GN types."""
    model, dataloader, loss_func = setup_model_and_data
    params = list(model.parameters())
    
    # Test EMPIRICAL type
    diaggn_empirical = DiagGNLinearOperator(
        model_func=model,
        loss_func=loss_func,
        params=params,
        data=dataloader,
        fisher_type=FisherType.EMPIRICAL,
    )
    diaggn_empirical.compute_diagonal_elements()
    
    # Test MC type
    diaggn_mc = DiagGNLinearOperator(
        model_func=model,
        loss_func=loss_func,
        params=params,
        data=dataloader,
        fisher_type=FisherType.MC,
        mc_samples=2,
    )
    diaggn_mc.compute_diagonal_elements()
    
    # Test TYPE2 type
    diaggn_type2 = DiagGNLinearOperator(
        model_func=model,
        loss_func=loss_func,
        params=params,
        data=dataloader,
        fisher_type=FisherType.TYPE2,
    )
    diaggn_type2.compute_diagonal_elements()
    
    # They should all have diagonal elements
    assert len(diaggn_empirical._diag_elements) > 0
    assert len(diaggn_mc._diag_elements) > 0
    assert len(diaggn_type2._diag_elements) > 0


def test_pytorch_interface_compatibility(setup_model_and_data):
    """Test compatibility with PyTorchLinearOperator interface."""
    model, dataloader, loss_func = setup_model_and_data
    params = list(model.parameters())
    
    # Create the DiagGN linear operator
    diaggn = DiagGNLinearOperator(
        model_func=model,
        loss_func=loss_func,
        params=params,
        data=dataloader,
        fisher_type=FisherType.EMPIRICAL,
    )
    
    # Test flat tensor input/output
    param_size = sum(p.numel() for p in params)
    flat_vector = torch.randn(param_size)
    
    # Matrix-vector product
    result = diaggn @ flat_vector
    
    # Check result shape
    assert result.shape == flat_vector.shape
    
    # Test batched operation
    batch_size = 3
    flat_vectors = torch.randn(param_size, batch_size)
    result_batch = diaggn @ flat_vectors
    
    # Check batched result shape
    assert result_batch.shape == (param_size, batch_size)
    
    # Verify the operation is identical to applying to individual vectors
    for i in range(batch_size):
        single_result = diaggn @ flat_vectors[:, i]
        torch.testing.assert_close(result_batch[:, i], single_result) 