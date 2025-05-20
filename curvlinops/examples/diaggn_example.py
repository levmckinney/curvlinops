"""Example usage of the diagonal Gauss-Newton Hessian approximation.

This example demonstrates how to use the DiagGNLinearOperator and 
DiagGNInverseLinearOperator to compute the diagonal approximation of the 
Gauss-Newton Hessian and its inverse for a simple neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from curvlinops import DiagGNLinearOperator, DiagGNInverseLinearOperator, GNType


class SimpleModel(nn.Module):
    """Simple feedforward neural network for demonstration."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


def create_synthetic_data(n_samples=100, input_dim=10, n_classes=2):
    """Create synthetic data for demonstration."""
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return x, y


def main():
    """Run the DiagGN example."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and data
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=2)
    model.to(device)
    
    # Generate synthetic data
    x, y = create_synthetic_data(n_samples=100, input_dim=10, n_classes=2)
    x, y = x.to(device), y.to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Define loss function
    loss_func = nn.CrossEntropyLoss()
    
    # Pre-train the model a bit
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("Pre-training the model...")
    for epoch in range(5):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_func(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    # Get model parameters
    params = list(model.parameters())
    
    # Create the DiagGN linear operator
    print("\nCreating DiagGN linear operator...")
    diaggn = DiagGNLinearOperator(
        model_func=model,
        loss_func=loss_func,
        params=params,
        data=dataloader,
        progressbar=True,
        gn_type=GNType.MC,  # Monte Carlo approximation
        mc_samples=5,       # Use 5 samples per data point
    )
    
    # Print shape information about the linear operator
    total_params = sum(p.numel() for p in params)
    print(f"DiagGN shape: {diaggn.shape}")
    print(f"Total parameter count: {total_params}")
    
    # Compute diagonal elements (this happens automatically on first use, but we make it explicit)
    diaggn.compute_diagonal_elements()
    
    # Print some stats about the diagonal elements
    print("\nDiagonal GN Statistics:")
    for i, param in enumerate(params):
        diag = diaggn._diag_elements[i]
        print(f"Parameter {i}, Shape: {param.shape}, "
              f"Diagonal Min: {diag.min().item():.6f}, "
              f"Max: {diag.max().item():.6f}, "
              f"Mean: {diag.mean().item():.6f}")
    
    # Compute trace and Frobenius norm
    print(f"\nTrace: {diaggn.trace.item():.6f}")
    print(f"Frobenius norm: {diaggn.frobenius_norm.item():.6f}")
    
    # Demonstrate both tensor list and flat tensor interfaces
    
    # 1. Tensor list interface (parameter by parameter)
    print("\nUsing tensor list interface:")
    v_list = [torch.randn_like(p) for p in params]
    Av_list = diaggn @ v_list
    
    print(f"Input shapes: {[v.shape for v in v_list]}")
    print(f"Output shapes: {[av.shape for av in Av_list]}")
    
    # 2. Flat tensor interface (all parameters flattened)
    print("\nUsing flat tensor interface:")
    flat_v = torch.randn(total_params, device=device)
    flat_Av = diaggn @ flat_v
    
    print(f"Input shape: {flat_v.shape}")
    print(f"Output shape: {flat_Av.shape}")
    
    # Create the inverse with damping
    print("\nCreating DiagGN inverse linear operator with damping...")
    damping = 1e-4
    diaggn_inv = DiagGNInverseLinearOperator(
        diag_gn_linop=diaggn,
        damping=damping,
        damping_type="add",  # Add damping to diagonal (D + λI)^-1
    )
    
    # Multiply the inverse with the result from the previous step
    A_inv_Av_list = diaggn_inv @ Av_list
    
    # Check if A_inv_Av ≈ v (this should be close to true when damping is small)
    total_error = sum((A_inv_A_v - v_i).norm() for A_inv_A_v, v_i in zip(A_inv_Av_list, v_list))
    print(f"Total error ||A^-1 A v - v||: {total_error.item():.6f}")
    
    # Demonstrate updating the damping parameter
    print("\nUpdating damping parameter:")
    for new_damping in [0.01, 0.1, 1.0]:
        diaggn_inv.update_damping(new_damping)
        A_inv_Av_list = diaggn_inv @ Av_list
        total_error = sum((A_inv_A_v - v_i).norm() for A_inv_A_v, v_i in zip(A_inv_Av_list, v_list))
        print(f"Damping: {new_damping:.4f}, Total error: {total_error.item():.6f}")
    
    # Demonstrate using the scaled damping type
    print("\nUsing scaled damping type:")
    diaggn_inv_scale = DiagGNInverseLinearOperator(
        diag_gn_linop=diaggn,
        damping=0.1,  # 10% identity, 90% diagonal
        damping_type="scale",  # ((1-λ)D + λI)^-1
    )
    A_inv_Av_scale = diaggn_inv_scale @ Av_list
    total_error_scale = sum((A_inv_A_v - v_i).norm() for A_inv_A_v, v_i in zip(A_inv_Av_scale, v_list))
    print(f"Total error with scaled damping: {total_error_scale.item():.6f}")


if __name__ == "__main__":
    main() 