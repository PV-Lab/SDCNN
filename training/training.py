# import
import os
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate seed for reproducible results
SEED = 99
np.random.seed(SEED)  # Seed NumPy's random number generator
random.seed(SEED)     # Seed Python's random module

# ==== USER-DEFINED VARIABLES ==== #
max_angle = 44.0  # Maximum angle achievable by the robot in the yaw-axis
px_to_mm = 0.117  # mm/pixel conversion factor
probe_stroke_mm = 4.5  # Width of the probe in millimeters
k_poses = 3  # Number of optimal poses to generate per material
augmented_random = joblib.load(os.getcwd() + '/data/augmented_dataset_fastSAM.pkl')  # Download dataset from: https://osf.io/download/c6xtg
# ================================ #
probe_stroke_pixels = probe_stroke_mm / px_to_mm  # Convert probe width to pixels

# ====== DEFINITIONS ====== #
def gaussian_kernel(size, sigma):
    """Create a 1D Gaussian kernel.

    Args:
        size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        Tensor: 1D Gaussian kernel.
    """
    x = torch.arange(size).float() - size // 2  # Coordinate grid centered at zero
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))  # Compute Gaussian values
    kernel_1d /= kernel_1d.sum()  # Normalize kernel
    return kernel_1d

def blur_tensor(tensor, kernel_size, sigma):
    """Apply Gaussian blur to a 2D tensor.

    Args:
        tensor (Tensor): 2D tensor to blur.
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        Tensor: Blurred 2D tensor.
    """
    kernel_1d = gaussian_kernel(kernel_size, sigma).unsqueeze(0)  # Create 1D Gaussian kernel
    kernel_2d = torch.mm(kernel_1d.t(), kernel_1d)  # Create 2D Gaussian kernel via outer product
    kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)  # Reshape to [1, 1, kernel_size, kernel_size]
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    blurred = F.conv2d(tensor, kernel_2d.to(device), padding=kernel_size // 2)  # Apply convolution
    blurred = blurred.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    return blurred

def soft_placement(x_coords, y_coords, width, height, sigma=0.1, threshold=0.5):
    """Create a 2D tensor with soft placements of contact points.

    Args:
        x_coords (Tensor): x-coordinates of contact points (batch_size, num_points).
        y_coords (Tensor): y-coordinates of contact points (batch_size, num_points).
        width (int): Width of the output tensor.
        height (int): Height of the output tensor.
        sigma (float, optional): Standard deviation for Gaussian peaks. Defaults to 0.1.
        threshold (float, optional): Threshold for soft thresholding. Defaults to 0.5.

    Returns:
        Tensor: 2D tensor with soft placements.
    """
    batch_size, num_points = x_coords.shape
    xx, yy = torch.meshgrid(torch.linspace(0, width - 1, width),
                            torch.linspace(0, height - 1, height),
                            indexing='xy')  # Create coordinate grid
    xx, yy = xx.to(x_coords.device), yy.to(y_coords.device)  # Ensure device compatibility
    xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1, -1)  # Expand grid dimensions
    yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1, -1)
    x_coords = x_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)  # Expand x coordinates
    y_coords = y_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)  # Expand y coordinates
    gaussian_peaks = torch.exp(-((xx - x_coords)**2 + (yy - y_coords)**2) / (2 * sigma**2)).to(device)  # Compute Gaussian peaks
    tensor_2d = torch.sum(gaussian_peaks, dim=1).to(device)  # Sum over points
    tensor_2d = torch.sigmoid((tensor_2d - threshold) * 10).to(device)  # Apply soft thresholding
    return tensor_2d

def differentiable_threshold_sum_batched(tensor, threshold=0.0255, steepness=100):
    """Sum tensor elements above a threshold in a differentiable manner for batched tensors.

    Uses a sigmoid function to weight elements based on their proximity to the threshold.
    The steepness parameter controls the sharpness of the thresholding.

    Args:
        tensor (Tensor): Batched tensor of shape [batch_size, ...].
        threshold (float, optional): Threshold value. Defaults to 0.0255.
        steepness (float, optional): Controls the sharpness of the thresholding. Defaults to 100.

    Returns:
        Tensor: Sum of elements above the threshold for each batch.
    """
    weights = torch.sigmoid(steepness * (tensor - threshold))  # Create weights for each element
    sum_above_threshold = torch.sum(tensor * weights, dim=list(range(1, tensor.dim())))  # Multiply weights and sum
    return sum_above_threshold

def create_contacts_differentiable_batched(mid_x, mid_y, rot_rad, probe_stroke_pixels):
    """Create contact points in a differentiable way for batch processing.

    Args:
        mid_x (Tensor): Midpoint x-coordinates (batch_size, num_segments).
        mid_y (Tensor): Midpoint y-coordinates (batch_size, num_segments).
        rot_rad (Tensor): Rotation angles in radians (batch_size, num_segments).
        probe_stroke_pixels (float): Length of the probe stroke in pixels.

    Returns:
        Tuple[Tensor, Tensor]: xranges and yranges tensors for contact points.
    """
    dx = (probe_stroke_pixels / 2) * torch.cos(rot_rad)  # Compute half-length offsets in x direction
    dy = (probe_stroke_pixels / 2) * torch.sin(rot_rad)  # Compute half-length offsets in y direction
    start_x = mid_x - dx  # Start x-coordinate of line segment
    start_y = mid_y - dy  # Start y-coordinate of line segment
    end_x = mid_x + dx  # End x-coordinate of line segment
    end_y = mid_y + dy  # End y-coordinate of line segment
    max_nsteps = torch.max(torch.abs(end_x - start_x)).int()  # Maximum number of steps needed
    max_xsteps = torch.max(torch.abs(start_x - end_x)).int()  # Maximum x-steps
    batch_size, num_segments = mid_x.size(0), mid_x.size(1)
    max_steps = torch.max(max_nsteps, max_xsteps)  # Maximum number of steps for interpolation
    yranges = torch.zeros(batch_size, num_segments, max_steps, device=mid_x.device).to(device)  # Initialize yranges tensor
    for i in range(batch_size):
        for j in range(num_segments):
            steps = torch.linspace(0, 1, int(torch.abs(end_x[i, j] - start_x[i, j]).item()), device=mid_x.device)  # Interpolation steps
            yranges[i, j, :steps.size(0)] = (1 - steps) * start_x[i, j] + steps * (end_x[i, j] - 1)  # Interpolate y-range
    yranges = yranges.flatten(start_dim=1)  # Flatten to 2D tensor
    yranges = torch.cat([yranges, yranges, yranges], dim=1)  # Replicate as in original code
    xranges = torch.zeros(batch_size, num_segments, max_steps, device=mid_x.device)  # Initialize xranges tensor
    for i in range(batch_size):
        for j in range(num_segments):
            steps = torch.linspace(0, 1, int(torch.abs(start_x[i, j] - end_x[i, j]).item()), device=mid_x.device)  # Interpolation steps
            range_i = (1 - steps) * end_y[i, j] + steps * start_y[i, j]
            xranges[i, j, :steps.size(0)] = torch.flip(range_i, dims=[0])  # Interpolate x-range
    xranges = xranges.flatten(start_dim=1)  # Flatten to 2D tensor
    xranges = torch.cat([xranges, xranges - 1, xranges + 1], dim=1)  # Replicate as in original code
    return xranges, yranges

def closeness_penalty(poses, k_poses=3, tolerance=2.0):
    """Calculate a penalty for poses that are too close to each other.

    Args:
        poses (Tensor): Pose tensor of shape (batch_size, k_poses * 3).
        k_poses (int, optional): Number of poses. Defaults to 3.
        tolerance (float, optional): Distance below which poses are penalized. Defaults to 2.0.

    Returns:
        Tensor: Mean normalized penalty across the batch.
    """
    batch_size = poses.size(0)
    midpoints = poses[:, :k_poses * 2].view(batch_size, k_poses, 2)  # Extract midpoints
    midpoints_expanded = midpoints.unsqueeze(2)  # Shape: (batch_size, k_poses, 1, 2)
    distances = torch.norm(midpoints_expanded - midpoints_expanded.transpose(1, 2), dim=3)  # Pairwise distances
    penalty = torch.relu(tolerance - distances)  # Penalty for distances below tolerance
    penalty = penalty * (1 - torch.eye(k_poses, device=device))  # Exclude self-comparison
    total_penalty = torch.sum(penalty, dim=(1, 2))  # Sum penalties for each batch item
    max_possible_penalty = tolerance * (k_poses * (k_poses - 1) / 2)  # Max possible penalty
    normalized_penalty = torch.clamp(total_penalty / max_possible_penalty, min=0.0, max=1.0)  # Normalize penalty
    return normalized_penalty.mean()

def objective_space(poses, droplet, k_poses):
    """Calculate the space objective for optimization.

    Args:
        poses (Tensor): Pose tensor of shape (batch_size, k_poses * 3).
        droplet (Tensor): Droplet images tensor of shape (batch_size, H, W).
        k_poses (int): Number of poses.

    Returns:
        Tensor: Mean space objective across the batch.
    """
    batch_size, H, W = droplet.shape
    max_values = torch.max(droplet.view(batch_size, -1), dim=1, keepdim=True)[0]  # Max values for normalization
    droplet = droplet / max_values.view(batch_size, 1, 1)  # Normalize droplet images
    midpoints = poses[:, :k_poses * 2].view(batch_size, k_poses, 2)  # Extract midpoints
    mid_y = midpoints[:, :, 0]  # y-coordinates
    mid_x = midpoints[:, :, 1]  # x-coordinates
    rotations = poses[:, k_poses * 2:] / 3  # Adjust rotations
    rot_rad = math.pi / 180. * -rotations  # Convert to radians and negate
    xranges, yranges = create_contacts_differentiable_batched(mid_x, mid_y, rot_rad, probe_stroke_pixels)  # Get contact ranges
    contact = soft_placement(xranges, yranges, 200, 200, sigma=0.5, threshold=0.5)  # Generate contact tensor
    drop = torch.stack([blur_tensor(droplet[i], kernel_size=101, sigma=10) for i in range(batch_size)])  # Blur images
    drop_edges = torch.stack([blur_tensor(droplet[i], kernel_size=101, sigma=500) for i in range(batch_size)])  # Blur edges
    drop_full = drop + drop_edges  # Combine blurred images
    contacts = contact * (drop_full / torch.max(drop_full))  # Element-wise multiplication and normalization
    space_obj_batch = 1 - torch.sum(contacts.view(batch_size, -1), dim=1) / differentiable_threshold_sum_batched(contact.view(batch_size, -1))  # Calculate space objective
    return space_obj_batch.mean()

def negative_penalty(output, gamma=1.0):
    """Calculate a penalty for negative values in the tensor.

    Args:
        output (Tensor): The output tensor of shape (batch_size, output_size).
        gamma (float, optional): Scaling factor for the penalty. Defaults to 1.0.

    Returns:
        Tensor: Scaled penalty for negative values.
    """
    negative_values = torch.relu(-output)  # Identify negative values
    penalty = (negative_values ** 2).mean().to(device)  # Compute mean squared penalty
    scaled_penalty = gamma * penalty  # Scale the penalty
    return scaled_penalty

def lower_bound_penalty(output, threshold=25.0, alpha=1.0):
    """Calculate a penalty for values below a specified threshold.

    Args:
        output (Tensor): The output tensor of shape (batch_size, output_size).
        threshold (float): Threshold value.
        alpha (float, optional): Scaling factor for the penalty. Defaults to 1.0.

    Returns:
        Tensor: Scaled penalty for values below the threshold.
    """
    lower_values = torch.relu(threshold - output)  # Identify values below threshold
    penalty = (lower_values ** 2).mean().to(device)  # Compute mean squared penalty
    scaled_penalty = alpha * penalty  # Scale the penalty
    return scaled_penalty

def upper_bound_penalty(output, threshold=200.0, beta=1.0):
    """Calculate a penalty for values exceeding a specified threshold.

    Args:
        output (Tensor): The output tensor of shape (batch_size, output_size).
        threshold (float): Threshold value.
        beta (float, optional): Scaling factor for the penalty. Defaults to 1.0.

    Returns:
        Tensor: Scaled penalty for excessive values.
    """
    excessive_values = torch.relu(output - threshold)  # Identify values exceeding threshold
    penalty = (excessive_values ** 2).mean().to(device)  # Compute mean squared penalty
    scaled_penalty = beta * penalty  # Scale the penalty
    return scaled_penalty

def objective_angle(poses, max_angle, k_poses=3, tolerance=0.5, beta=0.003, gamma=0.01):
    """Calculate a loss that penalizes lack of uniqueness among angles.

    Args:
        poses (Tensor): Pose tensor of shape (batch_size, k_poses * 3).
        max_angle (float): Maximum allowed angle.
        k_poses (int, optional): Number of poses. Defaults to 3.
        tolerance (float, optional): Angle tolerance for uniqueness. Defaults to 0.5.
        beta (float, optional): Scaling factor for upper bound penalty. Defaults to 0.003.
        gamma (float, optional): Scaling factor for negative value penalty. Defaults to 0.01.

    Returns:
        Tensor: Mean combined penalty across the batch.
    """
    angles = (poses[:, k_poses * 2:] / 3).to(device)  # Extract angles and adjust
    batch_size = angles.size(0)
    diffs = torch.abs(angles.unsqueeze(1) - angles.unsqueeze(2))  # Absolute differences between angles
    penalty = torch.relu(tolerance - diffs)  # Penalty for angles within tolerance
    penalty = penalty * (1 - torch.eye(k_poses, device=angles.device))  # Exclude self-comparison
    total_penalty = torch.sum(penalty, dim=(1, 2))  # Sum penalties
    max_penalty = tolerance * k_poses  # Maximum possible penalty
    normalized_penalty = torch.clamp(total_penalty / max_penalty, min=0.0, max=1.0)  # Normalize penalty
    negative_val_penalty = negative_penalty(angles, gamma)  # Penalty for negative angles
    excess_val_penalty = upper_bound_penalty(angles, threshold=max_angle, beta=beta)  # Penalty for angles exceeding max_angle
    combined_loss = normalized_penalty + negative_val_penalty + excess_val_penalty  # Combine penalties
    combined_loss_clamped = torch.clamp(combined_loss, min=0.0, max=1.0)  # Clamp combined loss
    return combined_loss_clamped.mean()

def combined_loss_function(output, img, k_poses=3, lower_threshold=30.0, upper_threshold=170.0, alpha=0.015, beta=0.0015, gamma=0.01):
    """Combined loss function including space and angle objectives and penalties.

    Args:
        output (Tensor): Model output tensor.
        img (Tensor): Input images tensor.
        k_poses (int, optional): Number of poses. Defaults to 3.
        lower_threshold (float, optional): Lower bound threshold. Defaults to 30.0.
        upper_threshold (float, optional): Upper bound threshold. Defaults to 170.0.
        alpha (float, optional): Scaling factor for lower bound penalty. Defaults to 0.015.
        beta (float, optional): Scaling factor for upper bound penalty. Defaults to 0.0015.
        gamma (float, optional): Scaling factor for negative value penalty. Defaults to 0.01.

    Returns:
        Tensor: Combined loss value.
    """
    loss = objective_space(poses=output, droplet=img, k_poses=k_poses)  # Space objective loss
    lower_penalty = lower_bound_penalty(output[:, :k_poses * 2], lower_threshold, alpha)  # Lower bound penalty
    excess_penalty = upper_bound_penalty(output[:, :k_poses * 2], upper_threshold, beta)  # Upper bound penalty
    neg_penalty = negative_penalty(output, gamma)  # Negative value penalty
    close_penalty = closeness_penalty(output, tolerance=2.0)  # Closeness penalty
    combined_loss = loss + lower_penalty + excess_penalty + neg_penalty + close_penalty  # Sum all losses
    combined_loss_clamped = torch.clamp(combined_loss, 0, 1)  # Clamp combined loss
    return combined_loss_clamped

def init_weights(m, lower=-0.5, upper=0.5, noise_std=0.01):
    """Initialize weights for layers.

    Args:
        m (nn.Module): The layer to initialize.
        lower (float, optional): Lower bound for uniform initialization. Defaults to -0.5.
        upper (float, optional): Upper bound for uniform initialization. Defaults to 0.5.
        noise_std (float, optional): Standard deviation for Gaussian noise. Defaults to 0.01.
    """
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(lower, upper)  # Initialize weights uniformly
        if noise_std > 0:
            noise = torch.normal(mean=0.0, std=noise_std, size=m.weight.data.size(), device=device)  # Generate noise
            m.weight.data.add_(noise)  # Add noise to weights
        if m.bias is not None:
            m.bias.data.fill_(0)  # Initialize biases to zero

# Make the CNN wider and shallower. Add attention:

class SelfAttention(nn.Module):
    """Self-Attention module."""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),  # Reduce channels
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),  # Restore channels
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)  # Compute attention map
        return x * attention_map  # Apply attention

class WideCNN(nn.Module):
    """Wide Convolutional Neural Network with Self-Attention.

    Args:
        k_poses (int): Number of poses to predict.
        scale_factor (float): Factor to scale the model's layer sizes.
    """
    def __init__(self, k_poses, scale_factor):
        super(WideCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, int(512 * scale_factor), kernel_size=3, padding=1)  # First convolutional layer
        self.bn1 = nn.BatchNorm2d(int(512 * scale_factor))  # Batch normalization
        self.attention = SelfAttention(int(512 * scale_factor))  # Self-Attention module
        self.conv2 = nn.Conv2d(int(512 * scale_factor), int(2048 * scale_factor), kernel_size=3, padding=1)  # Second convolutional layer
        self.bn2 = nn.BatchNorm2d(int(2048 * scale_factor))  # Batch normalization
        self.pool = nn.MaxPool2d(2)  # Max pooling layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% dropout rate
        self.flattened_size = int(2048 * scale_factor) * 50 * 50  # Calculate size after flattening
        self.fc1 = nn.Linear(self.flattened_size, int(8000 * scale_factor))  # First fully connected layer
        self.fc2 = nn.Linear(int(8000 * scale_factor), int(4000 * scale_factor))  # Second fully connected layer
        self.fc3 = nn.Linear(int(4000 * scale_factor), k_poses * 3)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # First convolutional block
        x = self.attention(x)  # Apply attention
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Second convolutional block
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Second fully connected layer
        x = self.fc3(x)  # Output layer
        return x

# ========= TRAIN NETWORK =========== #

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device

# Assuming 'augmented_random' is defined and is a list/array of images
imgs = torch.from_numpy(np.array(augmented_random)).view(len(augmented_random), 1, 200, 200).float().to(device)  # Load images tensor

# Splitting data into training and validation
train_size = int(0.7 * len(imgs))  # 70% for training
val_size = len(imgs) - train_size  # Remaining for validation

train_imgs, val_imgs = torch.utils.data.random_split(imgs, [train_size, val_size])  # Random split

# Convert Subset to tensor
train_imgs = torch.stack([train_imgs[i] for i in range(len(train_imgs))])  # Stack training images
val_imgs = torch.stack([val_imgs[i] for i in range(len(val_imgs))])  # Stack validation images

train_losses = []  # List to store training losses
val_losses = []  # List to store validation losses
lrs = []  # List to store learning rates

model = WideCNN(k_poses, scale_factor=0.15).to(device)  # Initialize model
optimizer = torch.optim.Adam(model.parameters(), lr=0.00008)  # Define optimizer
model.apply(lambda m: init_weights(m, lower=0.0008, upper=0.00081, noise_std=0.004))  # Initialize weights

E = 9  # Number of epochs
batch_size = 2  # Batch size
model_type = 'ModelB_highval'  # Model type name
running_min = float('inf')  # Initialize running minimum for validation loss
patience = 2  # Early stopping patience
patience_counter = 0  # Counter for early stopping
p_space = 0.5  # Weight for space objective
p_angle = 0.5  # Weight for angle objective

for epoch in range(E):

    model.train()  # Set model to training mode
    for i in range(0, len(train_imgs), batch_size):
        actual_batch_size = min(batch_size, len(train_imgs) - i)  # Adjust for the last batch
        batch_imgs = train_imgs[i:i + actual_batch_size].to(device)  # Get batch of images
        optimizer.zero_grad()  # Zero gradients
        output = model(batch_imgs)  # Forward pass
        loss = (p_space * combined_loss_function(output=output, img=batch_imgs.view(actual_batch_size, 200, 200)) +
                p_angle * objective_angle(poses=output, max_angle=max_angle)).to(device)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        train_losses.append(loss.item())  # Record training loss
        lrs.append(optimizer.param_groups[0]['lr'])  # Record learning rate

    # Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        validation_loss = 0
        for i in range(0, len(val_imgs), batch_size):
            actual_batch_size = min(batch_size, len(val_imgs) - i)  # Adjust for the last batch
            batch_val_imgs = val_imgs[i:i + actual_batch_size].to(device)  # Get batch of validation images
            val_output = model(batch_val_imgs)  # Forward pass
            val_loss = (p_space * combined_loss_function(output=val_output, img=batch_val_imgs.view(actual_batch_size, 200, 200)) +
                        p_angle * objective_angle(poses=val_output, max_angle=max_angle)).to(device)  # Compute validation loss
            val_losses.append(val_loss.item())  # Record validation loss
            validation_loss += val_loss.item() * batch_val_imgs.size(0)  # Accumulate validation loss
        validation_loss /= len(val_imgs)  # Compute average validation loss

    # Log training and validation losses
    file1 = open(os.getcwd() + f'/data/batch{batch_size}_{model_type}_loss_log.txt', 'w')  # Open log file
    print(f"Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {validation_loss}", file=file1)  # Log losses
    file1.close()  # Close log file

    # Early stopping and saving best model
    if validation_loss < running_min:
        running_min = validation_loss  # Update running minimum
        best_model_path = os.path.join(os.getcwd(), f'data/batch{batch_size}_{model_type}_cnn_weights_best.pth')  # Best model path
        torch.save(model.state_dict(), best_model_path)  # Save model weights
        file2 = open(os.getcwd() + f'/data/batch{batch_size}_{model_type}_best_weights_log.txt', 'w')  # Open log file
        print(f"Epoch {epoch}, best val loss {validation_loss}. Saving best weights . . .", file=file2)  # Log message
        file2.close()  # Close log file
        patience_counter += 1  # Increment patience counter
        if patience_counter >= patience:
            print("Early stopping triggered.")  # Early stopping message
            break  # Exit training loop

# Save training and validation losses and learning rates
np.savetxt(os.path.join(os.getcwd(), f'batch{batch_size}_{model_type}_train_losses.csv'), np.array(train_losses), delimiter=',')  # Save training losses
np.savetxt(os.path.join(os.getcwd(), f'batch{batch_size}_{model_type}_val_losses.csv'), np.array(val_losses), delimiter=',')  # Save validation losses
np.savetxt(os.path.join(os.getcwd(), f'batch{batch_size}_{model_type}_learning_rates.csv'), np.array(lrs), delimiter=',')  # Save learning rates
