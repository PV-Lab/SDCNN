import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch

def generateShapes(gridsize, radius, imagesize):
    """Generate a list of shapes by morphing circles into squares and blobs.

    Args:
        gridsize (int): Number of shapes along one axis (total shapes is gridsize*gridsize).
        radius (float): Base radius for the shapes.
        imagesize (int): Size of the output images (images are square).

    Returns:
        list: A list of 2D numpy arrays representing the shapes.
    """
    shapes = []
    x = np.linspace(-1, 1, imagesize)  # X-coordinate grid from -1 to 1
    y = np.linspace(-1, 1, imagesize)  # Y-coordinate grid from -1 to 1
    xx, yy = np.meshgrid(x, y)  # Create 2D coordinate grid
    theta = np.arctan2(yy, xx)  # Compute polar angle theta for each grid point
    r_grid = np.sqrt(xx ** 2 + yy ** 2)  # Compute radius r for each grid point
    scaling_factor = radius / (imagesize / 2)  # Scaling factor to adjust shape size

    for j in range(gridsize):
        for i in range(gridsize):
            t_x = i / (gridsize - 1)  # Horizontal morph factor (0 to 1)
            t_y = j / (gridsize - 1)  # Vertical morph factor (0 to 1)
            t_x_slow = t_x * 0.03 * i  # Adjust t_x to vary more slowly
            t_y_slow = t_y * 0.15 * j  # Adjust t_y to vary more slowly
            p_max = 100  # Maximum exponent to morph circle into square
            p = 2 + t_x_slow * (p_max - 2)  # Exponent p controls shape from circle (p=2) to square (p->infinity)
            r = scaling_factor * (np.abs(np.cos(theta)) ** p + np.abs(np.sin(theta)) ** p) ** (-1 / p)  # Superellipse equation
            noise_strength_max = 0.4  # Maximum amplitude for perturbations (blobbiness)
            noise_strength = t_y_slow * noise_strength_max  # Adjust noise strength
            k = 8  # Frequency of sinusoidal perturbations
            phi = np.random.uniform(0, 2 * np.pi)  # Random phase shift
            perturbation = 1 + noise_strength * np.sin(k * theta + phi)  # Sinusoidal perturbation applied to radius
            r = r * perturbation  # Apply perturbation to the shape's radius
            shape = r_grid <= r  # Generate shape mask as points inside the boundary
            shapes.append(shape)  # Add shape to the list
    return shapes


def plotPoses(shapes, poses, gridsize, effectorLengthPixels, save=False):
    """Plot the given shapes with their corresponding poses.

    Args:
        shapes (list): List of 2D numpy arrays representing shapes.
        poses (ndarray): Array of poses corresponding to each shape.
        gridsize (int): Number of shapes along one axis.
        effectorLengthPixels (int): Length of the effector in pixels.
        save (bool, optional): Whether to save the plot as an image. Defaults to False.
    """
    k_poses = poses.shape[1]  # Number of poses per shape
    fig, axes = plt.subplots(gridsize, gridsize, figsize=(10, 10))  # Create subplots
    idx = 0  # Index to iterate over shapes
    for j in range(gridsize):
        for i in range(gridsize):
            ax = axes[j, i]
            ax.imshow(shapes[idx], cmap='gray_r')  # Display the shape image
            ax.axis('off')  # Hide axis
            pose_x, pose_y = createContacts(poses[idx], effectorLengthPixels)  # Get contact points for current shape
            idx += 1  # Move to next shape
            for k in range(k_poses):
                ax.plot(pose_y[k], pose_x[k], c='w', lw=2, zorder=10)  # Plot contact line in white (foreground)
                ax.plot(pose_y[k], pose_x[k], c='k', lw=4, alpha=0.9, zorder=5)  # Outline contact line in black (background)
    plt.tight_layout()
    add_labels_with_arrows(fig, axes)  # Add labels and arrows to the plot
    if save:
        plt.savefig('./data/SDCNN_predicted_poses.png', dpi=300, bbox_inches='tight')
    plt.show()


def plotPosesDifferentiable(shapes, poses, gridsize, imagesize, effectorLengthPixels, device='cpu', save=False):
    """Plot shapes with poses using differentiable methods for visualization.

    Args:
        shapes (list): List of 2D numpy arrays representing shapes.
        poses (ndarray): Array of poses corresponding to each shape.
        gridsize (int): Number of shapes along one axis.
        imagesize (int): Size of the images.
        effectorLengthPixels (int): Length of the effector in pixels.
        device (str, optional): Device to perform computations on. Defaults to 'cpu'.
        save (bool, optional): Whether to save the plot as an image. Defaults to False.
    """
    k_poses = poses.shape[1]  # Number of poses per shape
    poses = torch.tensor(poses)  # Convert poses to tensor
    fig, axes = plt.subplots(gridsize, gridsize, figsize=(10, 10))  # Create subplots
    idx = 0  # Index to iterate over shapes
    for j in range(gridsize):
        for i in range(gridsize):
            ax = axes[j, i]
            contacts = torch.zeros(imagesize, imagesize)  # Initialize contact image
            for k in range(k_poses):
                xranges, yranges = create_contacts_differentiable_batched(
                    poses[idx, k, 1].reshape(1, 1),  # Midpoint x-coordinate
                    poses[idx, k, 0].reshape(1, 1),  # Midpoint y-coordinate
                    -(poses[idx, k, 2].reshape(1, 1)) * 3.14159 / 180,  # Rotation in radians (negative)
                    effectorLengthPixels,
                    device=device
                )
                contact = soft_placement(
                    xranges, yranges,
                    imagesize, imagesize,
                    sigma=1, threshold=0, device=device
                ).reshape(imagesize, imagesize)  # Generate soft contact image
                contacts += (contact - contact.min()) / (contact.max() - contact.min())  # Normalize and accumulate contacts
                contacts = contacts.clamp(0, 1)  # Clamp values between 0 and 1
            contacts = contacts.T ** 0.2  # Transpose and adjust contrast
            threshold = 0.0000  # Threshold for masking
            inflec = 0.6  # Inflection point for colormap
            masked_data = np.ma.masked_less(contacts, threshold)  # Mask values below threshold
            cmap = plt.cm.turbo  # Use turbo colormap
            cmap = cmap(np.arange(cmap.N))[int(inflec * cmap.N):]  # Adjust colormap for contacts
            cmap[:, -1] = np.linspace(0, 1, cmap.shape[0])  # Set transparency gradient
            cmap = colors.ListedColormap(cmap)
            ax.imshow(masked_data, cmap=cmap, zorder=10)  # Overlay contacts
            cmap = plt.cm.turbo
            cmap = cmap(np.arange(cmap.N))[:int(inflec * cmap.N)]  # Adjust colormap for shape
            cmap[:, -1] = np.linspace(0.95, 1, cmap.shape[0])  # Set transparency gradient
            cmap = colors.ListedColormap(cmap)
            im_blur = blur_tensor(torch.tensor(shapes[idx]).float(), 30, 8, device)  # Blur the shape image
            ax.imshow(im_blur, cmap=cmap)  # Display blurred shape
            x_grid = np.arange(0, imagesize, 10)  # Vertical gridlines every 10 pixels
            y_grid = np.arange(0, imagesize, 10)  # Horizontal gridlines every 10 pixels
            for xg in x_grid:
                ax.axvline(x=xg, color='k', linestyle='-', linewidth=0.5, alpha=0.2, zorder=20)  # Draw vertical gridlines
            for yg in y_grid:
                ax.axhline(y=yg, color='k', linestyle='-', linewidth=0.5, alpha=0.2, zorder=20)  # Draw horizontal gridlines
            ax.axis('off')  # Hide axis
            idx += 1  # Move to next shape
    plt.tight_layout()
    add_labels_with_arrows(fig, axes)
    if save:
        plt.savefig('./data/SDCNN_predicted_poses_differentiable.png', dpi=300, bbox_inches='tight')
    plt.show()


def gaussian_kernel(size, sigma):
    """Create a 1D Gaussian kernel.

    Args:
        size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        Tensor: 1D Gaussian kernel.
    """
    x = torch.arange(size).float() - size // 2  # Coordinate grid centered at zero
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))  # Compute Gaussian values
    kernel_1d /= kernel_1d.sum()  # Normalize kernel
    return kernel_1d


def blur_tensor(tensor, kernel_size, sigma, device):
    """Apply Gaussian blur to a 2D tensor.

    Args:
        tensor (Tensor): 2D tensor to blur.
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian.
        device (str): Device to perform computations on.

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


def create_contacts_differentiable_batched(mid_x, mid_y, rot_rad, probe_stroke_pixels, device):
    """Create contact points in a differentiable way for batch processing.

    Args:
        mid_x (Tensor): Midpoint x-coordinates (batch_size, num_segments).
        mid_y (Tensor): Midpoint y-coordinates (batch_size, num_segments).
        rot_rad (Tensor): Rotation angles in radians (batch_size, num_segments).
        probe_stroke_pixels (float): Length of the probe stroke in pixels.
        device (str): Device to perform computations on.

    Returns:
        Tuple[Tensor, Tensor]: xranges and yranges tensors for contact points.
    """
    dx = (probe_stroke_pixels / 2) * torch.cos(rot_rad)  # Compute half-length offsets in x direction
    dy = (probe_stroke_pixels / 2) * torch.sin(rot_rad)  # Compute half-length offsets in y direction
    start_x = mid_x - dx  # Calculate start x-coordinate of the line segment
    start_y = mid_y - dy  # Calculate start y-coordinate of the line segment
    end_x = mid_x + dx  # Calculate end x-coordinate of the line segment
    end_y = mid_y + dy  # Calculate end y-coordinate of the line segment
    max_nsteps = torch.max(torch.abs(end_x - start_x)).int()  # Maximum number of steps needed
    max_xsteps = torch.max(torch.abs(start_x - end_x)).int()  # Maximum x-steps
    batch_size, num_segments = mid_x.size(0), mid_x.size(1)
    max_steps = torch.max(max_nsteps, max_xsteps)  # Maximum number of steps for interpolation
    yranges = torch.zeros(batch_size, num_segments, max_steps, device=device)  # Initialize yranges tensor
    for i in range(batch_size):
        for j in range(num_segments):
            steps = torch.linspace(0, 1, int(torch.abs(end_x[i, j] - start_x[i, j]).item()), device=device)  # Interpolation steps
            yranges[i, j, :steps.size(0)] = (1 - steps) * start_x[i, j] + steps * (end_x[i, j] - 1)  # Interpolate y-range
    yranges = yranges.flatten(start_dim=1)  # Flatten to 2D tensor
    yranges = torch.cat([yranges, yranges, yranges], dim=1)  # Replicate yranges
    xranges = torch.zeros(batch_size, num_segments, max_steps, device=device)  # Initialize xranges tensor
    for i in range(batch_size):
        for j in range(num_segments):
            steps = torch.linspace(0, 1, int(torch.abs(start_x[i, j] - end_x[i, j]).item()), device=device)  # Interpolation steps
            range_i = (1 - steps) * end_y[i, j] + steps * start_y[i, j]
            xranges[i, j, :steps.size(0)] = torch.flip(range_i, dims=[0])  # Interpolate x-range
    xranges = xranges.flatten(start_dim=1)  # Flatten to 2D tensor
    xranges = torch.cat([xranges, xranges - 1, xranges + 1], dim=1)  # Replicate and adjust xranges
    return xranges, yranges


def soft_placement(x_coords, y_coords, width, height, device, sigma=0.1, threshold=0.5):
    """Create a 2D tensor with soft placements of contact points.

    Args:
        x_coords (Tensor): x-coordinates of contact points (batch_size, num_points).
        y_coords (Tensor): y-coordinates of contact points (batch_size, num_points).
        width (int): Width of the output tensor.
        height (int): Height of the output tensor.
        device (str): Device to perform computations on.
        sigma (float, optional): Standard deviation for Gaussian peaks. Defaults to 0.1.
        threshold (float, optional): Threshold for soft thresholding. Defaults to 0.5.

    Returns:
        Tensor: 2D tensor with soft placements.
    """
    batch_size, num_points = x_coords.shape
    xx, yy = torch.meshgrid(torch.linspace(0, width - 1, width),
                            torch.linspace(0, height - 1, height),
                            indexing='xy')  # Create coordinate grid
    xx, yy = xx.to(device), yy.to(device)
    xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1, -1)  # Expand grid dimensions
    yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1, -1)
    x_coords = x_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)  # Expand coordinates
    y_coords = y_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
    gaussian_peaks = torch.exp(-((xx - x_coords) ** 2 + (yy - y_coords) ** 2) / (2 * sigma ** 2)).to(device)  # Compute Gaussian peaks
    tensor_2d = torch.sum(gaussian_peaks, dim=1).to(device)  # Sum over contact points
    tensor_2d = torch.sigmoid((tensor_2d - threshold) * 10).to(device)  # Apply soft thresholding
    return tensor_2d


def createContacts(output, effectorLengthPixels):
    """Determine the contacted pixels of the probe within the image for given poses.

    Args:
        output (ndarray): Array of pose parameters.
        effectorLengthPixels (float): Length of the effector in pixels.

    Returns:
        Tuple[list, list]: Lists of x and y coordinates of contact points.
    """
    num_poses = output.shape[0]
    xcontact = []
    ycontact = []
    for k in range(num_poses):
        mid_y = output[k, 0]  # Midpoint y-coordinate
        mid_x = output[k, 1]  # Midpoint x-coordinate
        rotation = output[k, 2]  # Rotation angle in degrees
        rotation_radians = np.radians(-np.array(rotation))  # Convert rotation angle to radians (negative)
        dx = (effectorLengthPixels / 2) * np.cos(rotation_radians)  # Half-length offset in x
        dy = (effectorLengthPixels / 2) * np.sin(rotation_radians)  # Half-length offset in y
        start_x = mid_x - dx  # Start x-coordinate of line segment
        start_y = mid_y - dy  # Start y-coordinate of line segment
        end_x = mid_x + dx  # End x-coordinate of line segment
        end_y = mid_y + dy  # End y-coordinate of line segment
        probe_x = np.array([start_x, end_x]).T.round(0).astype(int)  # x-coordinates of probe line
        probe_y = np.array([start_y, end_y]).T.round(0).astype(int)  # y-coordinates of probe line
        line_y = np.arange(*probe_x, 1)  # Generate y-values for line
        ycontact.append(line_y)  # Append y-values
        xcontact.append(np.linspace(*probe_y, len(line_y)).round(0).astype(int))  # Generate x-values and append
    return xcontact, ycontact


def add_labels_with_arrows(fig, axes):
    """Add labels 'Sharp edges' and 'Non-convex' with arrows to the figure.

    Args:
        fig (Figure): Matplotlib figure object.
        axes (ndarray): Array of axes (subplots).
    """
    fig.canvas.draw()  # Ensure figure is drawn to get updated positions
    top_row = axes[0, :]
    left_column = axes[:, 0]
    x_text_top = 0.5  # Centered horizontally
    y_text_top = top_row[0].get_position().y1 + 0.02  # Just above the top row
    fig.text(
        x_text_top, y_text_top, 'Sharp edges',
        ha='center', va='bottom', fontsize=17, transform=fig.transFigure
    )
    arrow_right = FancyArrowPatch(
        (top_row[0].get_position().x0, y_text_top - 0.008),
        (top_row[-1].get_position().x1, y_text_top - 0.008),
        arrowstyle='->', lw=2, mutation_scale=20,
        transform=fig.transFigure,
        clip_on=False
    )
    fig.add_artist(arrow_right)
    x_text_left = left_column[0].get_position().x0 - 0.02  # Just to the left of the first column
    y_text_left = 0.5  # Centered vertically
    fig.text(
        x_text_left, y_text_left, 'Non-convex',
        ha='right', va='center', rotation='vertical', fontsize=17, transform=fig.transFigure
    )
    arrow_down = FancyArrowPatch(
        (x_text_left + 0.008, left_column[0].get_position().y1),
        (x_text_left + 0.008, left_column[-1].get_position().y0),
        arrowstyle='->', lw=2, mutation_scale=20,
        transform=fig.transFigure,
        clip_on=False
    )
    fig.add_artist(arrow_down)
