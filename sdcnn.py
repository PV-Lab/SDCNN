import numpy as np
import random
import os
import requests
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

def loadNetwork(filename, folder, device, k_poses=3, scale_factor=0.15):
    """Load the SDCNN network with pre-trained weights.

    Args:
        path (str): Path to the saved model weights.
        device (str): Device to load the model on ('cpu' or 'cuda').
        k_poses (int, optional): Number of poses to predict. Defaults to 3.
        scale_factor (float, optional): Factor to scale the model's layer sizes. Defaults to 0.15.

    Returns:
        nn.Module: Loaded SDCNN model.
    """
    class SelfAttention(nn.Module):
        def __init__(self, in_channels):
            super(SelfAttention, self).__init__()
            self.attention = nn.Sequential(  # Sequential layers to generate attention map
                nn.Conv2d(in_channels, in_channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 8, in_channels, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            attention_map = self.attention(x)  # Compute attention map
            return x * attention_map  # Apply attention to input

    class SDCNN(nn.Module):
        def __init__(self, k_poses, scale_factor):
            super(SDCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, int(512 * scale_factor), kernel_size=3, padding=1)  # First convolutional layer
            self.bn1 = nn.BatchNorm2d(int(512 * scale_factor))  # Batch normalization after conv1
            self.attention = SelfAttention(int(512 * scale_factor))  # Self-attention module
            self.conv2 = nn.Conv2d(int(512 * scale_factor), int(2048 * scale_factor), kernel_size=3, padding=1)  # Second convolutional layer
            self.bn2 = nn.BatchNorm2d(int(2048 * scale_factor))  # Batch normalization after conv2
            self.pool = nn.MaxPool2d(2)  # Max pooling layer
            self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% dropout rate
            self.flattened_size = int(2048 * scale_factor) * 50 * 50  # Calculate size after flattening
            self.fc1 = nn.Linear(self.flattened_size, int(8000 * scale_factor))  # First fully connected layer
            self.fc2 = nn.Linear(int(8000 * scale_factor), int(4000 * scale_factor))  # Second fully connected layer
            self.fc3 = nn.Linear(int(4000 * scale_factor), k_poses * 3)  # Output layer for pose predictions

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Convolution, batch norm, ReLU, and pooling
            x = self.attention(x)  # Apply attention here
            x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Second convolutional block with pooling
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc1(x))  # First fully connected layer with ReLU
            x = self.dropout(x)  # Apply dropout
            x = F.relu(self.fc2(x))  # Second fully connected layer with ReLU
            x = self.fc3(x)  # Output layer
            return x
    print('Loading SDCNN Weights . . .')
    weightsSDCNN = SDCNN(k_poses, scale_factor)  # Instantiate the model
    weightsSDCNN.load_state_dict(torch.load(folder+filename, map_location=torch.device(device))) # Load weights
    print('Successfully Loaded SDCNN Weights!')
    return weightsSDCNN


def predictPoses(shapes, SDCNN, erode=10, device='cpu', augment=True):
    """Predict poses from given shapes using the SDCNN model.

    Args:
        shapes (list or ndarray): Input images or shapes to predict poses from.
        SDCNN (nn.Module): Loaded SDCNN model.
        erode (int, optional): Erosion kernel size. Defaults to 10.
        device (str, optional): Device to run the computation on. Defaults to 'cpu'.
        augment (bool, optional): Augments outputs using multiple predictions for better performance.

    Returns:
        ndarray: Predicted poses.
    """
    shapes = np.array(shapes)  # Convert shapes to numpy array
    frame_width = shapes.shape[-1]  # Get frame width
    if len(shapes.shape) == 2:
        shapes = shapes.reshape(1, frame_width, frame_width)  # Reshape if single image
    shapes = torch.tensor(shapes.astype(np.float32))  # Convert to tensor
    poses = []  # Initialize list for poses
    for i in range(len(shapes)):
        img = shapes[i].view(frame_width, frame_width)  # Get individual image
        W, H = img.shape  # Get width and height
        img_eroded = torch.tensor(cv2.erode(img.numpy(), np.ones((erode, erode), np.uint8))).to(device)  # Apply erosion and move to device
        pose = np.empty((0, 3))  # Initialize pose array
        if augment:
            output = SDCNN(torch.tensor(np.array([img_eroded]*4)).reshape(4,1,200,200)) # augment output with multiple (4) predictions
            k_poses = output.shape[-1] // 3  # Number of poses
            selection = np.array([np.random.choice(int(output.shape[0]), k_poses, replace=False),np.random.choice(k_poses, k_poses, replace=False)]).T # select k_pose number of outputs from set of augmented predictions
            for n,k in selection:
                out = [output[n, k * 2].item(),  # x-coordinate
                       output[n, k * 2 + 1].item(),  # y-coordinate
                       output[n, k_poses * 2 + k].item() / 3]  # theta rotation (divided by 3, the model predicts theta*3)
                pose = np.vstack((pose, out))  # Stack pose

        else:
            output = SDCNN(img_eroded.view(1, 1, W, H))  # Get model output
            k_poses = output.shape[-1] // 3  # Number of poses
            for k in range(k_poses):
                out = [output[:, k * 2].item(),  # x-coordinate
                       output[:, k * 2 + 1].item(),  # y-coordinate
                       output[:, k_poses * 2 + k].item() / 3]  # theta rotation (divided by 3, the model predicts theta*3)
                pose = np.vstack((pose, out))  # Stack pose
        poses.append(pose)  # Add pose to list
    return np.array(poses)  # Return poses as array


def downloadWeights(url, filename, fn=None):
    """Download weights from a given URL and save them to a file.

    Args:
        url (str): The URL to download the file from.
        filename (str): The name to save the downloaded content as.
        fn (str, optional): Optional filename parameter. Defaults to the last part of the URL if None.

    Returns:
        None
    """
    directory = 'data'
    file_path = os.path.join(directory, filename)  # Construct the full file path

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    if os.path.exists(file_path):
        print(f"{file_path} is already downloaded.")  # Notify that the file is already downloaded
        return  # Exit the function if the file already exists

    if fn is None:
        fn = url.split('/')[-1]  # Extract filename from URL if fn is not provided

    print(f'Downloading {filename} from {url} . . .')
    r = requests.get(url)  # Send HTTP GET request to the URL
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(r.content)  # Write the content to a file in binary mode
        print("{} downloaded: {:.3f} MB".format(file_path, len(r.content) / 1024 / 1024))  # Print success message with file size
    else:
        print("URL not found:", url)  # Print error message if URL is not accessible