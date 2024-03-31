"""Inspired by https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # Basic block for ResNet

    def __init__(self, input_channels, output_channels, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # Batch normalization after first convolution
        self.bn1 = nn.BatchNorm2d(output_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # Batch normalization after second convolution
        self.bn2 = nn.BatchNorm2d(output_channels)

        # Shortcut connection to handle change in dimensions
        self.shortcut = nn.Sequential()
        # If stride is not 1 or input and output dimensions don't match
        if stride != 1 or input_channels != output_channels:
            # Use 1x1 convolution to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )

    # Forward pass through the block
    def forward(self, x):
        # Pass through first convolution, batch normalization, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Pass through second convolution and batch normalization
        out = self.bn2(self.conv2(out))
        # Add shortcut connection
        out += self.shortcut(x)
        # Apply ReLU activation
        out = F.relu(out)
        return out


# ResNet architecture
class ResNet(nn.Module):
    # Initialize ResNet
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.input_channels = 64  # Initial number of input channels

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        # Batch normalization after initial convolution
        self.bn1 = nn.BatchNorm2d(64)

        # Layers of ResNet
        # Each layer consists of multiple basic blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Fully connected layer for classification
        self.linear = nn.Linear(512, num_classes)

    # Helper function to create a layer with multiple blocks
    def _make_layer(self, block, output_channels, num_blocks, stride):
        # List of strides for each block in the layer
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # Create a block and append to the layer
            layers.append(block(self.input_channels, output_channels, stride))
            self.input_channels = output_channels
        return nn.Sequential(*layers)

    # Forward pass through the network
    def forward(self, x):
        # Pass through initial convolution, batch normalization, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Pass through each layer of ResNet
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Average pooling
        out = F.avg_pool2d(out, 4)
        # Flatten for fully connected layer
        out = out.view(out.size(0), -1)
        # Classification
        out = self.linear(out)
        return out


# Function to create ResNet18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # 18-layer ResNet with 2 blocks each for layers 2, 3, 4
