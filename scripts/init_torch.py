# Initializing script for Torch models.

import os
import pathlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_COUNT=6
FRAME_SIZE=14
HEADER_SIZE=18

RESIDUAL_CONV_FILTERS=0
RESIDUAL_LAYERS=1
CHANNELS = 14 * FRAME_COUNT + HEADER_SIZE

class KamiResidual(nn.Module):
    def __init__(self):
        super(KamiResidual, self).__init__()

        self.conv1 = []
        self.conv2 = []

        for _ in range(RESIDUAL_CONV_FILTERS):
            self.conv1.append(nn.Conv2d(CHANNELS, CHANNELS, (3, 3), padding=(1, 1)))
            self.conv2.append(nn.Conv2d(CHANNELS, CHANNELS, (3, 3), padding=(1, 1)))

        self.bn1 = nn.BatchNorm2d(CHANNELS)
        self.bn2 = nn.BatchNorm2d(CHANNELS)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        skip = x

        # N convolutional filters
        for conv in self.conv1:
            x = conv(x)

        # Batch norm
        x = self.bn1(x)

        # ReLU activation
        x = self.relu1(x)

        # N convolutional filters
        for conv in self.conv2:
            x = conv(x)

        # Batch norm
        x = self.bn2(x)

        # Skip connection
        x = x + skip

        # ReLU activation
        x = self.relu2(x)

        return x

class KamiNet(nn.Module):
    def __init__(self):
        super(KamiNet, self).__init__()

        # Pre-residual convolution
        self.pr_conv = nn.Conv2d(CHANNELS, CHANNELS, (3, 3), padding=(1, 1))
        self.pr_bn = nn.BatchNorm2d(CHANNELS)
        self.pr_relu = nn.ReLU()

        # Residual layers
        self.residuals = []

        for i in range(RESIDUAL_LAYERS):
            self.add_module('residual{}'.format(i), KamiResidual())

        # Policy head layers
        self.ph_conv1 = nn.Conv2d(CHANNELS, CHANNELS, (1, 1))
        self.ph_conv2 = nn.Conv2d(CHANNELS, CHANNELS, (1, 1))
        self.ph_bn = nn.BatchNorm2d(CHANNELS)
        self.ph_relu = nn.ReLU()
        self.ph_flat = nn.Flatten()
        self.ph_fc = nn.Linear(8 * 8 * CHANNELS, 4096)
        self.ph_out = nn.Softmax(-1)

        # Value head layers
        self.vh_conv = nn.Conv2d(CHANNELS, CHANNELS, (1, 1))
        self.vh_bn = nn.BatchNorm2d(CHANNELS)
        self.vh_relu1 = nn.ReLU()
        self.vh_flat = nn.Flatten()

        self.vh_fc1 = nn.Linear(8 * 8 * CHANNELS, 256)
        self.vh_relu2 = nn.ReLU()
        self.vh_fc2 = nn.Linear(256, 256)
        self.vh_fc3 = nn.Linear(256, 1)
        self.vh_out = nn.Tanh()

    def forward(self, headers, frames, lmm):
        # Reorder frame axes to NCHW for convolution
        x = frames.permute(0, 3, 1, 2)

        print('After frames permute: {}'.format(x.shape))

        # Expand headers, append to each frame
        print('Headers: before expansion: {}'.format(headers.shape))
        headers = headers.unsqueeze(-1)
        headers = headers.unsqueeze(-1)
        headers = headers.expand(-1, -1, 8, 8)
        print('Headers: after expansion: {}'.format(headers.shape))
        
        x = torch.cat((headers, x), dim=1)

        print('After headers concat: {}'.format(x.shape))

        # Forward pre-presidual convolution
        x = self.pr_conv(x)
        x = self.pr_bn(x)
        x = self.pr_relu(x)

        print('Before residuals: {}'.format(x.shape))

        # Forward residual layers
        for i in range(RESIDUAL_LAYERS):
            x = getattr(self, 'residual{}'.format(i))(x)

        print('After residuals: {}'.format(x.shape))

        # Forward policy head
        ph = x

        # 2 convolutional filters
        print('Policy: before convolution: {}'.format(ph.shape))

        ph = self.ph_conv1(ph)
        ph = self.ph_conv2(ph)

        print('Policy: after convolution: {}'.format(ph.shape))

        # Batch norm
        ph = self.ph_bn(ph)

        # ReLU activation
        ph = self.ph_relu(ph)

        print('Policy: before reshape: {}'.format(ph.shape))

        # Reshape (flat), forward through fully connected
        ph = self.ph_flat(ph)

        print('Policy: after reshape: {}'.format(ph.shape))

        ph = self.ph_fc(ph)

        # Perform exp
        ph = ph.exp()

        # Apply LMM
        ph = ph.mul(lmm)

        # Find sum
        ss = torch.sum(ph, dim=1, keepdim=True)

        # Expand total sum
        ss = ss.expand(-1, 4096)

        # Perform div
        ph = torch.div(ph, ss)

        # Forward value head
        vh = x

        # Convolutional filter
        vh = self.vh_conv(vh)

        # Batch norm, ReLU
        vh = self.vh_bn(vh)
        vh = self.vh_relu1(vh)

        # Flatten, dense layer
        print('Value: before flat: {}'.format(vh.shape))

        vh = self.vh_flat(vh)

        print('Value: before fc1: {}'.format(vh.shape))

        vh = self.vh_fc1(vh)

        print('Value: after fc1: {}'.format(vh.shape))

        # ReLU, then another dense
        vh = self.vh_relu2(vh)
        vh = self.vh_fc2(vh)
        vh = self.vh_fc3(vh)

        # Tanh activation
        vh = self.vh_out(vh)

        print('Final ph: {}, vh: {}'.format(ph.shape, vh.shape))

        # Output policy head, value head
        return ph, vh

print('Initializing model.')
model = KamiNet()
print('Done.')

dummy_inputs=(
    torch.randn(32, HEADER_SIZE),
    torch.randn(32, 8, 8, 14 * FRAME_COUNT),
    torch.randn(32, 4096),
)

input_names=['headers', 'frames', 'lmm']
output_names=['policy', 'value']

if os.path.exists(sys.argv[1]):
    print('Destination path already exists!')
    sys.exit(1)

print('Exporting model to {}.'. format(sys.argv[1]))
torch.jit.trace(model, example_inputs=dummy_inputs).save(sys.argv[1])
print('Done.')