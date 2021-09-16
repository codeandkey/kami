# Kami model type.

import consts

import math
import torch
import torch.nn as nn

def transform_input(headers, frames, data=False):
    """Transforms headers and frames into the final board input.
        Expects headers in NC format and frames in NFWHC format."""

    # Combine axes 1 (frameid) and 4 (framedata)

    frames = torch.swapaxes(frames, 1, 3)
    frames = torch.reshape(frames, (-1, 8, 8, consts.FRAME_COUNT * consts.FRAME_SIZE))

    # Reorder frame axes to NCWH for convolution
    frames = frames.permute(0, 3, 1, 2)

    # Expand headers, append to each frame
    headers = headers.unsqueeze(-1)
    headers = headers.unsqueeze(-1)
    headers = headers.expand(-1, -1, 8, 8)

    board = torch.cat((headers, frames), dim=1)

    if data: board = board.cpu().numpy()
    
    return board

class KamiResidual(nn.Module):
    def __init__(self):
        super(KamiResidual, self).__init__()

        self.conv1 = []
        self.conv2 = []

        # Dropout layer
        self.dropout = nn.Dropout(consts.DROPOUT)

        self.conv1 = nn.Conv2d(consts.FILTERS, consts.FILTERS, (3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(consts.FILTERS, consts.FILTERS, (3, 3), padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(consts.FILTERS)
        self.bn2 = nn.BatchNorm2d(consts.FILTERS)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        skip = x

        # First convolution
        x = self.conv1(x)

        # Batch norm
        x = self.bn1(x)

        # ReLU activation
        x = self.relu1(x)

        x = self.dropout(x)

        # Second convolution
        x = self.conv2(x)

        # Batch norm
        x = self.bn2(x)

        # Skip connection
        x = x + skip

        # ReLU activation
        x = self.relu2(x)

        x = self.dropout(x)

        return x

class KamiNet(nn.Module):
    def __init__(self):
        super(KamiNet, self).__init__()

        # Pre-residual convolution
        self.pr_conv = nn.Conv2d(consts.FRAME_SIZE * consts.FRAME_COUNT + consts.HEADER_SIZE, consts.FILTERS, (3, 3), padding=(1, 1))
        self.pr_bn = nn.BatchNorm2d(consts.FILTERS)
        self.pr_relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(consts.DROPOUT)

        # Residual layers
        self.residuals = []

        for i in range(consts.RESIDUALS):
            self.add_module('residual{}'.format(i), KamiResidual())

        # Policy head layers
        self.ph_conv1 = nn.Conv2d(consts.FILTERS, consts.FILTERS, (1, 1))
        self.ph_conv2 = nn.Conv2d(consts.FILTERS, consts.FILTERS, (1, 1))
        self.ph_bn = nn.BatchNorm2d(consts.FILTERS)
        self.ph_relu = nn.ReLU()
        self.ph_flat = nn.Flatten()
        self.ph_fc = nn.Linear(8 * 8 * consts.FILTERS, 4096)
        self.ph_out = nn.Softmax(-1)

        # Value head layers
        self.vh_conv = nn.Conv2d(consts.FILTERS, consts.FILTERS, (1, 1))
        self.vh_bn = nn.BatchNorm2d(consts.FILTERS)
        self.vh_relu1 = nn.ReLU()
        self.vh_flat = nn.Flatten()

        self.vh_fc1 = nn.Linear(8 * 8 * consts.FILTERS, 256)
        self.vh_relu2 = nn.ReLU()
        self.vh_fc2 = nn.Linear(256, 256)
        self.vh_fc3 = nn.Linear(256, 1)
        self.vh_out = nn.Tanh()

    def forward(self, headers, frames, lmm):
        x = transform_input(headers, frames)

        # Forward pre-presidual convolution
        x = self.pr_conv(x)
        x = self.pr_bn(x)
        x = self.pr_relu(x)
        x = self.dropout(x)

        # Forward residual layers
        for i in range(consts.RESIDUALS):
            x = getattr(self, 'residual{}'.format(i))(x)

        # Forward policy head
        ph = x

        # 2 convolutional filters
        ph = self.ph_conv1(ph)
        ph = self.ph_conv2(ph)

        # Batch norm
        ph = self.ph_bn(ph)

        # ReLU activation
        ph = self.ph_relu(ph)
        ph = self.dropout(ph)

        # Reshape (flat), forward through fully connected
        ph = self.ph_flat(ph)
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
        vh = self.dropout(vh)

        # Flatten, dense layer
        vh = self.vh_flat(vh)
        vh = self.vh_fc1(vh)

        # ReLU, then another dense
        vh = self.vh_relu2(vh)
        vh = self.dropout(vh)
        vh = self.vh_fc2(vh)
        vh = self.vh_fc3(vh)

        # Tanh activation
        vh = self.vh_out(vh)

        # Output policy head, value head
        return ph, vh

class Model:
    def __init__(self, path: str = None, allow_cuda=True):
        """Initializes a new model. Loads a model from `path` if it is provided,
           or generates a new model in place."""

        self.cuda = torch.cuda.is_available() and allow_cuda

        if path:
            self.model = torch.jit.load(path)
        else:
            self.model = KamiNet()

        if self.cuda:
            self.model.cuda()

        self.model.train(False)

    def save(self, path: str):
        """Exports the model to `path`."""

        inputs=(
            torch.randn(1, consts.HEADER_SIZE),
            torch.randn(1, 8, 8, consts.FRAME_SIZE * consts.FRAME_COUNT),
            torch.randn(1, 4096),
        )

        if self.cuda:
            inputs = tuple(map(lambda x: x.cuda(), inputs))

        torch.jit.trace(
            self.model,
            inputs
        ).save(path)

        print('Wrote model to {}.'.format(path))

    def to_tensor(self, data):
        """Converts an array of data to a tensor. Initializes the tensor on
           the CUDA device if it is available."""
        if self.cuda:
            return torch.tensor(data, dtype=torch.float32, device='cuda')
        else:
            return torch.tensor(data, dtype=torch.float32)

    def train(self, batches):
        """Trains the model in place on the training data provided in `training_batches`.
           Returns the starting average loss and the ending average loss."""

        # Convert batches to tensors, move to CUDA if needed
        def batch_to_cuda(batch):
            ((headers, frames, lmm), (mcts, result)) = batch

            return (
                (
                    self.to_tensor(headers),
                    self.to_tensor(frames),
                    self.to_tensor(lmm)
                ),
                (
                    self.to_tensor(mcts),
                    self.to_tensor(result)
                )
            )

        batches = list(map(batch_to_cuda, batches))

        self.model.train(True)

        def value_loss(value, result):
            return nn.MSELoss(reduction='sum')(value, result)

        def policy_loss(policy, mcts):
            return -torch.sum(mcts * torch.log(torch.add(policy, consts.POLICY_EPSILON)))

        optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=consts.LEARNING_RATE,
            weight_decay=consts.L2_REG_WEIGHT
        )

        first_avg_loss = None

        for epoch in range(consts.EPOCHS):
            for ((headers, frames, lmm), (mcts, result)) in batches:
                policy, value = self.model(headers, frames, lmm)

                p_loss = policy_loss(policy, mcts)
                v_loss = value_loss(value, result)

                if math.isnan(p_loss): raise RuntimeError('Policy loss became NaN!')
                if math.isnan(v_loss): raise RuntimeError('Value loss became NaN!')

                actual_loss = p_loss + v_loss

                optimizer.zero_grad()
                actual_loss.backward()
                optimizer.step()

            test_loss = 0
            test_vloss = 0
            test_ploss = 0

            with torch.no_grad():
                for ((headers, frames, lmm), (mcts, result)) in batches:
                    policy, value = self.model(headers, frames, lmm)
                    ploss = policy_loss(policy, mcts).cpu().item()
                    vloss = value_loss(value, result).cpu().item()
                    test_ploss += ploss
                    test_vloss += vloss
                    test_loss += ploss + vloss
                
            test_loss /= len(batches)
            test_vloss /= len(batches)
            test_ploss /= len(batches)

            print('Training: Epoch {}/{}, loss avg p{:.2} v{:.2} t{:.2}'.format(epoch + 1, consts.EPOCHS, test_ploss, test_vloss, test_loss), end='\r')

            if first_avg_loss is None:
                first_avg_loss = test_loss

            last_avg_loss = test_loss

        self.model.train(False)

        print()

        return first_avg_loss, last_avg_loss
