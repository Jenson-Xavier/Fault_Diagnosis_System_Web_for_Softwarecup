from torch import nn
import torch
import warnings
warnings.filterwarnings("ignore")


# 搭建cnn网络
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.SELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.act(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.SELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, 1, 1, 0)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        out += identity
        out = self.act(out)
        return out


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, 1),
            BottleNeck(8, 8),
            nn.Conv1d(8, 16, 3, 2, 1),
            BottleNeck(16, 16),
            nn.Conv1d(16, 16, 3, 2, 1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(416, 6),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = x.to(torch.float32)
        x = self.model(x)
        return x
