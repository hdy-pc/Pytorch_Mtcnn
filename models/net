import torch
import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()
        )
        # detection
        self.conv_p1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv_p2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv_p3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_layer(x)
        cond = torch.sigmoid(self.conv_p1(x))
        box_offset = self.conv_p2(x)
        land_offset = self.conv_p3(x)
        return cond, box_offset, land_offset

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()
        )
        self.line1 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU()
        )
        # detection
        self.line_r1 = nn.Linear(128, 1)
        # bounding box regression
        self.line_r2 = nn.Linear(128, 4)
        # lanbmark localization
        self.line_r3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.line1(x)

        label = torch.sigmoid(self.line_r1(x))
        box_offset = self.line_r2(x)
        land_offset = self.line_r3(x)

        return label, box_offset, land_offset

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()
        )
        self.line1 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU()
        )

        self.line_o1 = nn.Linear(256, 1)
        self.line_o2 = nn.Linear(256, 4)
        self.line_o3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv_layer(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.line1(x)

        label = torch.sigmoid(self.line_o1(x))
        box_offset = self.line_o2(x)
        land_offset = self.line_o3(x)

        return label, box_offset, land_offset

# num=torch.randn(2,3,48,48)
# net=ONet()
# out=net(num)[0]
# print(out.shape)
