import torch
import torch.nn as nn
import torch.nn.functional as F
# class NetModule(torch.nn.Module):
#     def __init__(self, x_num, dims, dropout):
#         super().__init__()
#         layers = list()
#         input_dim = x_num
#         for embed_dim in dims:
#             layers.append(torch.nn.Linear(input_dim, embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim, 1))
#         self.mlp = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         return self.mlp(x)

# class NetModule(nn.Module):
#     def __init__(self,x_num, dims, dropout):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1,20,kernel_size=3,stride=1,padding=1)
#         self.conv2 = nn.Conv2d(20,40,kernel_size=3,stride=1,padding=1)
#         self.conv2_drop = nn.Dropout2d()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(1960,500)
#         self.fc2 = nn.Linear(500,100)
#         self.fc3 = nn.Linear(100,1)

#     def forward(self, x):
#         #28*28->28*28->14*14->14*14->7*7
#         x = F.relu(self.pool1(self.conv1(x)))
#         x = F.relu(self.pool2(self.conv2_drop(self.conv2(x))))
#         x = x.view(-1,1960)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc3(x)
#         return x

class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NetModule(nn.Module):
    def __init__(self, x_num, dims, dropout):
        super(NetModule, self).__init__()
        block = Basicblock
        num_block = [1, 1, 1, 1]
        num_classes = 1
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [200,256]
        out = self.outlayer(x)
        return out