# AlexNet, large net, originally for 224 * 224 image size, first layer stride = 4
# here for 32 * 32  image size, i modified strides and paddings. channel keeps the same
# paper: ImageNet Classification with Deep Convolutional Neural Networks, 2012
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # from 32 to 16
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # here we need some dropout to reduce the rebundance
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 128 * 6 * 6)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return probas

def test():
    net = AlexNet()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# check if the net can work
# test()
