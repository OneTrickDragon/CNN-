import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes = 6):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features= self.conv1.out_channels)
        self.relu = nn.ReLU()
        #shape = (batch_size, 16, 150, 150)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #shape = (batch_size, 16, 75, 75)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels= self.conv1.out_channels*2,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.relu2 = nn.ReLU()
        #shape = (batch_Size, 32, 75, 75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        #shape = (batch_size, 32, 37, 37)

        self.conv3 = nn.Conv2d(in_channels= self.conv2.out_channels,
                               out_channels= self.conv2.out_channels*2,
                               kernel_size= 3, padding= 1, stride= 1)


