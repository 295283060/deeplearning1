from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):       
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.fc1 = nn.Linear(50 * 50 * 16, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1) 

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.feature_extr = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=1, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2), 
            nn.Conv2d(96, 256, kernel_size=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2),

            nn.Flatten()    
        )

        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.feature_extr(x)
        x = self.classifier(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=2) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.Softmax = nn.Softmax(1)

    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        out = self.Softmax(out)
        return out
    
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer

class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(densenet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.DB1 = self._make_dense_block(64, growth_rate,num=block_layers[0])
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(512)
        self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(1024)
        self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
        self.global_average = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.DB1(x)
        x = self.TL1(x)
        x = self.DB2(x)
        x = self.TL2(x)
        x = self.DB3(x)
        x = self.TL3(x)
        x = self.DB4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)



def Densenet(num_classes):
    return densenet(in_channel=3,num_classes=num_classes)