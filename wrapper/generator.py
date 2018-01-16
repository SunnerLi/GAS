from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

class Generator(nn.Module):
    """
        The pytorch wrapper of generative auxiliary strategy
    """
    def __init__(self, batch_size, base_filter, adopt_gas = True):
        super(Generator, self).__init__()
        self.base_filter = base_filter
        self.batch_size = batch_size

        # Create graph component
        self.fc     = nn.Linear(100, 7*7*128)
        self.batch1 = nn.BatchNorm1d(7*7*128)
        self.conv1  = nn.Conv2d(128, 4*64, 5, padding=2)
        self.batch2 = nn.BatchNorm1d(4*64)
        self.conv2  = nn.Conv2d(64, 4*32, 5, padding=2)
        self.batch3 = nn.BatchNorm1d(4*32)
        self.conv3  = nn.Conv2d(32, 1, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.batch1(self.fc(x)))
        x = x.view(self.batch_size, -1, 7, 7)        
        x = F.relu(self.batch2(self.conv1(x)))
        x = x.view(self.batch_size, -1, 14, 14)        
        x = F.relu(self.batch3(self.conv2(x)))
        x = x.view(self.batch_size, -1, 28, 28)
        x = F.tanh(self.conv3(x))
        return x

if __name__ == '__main__':
    net = Generator(batch_size = 32, base_filter = 32, adopt_gas = False)
    net.cuda()
    print(net)
    for i in range(1):
        batch_z = Variable(torch.rand(32, 100)).cuda()
        batch_imgs = net(batch_z)