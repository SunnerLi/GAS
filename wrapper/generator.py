from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

batch_size  = 32
base_filter = 32

class Generator(nn.Module):
    """
        The pytorch wrapper of generative auxiliary strategy
    """
    def __init__(self, batch_size, base_filter):
        super(Generator, self).__init__()
        self.base_filter = base_filter
        self.batch_size = batch_size

        # Create graph component
        self.fc     = nn.Linear(100, 7*7*base_filter*8)
        self.batch1 = nn.BatchNorm1d(7*7*base_filter*8)
        self.deconv1 = nn.ConvTranspose2d(base_filter * 8, base_filter * 4, kernel_size = 4, padding = 1, stride = 2)
        self.batch2 = nn.BatchNorm2d(base_filter * 4)
        self.deconv2 = nn.ConvTranspose2d(base_filter * 4, base_filter * 2, kernel_size = 4, padding = 1, stride = 2)
        self.batch3 = nn.BatchNorm2d(base_filter * 2)
        self.deconv3 = nn.ConvTranspose2d(base_filter * 2, base_filter, kernel_size = 3, padding = 1, stride = 1)
        self.batch4 = nn.BatchNorm2d(base_filter)
        self.deconv4 = nn.ConvTranspose2d(base_filter, 1, kernel_size = 3, padding = 1, stride = 1)
        self.batch5 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.batch1(self.fc(x)))
        x = x.view(-1, self.base_filter*8, 7, 7)
        x = F.relu(self.batch2(self.deconv1(x)))
        x = F.relu(self.batch3(self.deconv2(x)))
        x = F.relu(self.batch4(self.deconv3(x)))
        x = F.tanh(self.batch5(self.deconv4(x)))
        return x    

if __name__ == '__main__':
    net = Generator(batch_size = batch_size, base_filter = base_filter)
    net.cuda()
    print(net)
    for i in range(1):
        batch_z = Variable(torch.rand(batch_size, 100)).cuda()
        batch_imgs = net(batch_z)
        print(np.amin(batch_imgs.data.cpu().numpy()), np.amax(batch_imgs.data.cpu().numpy()))
        print(batch_imgs.size())
