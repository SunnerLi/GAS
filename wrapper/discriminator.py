from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    """
        The pytorch wrapper of generative auxiliary strategy
    """
    def __init__(self, batch_size, base_filter, adopt_gas = True):
        super(Discriminator, self).__init__()
        # Record variable
        self.adopt_gas = adopt_gas
        self.base_filter = base_filter
        self.batch_size = batch_size

        # Create graph component
        self.construct_conv()
        self.batch1 = nn.BatchNorm2d(self.base_filter)
        self.batch2 = nn.BatchNorm2d(self.base_filter * 2)
        self.batch3 = nn.BatchNorm2d(self.base_filter * 4)
        self.batch4 = nn.BatchNorm2d(self.base_filter * 8)
        self.fc = nn.Linear(256, 1)

    def construct_conv(self):
        if self.adopt_gas == False:
            self.conv1  = nn.Conv2d(1, self.base_filter, 3)
            self.conv2  = nn.Conv2d(self.base_filter, self.base_filter * 2, 3, padding = 1)
            self.conv3  = nn.Conv2d(self.base_filter * 2, self.base_filter * 4, 3, padding = 1)
            self.conv4  = nn.Conv2d(self.base_filter * 4, self.base_filter * 8, 3, padding = 1)
        else:
            self.inc_conv_br1_1, self.inc_conv_br2_shrink_1, self.inc_conv_br2_1 = self.inception_Conv2d(1, self.base_filter)
            self.inc_conv_br1_2, self.inc_conv_br2_shrink_2, self.inc_conv_br2_2 = self.inception_Conv2d(self.base_filter, self.base_filter * 2)
            self.inc_conv_br1_3, self.inc_conv_br2_shrink_3, self.inc_conv_br2_3 = self.inception_Conv2d(self.base_filter * 2, self.base_filter * 4)
            self.inc_conv_br1_4, self.inc_conv_br2_shrink_4, self.inc_conv_br2_4 = self.inception_Conv2d(self.base_filter * 4, self.base_filter * 8)

    def inception_Conv2d(self, in_features, out_features, kernel=3):
        return nn.Conv2d(in_features, out_features // 2, 1), \
            nn.Conv2d(in_features, out_features // 2, 1), \
            nn.Conv2d(out_features // 2, out_features // 2, kernel, padding=1) 

    def inception_block_forward(self, inc_conv_br1, inc_conv_br2_shrink, inc_conv_br2, x):
        incept_conv = torch.cat(
            (inc_conv_br1(x), inc_conv_br2(inc_conv_br2_shrink(x))), 1
        )
        return F.max_pool2d(F.relu(incept_conv), 2)

    def forward(self, x):
        if self.adopt_gas == False:
            x = F.max_pool2d(F.relu(self.batch1(self.conv1(x))), 2)
            x = F.max_pool2d(F.relu(self.batch2(self.conv2(x))), 2)
            x = F.max_pool2d(F.relu(self.batch3(self.conv3(x))), 2)
            x = F.max_pool2d(F.relu(self.batch4(self.conv4(x))), 2)    
        else:
            x = self.inception_block_forward(self.inc_conv_br1_1, self.inc_conv_br2_shrink_1, self.inc_conv_br2_1, x)
            x = self.inception_block_forward(self.inc_conv_br1_2, self.inc_conv_br2_shrink_2, self.inc_conv_br2_2, x)
            x = self.inception_block_forward(self.inc_conv_br1_3, self.inc_conv_br2_shrink_3, self.inc_conv_br2_3, x)
            x = self.inception_block_forward(self.inc_conv_br1_4, self.inc_conv_br2_shrink_4, self.inc_conv_br2_4, x)
        x = x.view(self.batch_size, -1)
        return F.sigmoid(self.fc(x))

if __name__ == '__main__':
    net = Discriminator(batch_size = 32, base_filter = 32, adopt_gas = False)
    net.cuda()
    print(net)
    for i in range(1):
        batch_imgs = Variable(torch.rand(32, 1, 28, 28)).cuda()
        batch_logits = net(batch_imgs)
