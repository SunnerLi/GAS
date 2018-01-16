from torchvision import datasets, transforms
from discriminator import Discriminator
from record import saveGeneratedBatch
from torch.autograd import Variable
from generator import Generator
from torch.optim import RMSprop
import numpy as np
import torch

class LSGAN(object):
    def __init__(self, adopt_gas = True):
        self.generator = Generator(batch_size = 32, base_filter = 32, adopt_gas = False)
        self.discriminator = Discriminator(batch_size = 32, base_filter = 32, adopt_gas = adopt_gas)
        self.generator.cuda()
        self.discriminator.cuda()
        self.gen_optimizer = RMSprop(self.generator.parameters())
        self.dis_optimizer = RMSprop(self.discriminator.parameters())

    def train(self, epoch, loader):
        self.generator.train()
        self.discriminator.train()
        for i, (batch_img, batch_tag) in enumerate(loader):
            # Get logits
            batch_img = Variable(batch_img.cuda())
            batch_z = Variable(torch.rand(32, 100).cuda())
            self.gen_image = self.generator(batch_z)
            true_logits = self.discriminator(batch_img)
            fake_logits = self.discriminator(self.gen_image)

            # Get loss
            self.dis_loss = torch.sum((true_logits - 1)**2 + (fake_logits)) / 2
            self.gen_loss = torch.sum((fake_logits - 1)**2) / 2

            # Update
            self.dis_optimizer.zero_grad()
            self.dis_loss.backward(retain_graph = True)
            self.dis_optimizer.step()
            if i % 5 == 0:
                self.gen_optimizer.zero_grad()
                self.gen_loss.backward()
                self.gen_optimizer.step()
                # break

    def eval(self):
        self.generator.eval()
        batch_z = Variable(torch.rand(32, 100).cuda())
        return self.generator(batch_z)

if __name__ == '__main__':
    dataset = datasets.MNIST(
        './data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]), download=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)
    net = LSGAN()
    for i in range(1):
        # net.train(i, loader)
        tensor = net.eval()
        tensor = tensor.transpose(1, 2).transpose(2, 3)
        tensor = tensor.data.cpu().numpy()
        tensor = tensor * 0.5 + 0.5
        print(tensor)
        saveGeneratedBatch(tensor, 8, i, output_dir='./output')
