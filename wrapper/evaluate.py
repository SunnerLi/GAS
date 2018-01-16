from record import saveGeneratedBatch, get_img, save_img
from torchvision import datasets, transforms, utils
from discriminator import Discriminator
from torch.autograd import Variable
from generator import Generator
from torch.optim import RMSprop
import numpy as np
import subprocess
import torch
import os

batch_size = 128

class LSGAN(object):
    def __init__(self, batch_size, adopt_gas = False):
        self.batch_size = batch_size
        self.generator = Generator(batch_size = self.batch_size, base_filter = 32)
        self.discriminator = Discriminator(batch_size = self.batch_size, base_filter = 32, adopt_gas = adopt_gas)
        self.generator.cuda()
        self.discriminator.cuda()
        self.gen_optimizer = RMSprop(self.generator.parameters())
        self.dis_optimizer = RMSprop(self.discriminator.parameters())

    def train(self, epoch, loader):
        self.generator.train()
        self.discriminator.train()
        self.gen_loss_sum = 0.0
        self.dis_loss_sum = 0.0
        for i, (batch_img, batch_tag) in enumerate(loader):
            # Get logits
            batch_img = Variable(batch_img.cuda())
            batch_z = Variable(torch.randn(self.batch_size, 100).cuda())
            self.gen_image = self.generator(batch_z)
            true_logits = self.discriminator(batch_img)
            fake_logits = self.discriminator(self.gen_image)

            # Get loss
            self.dis_loss = torch.sum((true_logits - 1)**2 + (fake_logits)) / 2
            self.gen_loss = torch.sum((fake_logits - 1)**2) / 2

            # Update
            self.dis_optimizer.zero_grad()
            self.dis_loss.backward(retain_graph = True)
            self.dis_loss_sum += self.dis_loss.data.cpu().numpy()[0]
            self.dis_optimizer.step()
            if i % 5 == 0:
                self.gen_optimizer.zero_grad()
                self.gen_loss.backward()
                self.gen_loss_sum += self.gen_loss.data.cpu().numpy()[0]
                self.gen_optimizer.step()

            if i > 300:
                break

    def eval(self):
        self.generator.eval()
        batch_z = Variable(torch.randn(32, 100).cuda())
        return self.generator(batch_z)

if __name__ == '__main__':
    # Create output folder
    args = ['mkdir', '-p', './output/dcgan']
    subprocess.call(" ".join(args), shell=True)
    args = ['mkdir', '-p', './output/gas']
    subprocess.call(" ".join(args), shell=True)

    # Get training data
    dataset = datasets.MNIST(
        './data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]), download=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # --------------------------------------------------------------------------------------
    # Usual LSGAN
    # --------------------------------------------------------------------------------------
    net = LSGAN(batch_size = batch_size)
    print('-' * 20 + ' DCGAN Discriminator' + '-' * 20)
    for i in range(100):
        tensor = net.eval()
        net.train(i, loader)
        tensor = tensor.transpose(1, 2).transpose(2, 3)
        tensor = tensor.data.cpu().numpy()
        tensor = tensor * 0.5 + 0.5
        print('epoch: ', i, '\tgenerator loss: ', net.gen_loss_sum, '\tdiscriminator loss: ', net.dis_loss_sum)
        utils.save_image(net.eval().data, './output/dcgan/' + str(i) + '.png', normalize=True)

    # --------------------------------------------------------------------------------------
    # GAS (Inception version)
    # --------------------------------------------------------------------------------------
    net = LSGAN(batch_size = batch_size, adopt_gas = True)
    print('-' * 20 + ' GAS Discriminator' + '-' * 20)
    for i in range(100):
        tensor = net.eval()
        net.train(i, loader)
        tensor = tensor.transpose(1, 2).transpose(2, 3)
        tensor = tensor.data.cpu().numpy()
        tensor = tensor * 0.5 + 0.5
        print('epoch: ', i, '\tgenerator loss: ', net.gen_loss_sum, '\tdiscriminator loss: ', net.dis_loss_sum)
        utils.save_image(net.eval().data, './output/gas/' + str(i) + '.png', normalize=True)

    # Merge result
    image_name_list = sorted(os.listdir('./output/dcgan/'))
    print(image_name_list)
    for i, img_name in enumerate(image_name_list):
        img1 = get_img('./output/dcgan/' + img_name)
        img2 = get_img('./output/gas/' + img_name)
        save_img('./output/' + str(i) + '.png', np.concatenate((img1, img2), axis=1))