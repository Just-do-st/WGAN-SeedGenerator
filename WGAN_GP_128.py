import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
plt.switch_backend('agg')
import os
from tensorboard_logger import Logger
from itertools import chain
from torchvision import utils

SAVE_PER_TIMES = 100


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        # Z latent vector 100
        self.convtd1 = nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.ru1 = nn.ReLU(True)

        # State (1024x4x4)
        self.convtd2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.ru2 = nn.ReLU(True)

        # State (512x8x8)
        self.convtd3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.ru3 = nn.ReLU(True)

        # State (256x16x16)
        self.convtd4 = nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(num_features=256),
        # self.ru4 = nn.ReLU(True),

        # State (cx32x32)
        self.fc = nn.Linear(in_features=32 * 32, out_features=128 * 128)
        # output of main module --> Image (Cx64x64)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.convtd1(x)
        x = self.bn1(x)
        x = self.ru1(x)

        x = self.convtd2(x)
        x = self.bn2(x)
        x = self.ru2(x)

        x = self.convtd3(x)
        x = self.bn3(x)
        x = self.ru3(x)

        x = self.convtd4(x)

        x = x.view(x.size(0), -1)  # 展平操作,以便输入全连接层进行处理。
        x = self.fc(x)

        x = self.output(x)

        return x.view(64, 1, 128, 128)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        
        # Image (Cx128x128)
        self.cv = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.ru = nn.LeakyReLU(0.2, inplace=True)
        
        # Image (64x64x64)
        self.cv0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(num_features=128)
        self.ru0 = nn.LeakyReLU(0.2, inplace=True)

        # Image (128x32x32)
        self.cv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.ru1 = nn.LeakyReLU(0.2, inplace=True)

        # State (256x16x16)
        self.cv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.ru2 = nn.LeakyReLU(0.2, inplace=True)

        # State (512x8x8)
        self.cv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=1024)
        self.ru3 = nn.LeakyReLU(0.2, inplace=True)
        # output of main module --> State (1024x4x4)

       
        # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
        self .fc = nn.Linear(in_features=1024 * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.cv(x)
        x = self.bn(x)
        x = self.ru(x)

        x = self.cv0(x)
        x = self.bn0(x)
        x = self.ru0(x)

        x = self.cv1(x)
        x = self.bn1(x)
        x = self.ru1(x)

        x = self.cv2(x)
        x = self.bn2(x)
        x = self.ru2(x)

        x = self.cv3(x)
        x = self.bn3(x)
        x = self.ru3(x)

        x = x.view(x.size(0), -1)  # 展平操作,以便输入全连接层进行处理。
        x = self.fc(x)

        return x.view(64, 1)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class WGAN_GP(object):
    def __init__(self, args):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # Check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 64

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # 替换为 RMSprop 优化器
        # self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        # self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=self.learning_rate)
        self.d_scheduler = lr_scheduler.StepLR(self.d_optimizer, step_size=30, gamma=0.1)
        self.g_scheduler = lr_scheduler.StepLR(self.g_optimizer, step_size=30, gamma=0.1)

        # Set the logger
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader):
        G_losses = []
        D_losses = []

        self.t_begin = t.time()
        self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            d_loss_avg = 0.0
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()

                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                images = images.view(64, 1, 128, 128)
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(
                    f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
                d_loss_avg += d_loss
            print(d_loss_avg)
            D_losses.append(d_loss_avg.item() / self.critic_iter)

            self.d_scheduler.step()
            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            self.g_scheduler.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            G_losses.append(g_loss.tolist())
            
            # 每个 epoch 结束后绘制折线图
            plt.plot(G_losses, label='Generator Loss')
            plt.plot(D_losses, label='Discriminator Loss')
            # plt.legend()
            plt.title('WGAN Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            # 保存图像为文件
            plt.savefig('aaaa_GP_128.png')


            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()
                # # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(125):
                #     samples  = self.data.__next__()
                # #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                # #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                # #
                # # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')

                # # Denormalize images and save them in grid 8x8
                # z = self.get_torch_variable(torch.randn(800, 100, 1, 1))
                # samples = self.G(z)
                # samples = samples.mul(0.5).add(0.5)
                # samples = samples.data.cpu()[:32]
                # grid = utils.make_grid(samples)
                # utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

                # Testing
                time = t.time() - self.t_begin
                # print("Real Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                # output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                # self.file.write(output)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_cost.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data

                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value.cpu(), g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'generated_images': self.generate_img(z, self.number_of_images)
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # self.file.close()

        # Save the trained parameters
        self.save_model()



    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                      prob_interpolated.size()),
                                  create_graph=True, retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 128, 128)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 128, 128)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 128, 128))
            else:
                generated_images.append(sample.reshape(128, 128))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, seed in enumerate(data_loader):
                yield seed

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1 * alpha + z2 * (1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.C, 256, 256).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
