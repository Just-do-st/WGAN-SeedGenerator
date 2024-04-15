import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from WGAN_GP import WGAN_GP
# from WGAN_GC_128 import WGAN_CP
from WGAN_GP_128 import WGAN_GP

from struct_SeedDataset import SeedDataset

class Arg:
    def __init__(self):
        self.channels = 1
        self.cuda = False
        self.generator_iters = 200
        self.load_G = './generator.pkl'
        self.load_D = './discriminator.pkl'

args=Arg()
# model = WGAN_CP(args=args)
model = WGAN_GP(args=args)

# 从本地加载数据集
train_dataset = torch.load('./data/dataset.pth')
# # 定义转换
# transform = transforms.Compose([
#     transforms.Resize((300, 300)),  # 将图像调整为 300x300 大小
#     transforms.ToTensor()  # 将图像转换为张量
# ])
# # 对数据集应用转换
# train_dataset = train_dataset.transform(transform)
batch_size = model.batch_size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model.train(train_dataloader)


# model.evaluate(train_dataloader, args.load_D, args.load_G)
# for i in range(50):
#     model.generate_latent_walk(i)

