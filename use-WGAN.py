import torch
from WGAN_GC_128 import Generator
import os


if not os.path.exists('gener_seeds/'):
    os.makedirs('gener_seeds/')



# 加载训练好的生成器模型
generator = Generator(1)  # 初始化生成器模型，这里填写您的生成器模型的初始化参数
generator.load_state_dict(torch.load("./generator.pkl"))

# 设置模型为评估模式
generator.eval()

# 生成样本
num_samples = 1
with torch.no_grad():
    for i in range(num_samples):
        # 生成随机噪声向量
        # 这里的100是噪声向量的大小，请根据您的模型参数进行修改
        noise = torch.randn(64, 100, 1, 1)
        # 使用生成器生成样本
        generated_sample = generator(noise)
        generated_sample = generated_sample * 128 + 128  # 将数据映射到 0-255 范围
        generated_sample = generated_sample.to(torch.int)
        for id, sample in enumerate(generated_sample):
          print(id)
          generated_bytes = sample.byte()
          # 将字节流写入二进制文件
          output_file_path = "gener_seeds/generated_samples_"+str(id)
          print(output_file_path)
          with open(output_file_path, "wb") as f:
              f.write(generated_bytes.numpy().tobytes())


