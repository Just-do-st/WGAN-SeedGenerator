import torch
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class SeedDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # 这里根据需要自行处理数据
        return torch.tensor(sample, dtype=torch.float32)
