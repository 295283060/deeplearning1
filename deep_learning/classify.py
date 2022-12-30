import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import glob
import os
from PIL import Image
import pandas as pd
import numpy as np
import imghdr
from utils import *

class MyDataset(Dataset):
    def __init__(self, path, Train=True, Len=-1, resize=-1, img_type='jpg'):
        if resize != -1:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        img_format = '*.%s' % img_type
        
        if Len == -1:
            self.dataset = [np.array(transform(Image.open(name).convert("RGB"))) for name in
                            glob.glob(os.path.join(path, img_format)) if imghdr.what(name)]
        else:
            self.dataset = [np.array(transform(Image.open(name).convert("RGB"))) for name in
                            glob.glob(os.path.join(path, img_format))[:Len] if imghdr.what(name)]
        self.dataset = np.array(self.dataset)
        self.dataset = torch.Tensor(self.dataset)
        self.Train = Train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=16, type=int, help='batchSize')
parser.add_argument('--lr', default=0.01, type=float, help='')
parser.add_argument('--gpu', default=0, type=float, help='')
parser.add_argument('--epochs', default="501", type=int)
opt = parser.parse_args(args=[])

batch_size = opt.batch_size
learning_rate = opt.lr
num_epoches = opt.epochs
model_name = opt.model

def set_random_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed(45645)

#训练和测试集预处理
test_dataset = MyDataset(path=r'data/testimages', resize=224, Len=1084, img_type='jpg')

#加载数据集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

path= 'model/' + model_name + '_\d*.pth'
model = torch.load(path)
print(model)
if torch.cuda.is_available():
    model = model.cuda()
print('Start test!')
model.eval()

csvpath = "data/sample_submission.csv"
data = pd.read_csv(csvpath)
data = data.to_numpy()

index=0

for img in test_loader:
    if torch.cuda.is_available():
        img = img.cuda()
    out = model(img)
    for item in out:
        if item[0]>item[1]:
            data[index][1]=-1
        else:
            data[index][1]=1
        index+=1

print(data)

get_csv(data)