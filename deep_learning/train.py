import argparse
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
import glob
import os
from PIL import Image
import pandas as pd
import numpy as np
import piexif
import imghdr
from main import *


class MyDataset(Dataset):
    def __init__(self, path, Train=True, Len=-1, resize=-1, img_type='jpg', remove_exif=False, labelpath=''):
        all_label = pd.read_csv(labelpath)
        label = all_label.to_numpy()[:,1]
        for i in range(len(label)):
            if label[i] == -1:
                label[i] = 0
        label = label.astype(float)
        label = torch.from_numpy(label)
        self.label = F.one_hot(label.to(torch.int64),2)
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

        if remove_exif:
            for name in glob.glob(os.path.join(path, img_format)):
                try:
                    piexif.remove(name)
                except Exception:
                    continue
        
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
        return self.dataset[idx],self.label[idx]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=16, type=int, help='batchSize')
parser.add_argument('--lr', default=0.01, type=float, help='')
parser.add_argument('--gpu', default=0, type=float, help='')
parser.add_argument('--epochs', default="501", type=int)
opt = parser.parse_args(args=[])

batch_size = opt.batch_size
learning_rate = opt.lr
num_epoches = opt.epochs

#训练和测试集预处理
train_dataset = MyDataset(path=r'data/trainimages', resize=28, Len=2500, img_type='jpg',labelpath='data/train.csv')
val_dataset = MyDataset(path=r'data/valimages', resize=28, Len=500, img_type='jpg',labelpath='data/val.csv')
#加载数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model_name = 'Densenet'
model = Densenet()
 
criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练
print('Start Training!')
torch.cuda.empty_cache()
iter = 0 #迭代次数
for epoch in range(num_epoches):
    for img, label in train_loader:
        label = label.to(torch.float)        
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter+=1
        if iter%200 == 0:
            print('epoch: {}, iter:{}, loss: {:.4}'.format(epoch, iter, loss.data.item()))

    if (epoch%50 == 0) & (epoch != 0):
        i = epoch/50
        torch.save(model, model_name + '_%03d.pth'% i)

        # 评估
        print('Start eval!')
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for img, label in val_loader:
            label = label.to(torch.float)  
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data.item()*label.size(0)
            mask = (out == out.max(dim=1, keepdim=True)[0]).to(dtype=torch.float)
            num_correct = (mask == label).sum()/2
            eval_acc += num_correct.item()

        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_dataset)), eval_acc / (len(val_dataset))))