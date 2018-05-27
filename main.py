import torch
from torch.utils.data import DataLoader
from torch import nn
from camvid_dataset import CamVidDataset
from segnet import SegNet
import time
import matplotlib.pyplot as plt
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('--mode', choices=['train', 'val', 'test'])
parse.add_argument('--batch_size', '-b', type=int, default=16)
parse.add_argument('--resume', type=bool, default=False)
parse.add_argument('--epochs', type=int, default=33)
args = parse.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SegNet()
if args.mode == 'train':
    model.load_state_dict(torch.load('transfer-vgg16-for11classes.pth'))
else:
    model.load_state_dict(torch.load('segnet_weight_11classes.pth'))

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)


def train(epochs):
    model.train()
    train_dataset = CamVidDataset(phase='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # median frequency balancing
    class_loss_weight = torch.Tensor([0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823,
                                      6.2478, 7.3614, 0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_loss_weight)
    min_loss = float('inf')
    for i in range(epochs):
        start_time = time.time()
        mean_loss = 0
        idx = 0
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            mean_loss += loss
            idx += 1
        mean_loss /= idx
        end_time = time.time()
        elapse_time = end_time - start_time
        print(f'epoch {i} loss: {mean_loss}, elapse time: {elapse_time}')
        if mean_loss < min_loss:
            print(f'in epoch {i}, loss decline')
            min_loss = mean_loss
            state_dict = model.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
                torch.save(state_dict, 'segnet_weight_11classes.pth')

def val():
    model.eval()
    val_dataset = CamVidDataset(phase='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    for idx, (img, label) in enumerate(val_loader):
        img = img.to(device)
        label = label.cpu().numpy()
        label = label.squeeze()
        output = model(img)
        _, predict = torch.max(output, dim=1)
        pred = predict.cpu().numpy()
        name = val_dataset.raw_images[idx]
        pred = pred.squeeze()
        plt.imsave(f'./CamVid600/mask_light/{name}', label)
        plt.imsave(f'./CamVid600/val_result/{name}', pred)


if __name__ == '__main__':
    if args.mode == 'train':
        train(args.epochs)
    elif args.mode == 'val':
        val()