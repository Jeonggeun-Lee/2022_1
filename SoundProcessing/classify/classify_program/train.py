from piano_roll_dataset import DS
from torch.utils.data import DataLoader
from config import config
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm
from model import Classifier
import numpy as np


batch_size = config['batch_size']
piano_roll_train = DS(config['x_train_path'], config['y_train_path'])
piano_roll_val = DS(config['x_val_path'], config['y_val_path'])
piano_roll_test = DS(config['x_test_path'], config['y_test_path'])

train_loader = DataLoader(piano_roll_train, batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(piano_roll_val, batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(piano_roll_test, batch_size, shuffle=False, drop_last=True)


def eval_net(net, data_loader, device=config['device']):
    net.eval()
    net = net.type(torch.float).to(device)
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.type(torch.float).to(device)
        y = y.to(torch.int8).to(device)
        with torch.no_grad():
            y_pred = net(x).argmax(1).to(device)
        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)#.to(device)
    ypreds = torch.cat(ypreds)#.to(device)

    acc = (ys == ypreds).float().sum()/len(ys)
    return acc.item()


def train_net(net,
              train_loader,
              val_loader,
              optimizer_cls=optim.Adam,
              loss_fn=nn.CrossEntropyLoss(),
              n_iter=config['num_epochs'],
              device=config['device']):
    net = net.type(torch.float).to(device)
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    loss_fn = loss_fn.to(device)
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        net = net.type(torch.float).to(device)
        n = 0
        n_acc = 0

        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.type(torch.float).to(device)
            yy = yy.to(torch.int64)
            yy = F.one_hot(yy, num_classes=3)
            yy = yy.type(torch.float).to(device)
            h = net(xx)
            h = h.to(device)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            y_pred = h.argmax(1)
            y_pred = y_pred.type(torch.int8).to(device)
            yy = yy.argmax(1).to(device)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss/i)
        train_acc.append(n_acc/n)

        val_acc.append(eval_net(net, val_loader, device))
        print(epoch+1, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)
        np.save('./log/train_losses', train_losses)
        np.save('./log/train_acc', train_acc)
        np.save('./log/val_acc', val_acc)
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), config['parameter_path']+'/'+str(epoch+1)+'epochs', pickle_protocol=4)
if __name__=='__main__':
    net = Classifier()
    train_net(net, train_loader, val_loader)
    print('test_acc: ', eval_net(net, test_loader))