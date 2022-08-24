import tqdm
import numpy as np
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from math import sqrt
import os


from dataset import SteelDataset
from model import NetModule

class CheckSaver():
    def __init__(self, save_path):
        self.save_path = save_path
    def save(self, model, epoch_i):
        torch.save(model.state_dict(), os.path.join(self.save_path, f'epoch{epoch_i}.pt'))
        return

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (x, y) in enumerate(loader):
        # print('x',x.shape)
        # print('y',y.shape)
        x, y = x.to(device), y.to(device)
        pred_y = model(x)

        loss = criterion(pred_y, y.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    model.eval()

    y_list = []
    pred_list = []
    mse_list = []
    mae_list = []

    with torch.no_grad():
        loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for x, y in loader:
            # print(x.shape)
            # print(y.shape)
            x, y = x.to(device), y.to(device)
            pred_y = model(x)

            y_list.extend(y.tolist())
            pred_list.extend(pred_y.tolist())

            loss_mse = torch.nn.functional.mse_loss(pred_y, y.float(), reduction='none')
            loss_mae = torch.nn.functional.l1_loss(pred_y, y.float(), reduction='none')
            mse_list.extend(loss_mse.tolist())
            mae_list.extend(loss_mae.tolist())


    mse = np.array(mse_list).mean()
    rmse = sqrt(mse)
    mae = np.array(mae_list).mean()
    var = np.array(y_list).var()
    r2 = 1-mse/var
    return rmse, mae, r2


def main(task,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)

    train_dataset = SteelDataset('./data/train.csv',task)
    test_dataset = SteelDataset('./data/test.csv',task)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = NetModule(train_dataset.x_num, dims=(1024,512,256,128,64), dropout=0.2).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    saver = CheckSaver(save_path=save_dir)

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        rmse, mae, r2 = test(model, test_data_loader, device)
        print('epoch:', epoch_i,'task:',task)
        print('RMSE {}, MAE {}, R2 {}'.format(rmse, mae, r2))
        saver.save(model, epoch_i)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    main(args.task,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)