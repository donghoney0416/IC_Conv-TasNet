import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple, Union

# import mir_eval
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
# from pesq import pesq
from pypesq import pesq
from pystoi import stoi

import data_manager
from config import config
from utils import print_to_file

from utility import sdr

import matplotlib.pyplot as plt
import train
import audio_utils

" setting parameters "
config.train_shuffle = False
dataset = 'test'


def run_test(runner, dataloader):
    eval_vector     = torch.empty(0).cuda(runner.in_device)
    PESQ_vector     = torch.empty(0).cuda(runner.in_device)
    STOI_vector = torch.empty(0).cuda(runner.in_device)
    SDR_vector = torch.empty(0).cuda(runner.in_device)

    pbar = tqdm(dataloader, desc='test', postfix='-', dynamic_ncols=True)
    for i_batch, (x_, y_) in enumerate(pbar):
        # data
        x = x_[0]
        y = y_[0]
        pad_size = x_[1]

        x = x.to(runner.in_device)  # B, C, F, T
        y = y.to(runner.out_device)  # B, T

        # forward path
        with torch.no_grad():
            out = runner.model(x).detach()

        # calculate SDR
        SDR = sdr.SDR_torch(out, y, mask=None)  # SDR
        SDR_vector = torch.cat((SDR_vector, SDR))

        L = y.size(1)
        y = y.cpu()
        y = y.detach().numpy()
        out = out.cpu()
        out = out.detach().numpy()

        PESQ = []
        STOI = []
        # calculate loss
        for i in range(L):
            PESQ_unit = pesq(y[i][0], out[i][0], 16000)
            # PESQ_unit = pesq(fs=16000, ref=y[i][0], deg=out[i][0], mode='wb')
            PESQ.append(PESQ_unit)

            STOI_unit = stoi(y[i][0], out[i][0], 16000, extended=False)
            STOI.append(STOI_unit)

        PESQ = torch.Tensor(PESQ)
        device = torch.device("cuda")
        PESQ = PESQ.to(device)
        PESQ_vector = torch.cat((PESQ_vector, PESQ))

        STOI = torch.Tensor(STOI)
        device = torch.device("cuda")
        STOI = STOI.to(device)
        STOI_vector = torch.cat((STOI_vector, STOI))

    return PESQ_vector, STOI_vector, SDR_vector

def draw_histogram(vector, bins):
    fig = plt.figure(figsize=(12, 6))
    plt.hist(vector, bins=bins, alpha=0.75)

    return fig


def main():
    # get data loader
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(config)

    runner = train.Runner(config)
    runner.model.eval()

    if dataset == 'train':
        data_loader = train_loader
    elif dataset == 'valid':
        data_loader = valid_loader
    else:
        data_loader = test_loader

    # load parameters
    state_dict = torch.load(Path(runner.writer.logdir, 'max.pt'), map_location='cpu')
    if isinstance(runner.model, nn.DataParallel):
        runner.model.module.load_state_dict(state_dict)
    else:
        runner.model.load_state_dict(state_dict)


    # calculate loss
    PESQ_vector, STOI_vector, SDR_vector = run_test(runner, data_loader)
    print(f'PESQ ={torch.mean(PESQ_vector)}')
    print(f'STOI ={torch.mean(STOI_vector)}')
    print(f'SDR ={torch.mean(SDR_vector)}')
    idx = torch.argsort(PESQ_vector, descending=False)       # descending order index

    # setting the log path
    path_test_result = Path(runner.writer.logdir, f'test_{dataset}_dataset') # original
    writer = SummaryWriter(logdir=path_test_result)
    os.makedirs(path_test_result, exist_ok=True)

    # start writing
    print('writing...')

    # draw histogram
    fig = draw_histogram(PESQ_vector.cpu().numpy(), bins=100)
    writer.add_figure('Loss Histogram', fig, 0)

    fig = draw_histogram(STOI_vector.cpu().numpy(), bins=100)
    writer.add_figure('Loss Histogram', fig, 0)

    fig = draw_histogram(SDR_vector.cpu().numpy(), bins=100)
    writer.add_figure('SDR Histogram', fig, 0)

    # test and writing for worst case
    idx_l = len(idx)

    # # wirte small part of data
    for (rank, i) in enumerate(idx[:30]):
        # if rank < n_worst_loss or rank > len(idx)-1-n_best_loss:        # for audios that have high loss and low loss
        audio_utils.test_writing(runner, data_loader, i, 'test', 0, writer, rank)

    for (rank, i) in enumerate(idx[-30:]):
        # if rank < n_worst_loss or rank > len(idx)-1-n_best_loss:        # for audios that have high loss and low loss
        audio_utils.test_writing(runner, data_loader, i, 'test', 0, writer, rank+idx_l-30)

    for (rank, i) in enumerate(idx[idx_l//2-15 :idx_l//2+15]):
        # if rank < n_worst_loss or rank > len(idx)-1-n_best_loss:        # for audios that have high loss and low loss
        audio_utils.test_writing(runner, data_loader, i, 'test', 0, writer, rank+idx_l//2-15)

    writer.close()


if __name__ == '__main__':
    main()
