import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import random

import audio_utils
import conv_tasnet_ic
import data_manager
from config import config
from utils import print_to_file

from utility import sdr


class Runner(object):
    """
    Wrapper class to run model
    """
    def __init__(self, config):
        print("Initializing model...")
        self.model = conv_tasnet_ic.TasNet()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'\tTotal param #: {total_params}')
        print(f'\tBatch size: {config.batch_size}')
        print(f'\tlogdir: {config.logdir}')

        print("Initializing logger, optimizer and losses...")
        self.writer = SummaryWriter(logdir=config.logdir)
        self.criterion = sdr.negative_SDR()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.writer.add_text('Text', 'Parameter #: ' + str(total_params), 0)
        self.writer.add_text('Text', 'Batch size: ' + str(config.batch_size), 0)

        print("Initializing devices...")
        self._init_device(config.device, config.out_device)

        print("Saving configurations...")
        save_config_path = Path(self.writer.logdir, 'config.txt')
        if not save_config_path.exists():
            print_to_file(save_config_path, config.print_params)

    def _init_device(self, device, out_device):
        if device == 'cpu':
            self.in_device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return

        # device type: List[int]
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device[-1])]
        else:  # sequence of devices
            if type(device[0]) != int:
                device = [int(d[-1]) for d in device]

        self.in_device = torch.device(f'cuda:{device[0]}')

        if len(device) > 1:
            if type(out_device) == int:
                self.out_device = torch.device(f'cuda:{out_device}')
            else:
                self.out_device = torch.device(out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])

            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=device,
                                               output_device=self.out_device)
        else:
            self.out_device = self.in_device
            self.str_device = str(self.in_device)

        self.model.cuda(self.in_device)
        self.criterion.cuda(self.out_device)

        torch.cuda.set_device(self.in_device)

    # Running model for train, test and validation.
    def run(self, dataloader, stage: str, epoch: int):
        self.model.train() if stage == 'train' else self.model.eval()

        avg_loss = 0.
        avg_eval = 0.
        for i_batch, (x_, y_) in enumerate(tqdm(dataloader, desc=f'{stage}: {epoch:3d}', dynamic_ncols=True)):
            mixed_input = x_[0]
            ground_truth = y_[0]

            mixed_input = mixed_input.to(self.in_device)        # B, C, T
            ground_truth = ground_truth.to(self.out_device)     # B, C, T

            if stage == 'train':
                reconstructed = self.model(mixed_input)         # B, C, 1, T
            else:
                with torch.no_grad():
                    reconstructed = self.model(mixed_input)

            loss = self.criterion(reconstructed, ground_truth)
            with torch.no_grad():
                eval_result = sdr.batch_SI_SDRi_torch(mixture=mixed_input[:, config.reference_channel_idx],
                                                      estimation=reconstructed,
                                                      origin=ground_truth)
            if stage == 'train':
                self.optimizer.zero_grad()  # make all gradients zero
                loss.backward()             # calculate all gradients
                self.optimizer.step()       # update parameters using gradients
                loss = loss.item()
            elif stage == 'valid':
                loss = loss.item()
            else:
                pass
            avg_loss += loss
            avg_eval += eval_result

        avg_loss = avg_loss / len(dataloader.dataset)
        avg_eval = avg_eval / len(dataloader.dataset)
        print(f"[Loss: {avg_loss}]\n")

        return avg_loss, avg_eval


def trainer():
    # data number that will be plotted in training
    dat_num = 7

    print("Initializing data loaders...")
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(config)

    # Initializing runner object
    # It holds a model, optimizer, etc.!
    runner = Runner(config)

    # Setting the layout of custom scalars
    dict_custom_scalars = dict(loss=['Multiline', ['loss/train', 'loss/valid']],
                               eval=['Multiline', ['eval/train', 'eval/valid']])
    runner.writer.add_custom_scalars(dict(training=dict_custom_scalars))

    print(f'Start training on {runner.str_device}...\n')
    for epoch in range(config.num_epochs):
        # Training
        train_loss, train_eval = runner.run(train_loader, 'train', epoch)
        runner.writer.add_scalar('loss/train', train_loss, epoch)
        runner.writer.add_scalar('eval/train', train_eval, epoch)

        # Record the mixture audio and the separated audio in trainning data
        audio_utils.train_writing(runner, train_loader, dat_num, 'train', epoch, runner.writer)

        # Validation
        valid_loss, valid_eval = runner.run(valid_loader, 'valid', epoch)
        runner.writer.add_scalar('loss/valid', valid_loss, epoch)
        runner.writer.add_scalar('eval/valid', valid_eval, epoch)

        # Record the mixture audio and separated audio in validset
        audio_utils.train_writing(runner, valid_loader, dat_num, 'valid', epoch, runner.writer)

        # Saving the model
        if isinstance(runner.model, torch.nn.DataParallel):
            state_dict = runner.model.module.state_dict()
        else:
            state_dict = runner.model.state_dict()

        if epoch == 0:
            min_valid_loss = valid_loss
        if min_valid_loss > valid_loss or epoch == 0:
            min_valid_loss = valid_loss
            torch.save(state_dict, Path(runner.writer.logdir, 'max.pt'))

        # Text-logging the loss for each epoch
        with open(config.logdir + '/loss.txt', 'a') as f:
            f.write(str(epoch) + ': ' + str(float(min_valid_loss)) + '\n')
    print('Training finished!')
    runner.writer.close()


if __name__ == '__main__':
    # Fix random seeds
    random_seed = 365
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if type(config.device) == int:
        if len([config.device]) > 1:
            torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Overwriting warning
    if list(Path(config.logdir).glob('events.out.tfevents.*')):
        while True:
            s = input(f'WARNING: "{config.logdir}" already has tfevents. Continue? (y/n)\n')
            if s.lower() == 'y':
                shutil.rmtree(config.logdir)
                os.makedirs(config.logdir)
                break
            elif s.lower() == 'n':
                exit()
    trainer()
