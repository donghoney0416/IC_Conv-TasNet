import os
from hparams import hparams
import data_manager
from pathlib import Path
from tensorboardX import SummaryWriter
from utility import sdr

import numpy as np
import torch
import torch.nn as nn

import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf



def spectrogram(wave, ref=1.0):
    S = abs(librosa.core.stft(wave, n_fft=hparams.fft_size, hop_length=hparams.hop_size,
                              win_length=hparams.win_size, window='hann'))
    # normalize modification
    S = S / (S.max() + 1e-3)
    logS = librosa.core.amplitude_to_db(S, ref=ref)
    return logS


def draw_spectrogram(wave, ref=1.0, max_val=None, loss=None):
    """
    wave -> spectrogram figure

    """
    logS = spectrogram(wave, ref)
    fig = plt.figure(figsize=(12, 4))
    librosa.display.specshow(logS, sr=hparams.sample_rate, hop_length=hparams.hop_size, x_axis='time',
                             y_axis='linear', cmap='magma')
    if loss == None:
        plt.title('power spectrogram')
    else:
        plt.title(f'power spectrogram [Loss = {loss}]')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    if max_val is not None:
        plt.clim(max_val-80.0, max_val)
    return fig


def torch_load(runner, epoch):
    # state_dict = torch.load(Path(runner.writer.logdir, f'{epoch}.pt'), map_location='cpu')
    state_dict = torch.load(Path(runner.writer.logdir, '99.pt'), map_location='cpu')
    if isinstance(runner.model, nn.DataParallel):
        runner.model.module.load_state_dict(state_dict)
    else:
        runner.model.load_state_dict(state_dict)
    return 0

def test_writing(runner, dataloader, dat_num, mode: str, epoch, writer, rank=None):
    with torch.no_grad():
        mixture, target = dataloader.dataset.__getitem__(dat_num)
        mixture = mixture.unsqueeze(0).to(runner.in_device)
        target = target.unsqueeze(0).to(runner.in_device)

        # if mode == 'test':
        #    torch_load(runner, epoch)

        separated = runner.model(mixture).detach()
        loss = runner.criterion(separated, target)
        loss = loss.round()
        _, nsource, nsample = separated.shape


        mixture = mixture.squeeze(0).cpu()
        mixture = mixture[0]
        target = target.squeeze().cpu()
        separated = separated.squeeze().cpu()

        mixture = np.asfortranarray(mixture.numpy())
        target = target.numpy()
        separated = separated.numpy()

        if rank is not None:
            write_name = f'{rank}'.zfill(5) + 'th_' + mode + f'{dat_num}_Tot.Loss_{loss}'
        else:
            write_name = mode + f'{dat_num}_Tot.Loss_{loss}'


        # write audio
        writer.add_audio(write_name + '/mixture', mixture, epoch, sample_rate=hparams.sample_rate)

        writer.add_audio(write_name + f'/target_sources/source_loss_{loss}', target, epoch,
                              sample_rate=hparams.sample_rate)
        writer.add_audio(write_name + f'/separated_sources/source_loss_{loss}', separated, epoch,
                              sample_rate=hparams.sample_rate)


        # calculate the maximum value of mixture_STFT
        mix_logS = spectrogram(mixture)
        max = mix_logS.max()

        # calculate and write mixture STFT
        fig = draw_spectrogram(mixture, max_val=max)
        writer.add_figure(write_name + '/mixture_stft', fig, epoch)

        fig = draw_spectrogram(target, max_val=max)
        writer.add_figure(write_name + f'/target_sources_stft/source_loss_{loss}', fig, epoch)

        fig = draw_spectrogram(separated, max_val=max, loss=loss)
        writer.add_figure(write_name + f'/separated_sources_stft/source_loss_{loss}', fig, epoch)

        # scaling modification
        amp_sep = np.sqrt((separated ** 2).sum())
        amp_mix = np.sqrt((mixture ** 2).sum())
        amp_tar = np.sqrt((target ** 2).sum())

        scaled_separated = separated * (10 / (amp_sep + 1e-3))
        scaled_mixture = mixture * (10 / (amp_mix + 1e-3))
        scaled_target = target * (10 / (amp_tar + 1e-3))

        # save wav file
        wav_dir = hparams.logdir + '/wav_file/' + write_name
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
        # librosa.output.write_wav(wav_dir + '/separated.wav', scaled_separated, hparams.sample_rate)
        # librosa.output.write_wav(wav_dir + '/mixture.wav', scaled_mixture, hparams.sample_rate)
        # librosa.output.write_wav(wav_dir + '/target.wav', scaled_target, hparams.sample_rate)
        sf.write(wav_dir + '/separated.wav', scaled_separated, hparams.sample_rate)
        sf.write(wav_dir + '/mixture.wav', scaled_mixture, hparams.sample_rate)
        sf.write(wav_dir + '/target.wav', scaled_target, hparams.sample_rate)

def train_writing(runner, dataloader, dat_num, mode: str, epoch, writer, rank=None):
    with torch.no_grad():
        mixture, target = dataloader.dataset.__getitem__(dat_num)
        mixture = mixture.unsqueeze(0).to(runner.in_device)
        target = target.unsqueeze(0).to(runner.in_device)

        # if mode == 'test':
        #    torch_load(runner, epoch)

        separated = runner.model(mixture).detach()
        loss = runner.criterion(separated, target)
        loss = loss.round()
        _, nsource, nsample = separated.shape

        mixture = mixture.squeeze(0).cpu()
        separated = separated.squeeze(0).cpu().numpy()
        target = target.squeeze(0).cpu()

        mix_mean = mixture.mean(dim=0)

        # perm_separated = perm_separated.squeeze(0).cpu()
        #
        # mixture_copy = torch.tensor([mixture.tolist() for _ in range(nsource)])
        #
        # # perm_loss = sdr.calc_logmse_torch_with_bias(perm_separated, target).round()
        # perm_loss = sdr.calc_logmse_torch_with_bias(perm_separated, target, mixture_copy).round()


        mix_mean = mix_mean.numpy()
        target = target.numpy()
        # perm_separated = perm_separated.numpy()


        if rank is not None:
            write_name = f'{rank}th_' + mode + f'{dat_num}_Tot.Loss_{loss}'
        else:
            write_name = mode + f'{dat_num}'

        writer.add_audio(write_name + '/mixture', mix_mean, epoch, sample_rate=hparams.sample_rate)

        for i in range(nsource):
            writer.add_audio(write_name + f'/target_sources/source{i}', target[i], epoch,
                                  sample_rate=hparams.sample_rate)
            writer.add_audio(write_name + f'/separated_sources/source{i}', separated[i], epoch,
                                  sample_rate=hparams.sample_rate)


        # calculate the maximum value of mixture_STFT
        mix_logS = spectrogram(mix_mean)
        max = mix_logS.max()

        # calculate and write mixture STFT
        fig = draw_spectrogram(mix_mean, max_val=max)
        writer.add_figure(write_name + '/mixture_stft', fig, epoch)


        for i in range(nsource):
            fig = draw_spectrogram(target[i], max_val=max)
            writer.add_figure(write_name + f'/target_sources_stft/source{i}', fig, epoch)

            fig = draw_spectrogram(separated[i], max_val=max, loss=loss)
            writer.add_figure(write_name + f'/separated_sources_stft/source{i}', fig, epoch)



if __name__ == '__main__':
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    wave = train_loader.dataset.__getitem__(0)[0].numpy()

    # plt.figure()

    fig = draw_spectrogram(wave)
    plt.show()

    writer = SummaryWriter(logdir='./runs/logmse_smalldata/train_0')


    wave = train_loader.dataset.__getitem__(1)[0].numpy()

    fig = draw_spectrogram(wave)
    plt.show()

    writer.close()

    # plt.plot(fig)

