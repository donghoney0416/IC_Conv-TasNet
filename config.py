from pathlib import Path
from typing import Dict, Sequence, Union
from dataclasses import dataclass, field, asdict

@dataclass
class Configurations(object):
    # Model hyperparameters
    model_mic_num: int = 4
    model_ch_dim: int = 8
    model_enc_dim: int = 512
    model_feature_dim: int = 128
    model_win: int = 16
    model_layer: int = 8
    model_stack: int = 1
    model_kernel: int = 3
    model_num_spk: int = 1
    model_causal: bool = False

    # Dataset settings
    dataset_path: Dict[str, Path] = field(init=False)
    supported_dataset_names = ['chime2', 'chime3', 'chime3_new', 'FSD', 'DNS']
    reference_channel_idx: int = 0
    dns_clean_wet_use: bool = False

    # Feature configs
    sample_rate: int = 16000

    # FFT configs for plotting
    fft_size: int = 1024
    win_size: int = 512     # 512 -> 32ms
    hop_size: int = 64      # 256 -> 16ms

    # Log directory
    logdir: str = 'logs/220515_DNS_chdim8_st1_cleandry'

    # Training configs
    dataset_name: str = 'DNS'
    batch_size: int = 32
    train_shuffle: bool = True
    num_epochs: int = 100
    learning_rate: float = 1e-3

    # Device configs
    """
    'cpu', 'cuda:n', the cuda device #, or the tuple of the cuda device #.
    """
    device: Union[int, str, Sequence[str], Sequence[int]] = (4, 5, 6, 7)
    out_device: Union[int, str] = 4
    num_workers: int = 0            # should be 0 in Windows

    def __post_init__(self):
        # Dataset path settings
        self.dataset_path = dict(chime3_new=Path('./chime3_small/data/audio/16kHz/isolated'),
                                 DNS=Path('./DNS_processed'))

    def print_params(self):
        print('-------------------------')
        print('Hyper Parameter Settings')
        print('-------------------------')
        for k, v in asdict(self).items():
            print(f'{k}: {v}')
        print('-------------------------')

config = Configurations()
