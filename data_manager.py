import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
import librosa

from copy import copy
from typing import List, Sequence, Tuple


class CustomDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given dataset path.
    """
    def __init__(self, stage: str, config):
        # Currently, only the CHiME-3 dataset is supported
        self.config = config
        self.data_channel_dim = None
        dataset_path = config.dataset_path[config.dataset_name]

        if config.dataset_name == 'chime3_new':
            self.data_channel_dim = 6
            if stage == 'train':
                stage_path = 'tr05'
            elif stage == 'valid':
                stage_path = 'dt05'
            elif stage == 'test':
                stage_path = 'et05'
            else:
                raise ValueError(f"dataset stage should be 'train', 'valid', or 'test', not {stage}." )

            # Gathering the list of files
            ref_files = glob.glob(str(dataset_path.joinpath(
                stage_path + '*' + f'simu_clean/*.CH{config.reference_channel_idx + 1}.wav')), recursive=True)
            mix_files = glob.glob(str(dataset_path.joinpath(
                stage_path + '*' + 'simu/*.wav')), recursive=True)

            # Sorting file lists for synchronization
            ref_files = list(map(lambda x: x.replace('\\', '/'), ref_files))
            ref_files = sorted(ref_files, key=lambda x: "/".join(x.split('/')[-1:]))
            mix_files = list(map(lambda x: x.replace('\\', '/'), mix_files))
            mix_files = sorted(mix_files, key=lambda x: "/".join(x.split('/')[-1:]))

            # To pair ref with mix file names
            all_files = []
            for file_idx in range(len(ref_files)):
                mix = []
                for data_channel_idx in range(self.data_channel_dim):
                    mix.append(mix_files[file_idx * self.data_channel_dim + data_channel_idx])  # use multi channel
                    # mix.append(mix_files[file_idx * self.data_channel_dim + config.reference_channel_idx])            # use single channel
                all_files.append([mix, ref_files[file_idx]])
            self.all_files = all_files
        elif config.dataset_name == 'DNS':
            self.data_channel_dim = 4
            if stage == 'train':
                stage_path = 'train'
            elif stage == 'valid':
                stage_path = 'valid'
            elif stage == 'test':
                stage_path = 'test'
            else:
                raise ValueError(f"dataset stage should be 'train', 'valid', or 'test', not {stage}." )

            # Gathering the list of files
            ref_files = glob.glob(str(dataset_path.joinpath(stage_path + f'/__clean/*.wav')), recursive=True) \
                if not config.dns_clean_wet_use else \
                glob.glob(str(dataset_path.joinpath(stage_path + f'/__clean_wet/*.wav')), recursive=True)
            mix_files = glob.glob(str(dataset_path.joinpath(stage_path + f'/__noisy/*.wav')), recursive=True)

            # Sorting file lists for synchronization
            ref_files = list(map(lambda x: x.replace('\\', '/'), ref_files))
            ref_files = sorted(ref_files, key=lambda x: "/".join(x.split('_')[-1:]))
            mix_files = list(map(lambda x: x.replace('\\', '/'), mix_files))
            mix_files = sorted(mix_files, key=lambda x: "/".join(x.split('_')[-1:]))
            self.all_files = list(map(list, zip(*[mix_files, ref_files])))
        else:
            raise ValueError(f"dataset_name should be in {config.supported_dataset_names}, "
                             f"not {config.dataset_name}.")


    def __getitem__(self, idx: int) -> Tuple:
        filename = self.all_files[idx]

        if self.config.dataset_name == 'chime3_new':
            mixed_input = []
            for ii in range(self.data_channel_dim):
                mixed_input.append(librosa.core.load(filename[0][ii], sr=None, mono=False)[0])
            mixed_input = np.stack(mixed_input, axis=0)     # multi channel
            # mixed_input = mixed_input[config.reference_channel_idx:config.reference_channel_idx+1]                  # single channel

            # reference_5ch
            reference_audio = np.expand_dims(librosa.core.load(filename[1], sr=None, mono=False)[0], axis=0)

            mixed_input = torch.from_numpy(mixed_input)
            reference_audio = torch.from_numpy(reference_audio)
        elif self.config.dataset_name == 'DNS':
            mixed_input = librosa.core.load(filename[0], sr=None, mono=False)[0]
            reference_audio = librosa.core.load(filename[1], sr=None, mono=False)[0][self.config.reference_channel_idx:self.config.reference_channel_idx+1]
            mixed_input = torch.from_numpy(mixed_input)
            reference_audio = torch.from_numpy(reference_audio)
        else:
            raise ValueError(f"dataset_name should be in {self.config.supported_dataset_names}, "
                             f"not {self.config.dataset_name}.")

        return mixed_input, reference_audio

    def __len__(self):
        len_int = len(self.all_files)
        return len_int - len_int % self.config.batch_size

    @staticmethod
    def custom_collate(batch: List[Tuple]) -> Tuple:
        # TODO: make custom collate function if needed
        """
        Puts each data field into a tensor with outer dimension batch size
        """
        def pad_size(vec, pad, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = pad - vec.size(dim)
            return pad_size

        def pad_tensor(vec, dim, pad_size):
            return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

        elem = batch[0]
        elem_type = type(elem)

        if isinstance(elem, torch.Tensor):
            max_len = max(map(lambda x: x[0].size(-1), batch))
            padding_size = list(map(lambda x: pad_size(x, pad=max_len, dim=-1), batch))
            batch = list(map(lambda x: pad_tensor(x, dim=-1, pad_size=pad_size(x, pad=max_len, dim=-1)), batch))

            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out), padding_size
        else:
            transposed = zip(*batch)
            return [CustomDataset.custom_collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        """
        Split the dataset into `len(ratio)` datasets.

        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1

        :type dataset: SALAMIDataset
        :type ratio: Sequence[float]

        :rtype: Sequence[Dataset]
        """
        if not isinstance(dataset, cls):
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(dataset.all_files),
                             dtype=int)
        result = [copy(dataset) for _ in range(n_split)]
        # all_f_per = np.random.permutation(a._all_files)

        # TODO: split data
        for ii in range(n_split):
            result[ii].all_files = dataset.all_files[idx_data[ii]:idx_data[ii + 1]]

        return result


def get_dataloader(config):
    """
    Load dataloader for train, valid, and test stage.
    """
    loader_kwargs = dict(batch_size=config.batch_size,
                         drop_last=False,
                         num_workers=config.num_workers,
                         pin_memory=True,
                         collate_fn=CustomDataset.custom_collate)

    train_set = CustomDataset(stage='train', config=config)
    valid_set = CustomDataset(stage='valid', config=config)
    test_set = CustomDataset(stage='test', config=config)

    train_loader = DataLoader(train_set, shuffle=config.train_shuffle, **loader_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader
