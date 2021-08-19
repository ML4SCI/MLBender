"""
This file has two main components.
1. Function `fixed_to_table`, which converts the fixed form h5 to table form h5.
Usage: fixed_to_table(example.h5)
2. Class CustomTrainLoaderLHC, this class is an optimized trainloader that works on table form h5 files.
USage: dataloader = CustomTrainLoaderLHC('some_table_hdf.h5')
"""
import pandas as pd
import torch


def fixed_to_table(fixed_file, output_file=None, chunksize=10000):
    """
    Convert fixed format hdf5 file to table format for fast random access.
    Args:
        fixed_file -> Name of the input file
        output_file -> Name of the output file (If not provided, will use the input file_name with 'table' appended to its
    name)
        chunksize -> Number of rows to process at a time (Depends on the amount of RAM available). (Default: 10000)
    """
    if output_file is None:
        file_name, file_extension = fixed_file.split('.')
        output_file = file_name + '_table.' + file_extension

    store = pd.HDFStore(fixed_file)

    nrows = store.get_storer('df').shape[0]

    i = 0
    while i < nrows:
        al_df = store.select(key='df', start=i, stop=i + chunksize).astype('float32')
        if i == 0:
            al_df.to_hdf(output_file, 'df', mode='w', format='table')
        else:
            al_df.to_hdf(output_file, 'df', mode='a', append=True, format='table')
        i += chunksize

    store.close()

    return output_file


class LHCAnomalyDataset(torch.utils.data.Dataset):
    """
    LHC 2020 R and D dataset for Anomaly Detection

    Usage:
    dataset = LHCAnomalyDataset(path_to_hdf_file)

    Methods:
    len(dataset) -> Returns the number of rows in the dataset
    dataset[i] -> Returns the i-th row in the dataset.
    """

    def __init__(self, hdf_file):
        """
        Args:
            hdf_file_file (string): Path to the h5 file with binary label.
        """
        self.hdf_store = pd.HDFStore(hdf_file, mode='r')

    def __len__(self):
        return self.hdf_store.get_storer('df').nrows

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            idx = [idx]

        fetched_data = torch.tensor(pd.read_hdf(self.hdf_store, 'df', where=pd.Index(idx)).values)

        return fetched_data[:, :-1], fetched_data[:, -1]

    def __del__(self):
        self.hdf_store.close()


class CustomTrainLoaderLHC:
    """
    Custom DataLoader class which speeds up batching and shuffling.
    - Not tested on multi-threads yet.

    Usage:
    dataloader = CustomTrainLoaderLHC(hdf_file_path, batch_size, shuffle=True)

    Methods:
    Simply use as an iterator, e.g.
    for data, labels in dataloader:
        pass
    """
    def __init__(self, file_name, batch_size=500, shuffle=True):
        self.ds = LHCAnomalyDataset(file_name)
        if shuffle:
            inner_sampler = torch.utils.data.sampler.RandomSampler(self.ds)
        else:
            inner_sampler = torch.utils.data.sampler.SequentialSampler(self.ds)
        self.sampler = torch.utils.data.sampler.BatchSampler(
            inner_sampler,
            batch_size=batch_size,
            drop_last=False)
        self.sampler_iter = iter(self.sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            indices = next(self.sampler_iter)
        except StopIteration:
            self.sampler_iter = iter(self.sampler)
            raise StopIteration
        return self.ds[indices]

    def __len__(self):
        return len(self.sampler)


@torch.no_grad()
def normalize_data(batch_data, batch_std=None, batch_mean=None):
    """
    Normalize data using Standard Normalizer.

    Usage:
    normalized_data, batch_std, batch_mean = normalize_data(batch_data)

    Args:
        batch_data -> data to normalize
        batch_std -> Standard Deviation (If None, calculated automatically)
        batch_mean -> Mean (Average) (If None, calculated automatically)
    """
    if batch_std is None:
        batch_std = torch.std(batch_data, 0)
    if batch_mean is None:
        batch_mean = torch.mean(batch_data, 0)
    batch_data = (batch_data - batch_mean) / batch_std
    batch_data = torch.nan_to_num(batch_data)

    return batch_data, batch_std, batch_mean


@torch.no_grad()
def denormalize_data(batch_data, batch_std, batch_mean):
    """
    Denormalize data given the standard deviation and mean.
    """
    batch_data = (batch_data * batch_std) + batch_mean

    return batch_data


@torch.no_grad()
def dequantize_data(batch_data, batch_max=None, batch_min=None):
    """
    Dequantize data using random noise and logit function.
    It uses Max min to first change the range of the data to [0, 1].

    Usage:
    dequantized_data, batch_max, batch_min, rand_noise = dequantize_data(batch_data)

    Args:
        batch_data -> Data to dequantize
        batch_max -> Maximum value in the batch (one for each feature) (Calculated automatically if not provided).
        batch_min -> Minimum Value (feature wise) (Calculated automatically if not provided).
    """
    if batch_max is None:
        batch_max = batch_data.max(0)[0]
    if batch_min is None:
        batch_min = batch_data.min(0)[0]
    ret_data = (torch.nan_to_num((batch_data - batch_min) / (batch_max - batch_min)) * 255.0)
    rand_noise = torch.rand_like(ret_data)
    ret_data = ret_data + rand_noise
    ret_data = ret_data / 255.0
    ret_data = torch.logit(ret_data, eps=1e-6)

    return ret_data, batch_max, batch_min, rand_noise


@torch.no_grad()
def reverse_dequantize_data(batch_data, batch_max, batch_min, rand_noise=None):
    """
    Convert the data back to original form.

    Args:
        rand_noise: Random Noise to subtract from the data (Ignored if not provided).
    """
    ret_data = torch.sigmoid(batch_data)
    if rand_noise is not None:
        ret_data *= 255.0
        ret_data -= rand_noise
        ret_data /= 255.0
    ret_data = ret_data * (batch_max - batch_min) + batch_min

    return ret_data

if __name__ == '__main__':
    dl = CustomTrainLoaderLHC('Datasets/events_anomalydetection_tiny_table.h5', shuffle=False)
    # n_epochs = 2
    # for epoch in range(n_epochs):
    #     print(f'Epochs: {epoch}')
    #     for i, (data, label) in enumerate(dl):
    #         print('LOL')
    #         if i == 0:
    #             print(data[0])
    data, _ = next(iter(dl))
    print(data)
    _, max_, min_, rand_noise = dequantize_data(data)
    data, _, _, rand_noise = dequantize_data(data, max_, min_)
    print(reverse_dequantize_data(data, max_, min_, rand_noise))
