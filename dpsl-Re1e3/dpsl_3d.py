from pathlib import Path
import torch
import h5py

from neuralop.datasets import UnitGaussianNormalizer, TensorDataset
from neuralop.datasets import DefaultDataProcessor, PositionalEmbedding2D


def load_dpsl_3d(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    T_in=10, T=10,
    test_resolutions=[64],
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    """Loads a small Darcy-Flow dataset

    Training contains 1000 samples in resolution 16x16.
    Testing contains 100 samples at resolution 16x16 and
    50 samples at resolution 32x32.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    test_resolutions : int list, default is [64],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : torch DataLoader
    testing_dataloaders : dict (key: DataLoader)
    """
    for res in test_resolutions:
        if res not in [16, 32, 64]:
            raise ValueError(
                f"Only 32 and 64 are supported for test resolution, "
                f"but got test_resolutions={test_resolutions}"
            )
    path = Path(__file__).resolve().parent.joinpath("data")
    return load_dpsl_3d_pt(
        str(path),
        n_train=n_train,
        n_tests=n_tests,
        T_in=T_in, T=T,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        test_resolutions=test_resolutions,
        train_resolution=64,
        grid_boundaries=grid_boundaries,
        positional_encoding=positional_encoding,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
    )


def load_dpsl_3d_pt(
    data_path,
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    T_in=10, T=10,
    test_resolutions=[64],
    train_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    """Load the DPSL 3D dataset"""
    TRAIN_PATH = '/work/atif/datasets/dpsl_Re1e3_dx_64/dpsl_data_N10000_T51_ux_uy_vort.h5'
    ftrain = h5py.File(TRAIN_PATH, 'r')
    x_train = torch.tensor(ftrain['vortz'][0:n_train,0:T_in,:,:]).unsqueeze(channel_dim).type(torch.float32).clone()
    y_train = torch.tensor(ftrain['vortz'][0:n_train,T_in:T_in+T,:,:]).unsqueeze(channel_dim).type(torch.float32).clone()
    # unsqueeze uplifts torch.Size([1000, 16, 16]) to torch.Size([1000, 1, 16, 16])
    print(f"\n### x_train shape {x_train.shape}")

    idx = test_resolutions.index(train_resolution)     # 0
    test_resolutions.pop(idx)                          
    n_test = n_tests.pop(idx)                          # 100
    test_batch_size = test_batch_sizes.pop(idx)        # 32

    x_test  = torch.tensor(ftrain['vortz'][-n_test:,0:T_in,:,:]).unsqueeze(channel_dim).type(torch.float32).clone()
    y_test  = torch.tensor(ftrain['vortz'][-n_test:,T_in:T_in+T,:,:]).unsqueeze(channel_dim).type(torch.float32).clone()
 
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        #y_train = output_encoder.transform(y_train)
    else:
        output_encoder = None
 
    train_db = TensorDataset(
        x_train,
        y_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_db = TensorDataset(
        x_test,
        y_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {train_resolution: test_loader}
    print(f"\n\n test loader loaders {test_loader} {test_loaders}\n\n")
    
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    print(f"\n\n dataprocessor {input_encoder} {output_encoder} {pos_encoding} \n\n")
    return train_loader, test_loaders, data_processor

# # # Functions to support input output time loading via channels

def load_dpsl_3d_channels(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    T_in=10, T=10,
    test_resolutions=[64],
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    """Loads a small Darcy-Flow dataset

    Training contains 1000 samples in resolution 16x16.
    Testing contains 100 samples at resolution 16x16 and
    50 samples at resolution 32x32.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    test_resolutions : int list, default is [64],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : torch DataLoader
    testing_dataloaders : dict (key: DataLoader)
    """
    for res in test_resolutions:
        if res not in [16, 32, 64]:
            raise ValueError(
                f"Only 32 and 64 are supported for test resolution, "
                f"but got test_resolutions={test_resolutions}"
            )
    path = Path(__file__).resolve().parent.joinpath("data")
    return load_dpsl_3d_channels_pt(
        str(path),
        n_train=n_train,
        n_tests=n_tests,
        T_in=T_in, T=T,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        test_resolutions=test_resolutions,
        train_resolution=64,
        grid_boundaries=grid_boundaries,
        positional_encoding=positional_encoding,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
    )


def load_dpsl_3d_channels_pt(
    data_path,
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    T_in=10, T=10,
    test_resolutions=[64],
    train_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    """Load the DPSL 3D dataset"""
    TRAIN_PATH = '/work/atif/datasets/dpsl_Re1e3_dx_64/dpsl_data_N10000_T51_ux_uy_vort.h5'
    ftrain = h5py.File(TRAIN_PATH, 'r')
    x_train = torch.tensor(ftrain['vortz'][0:n_train,0:T_in,:,:]).type(torch.float32).clone()
    y_train = torch.tensor(ftrain['vortz'][0:n_train,T_in:T_in+T,:,:]).type(torch.float32).clone()
    # unsqueeze uplifts torch.Size([1000, 16, 16]) to torch.Size([1000, 1, 16, 16])
    print(f"\n### x_train shape {x_train.shape}")

    idx = test_resolutions.index(train_resolution)     # 0
    test_resolutions.pop(idx)                          
    n_test = n_tests.pop(idx)                          # 100
    test_batch_size = test_batch_sizes.pop(idx)        # 32

    x_test  = torch.tensor(ftrain['vortz'][-n_test:,0:T_in,:,:]).type(torch.float32).clone()
    y_test  = torch.tensor(ftrain['vortz'][-n_test:,T_in:T_in+T,:,:]).type(torch.float32).clone()
 
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        #y_train = output_encoder.transform(y_train)
    else:
        output_encoder = None
 
    train_db = TensorDataset(
        x_train,
        y_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_db = TensorDataset(
        x_test,
        y_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {train_resolution: test_loader}
    print(f"\n\n test loader loaders {test_loader} {test_loaders}\n\n")
    
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    print(f"\n\n dataprocessor {input_encoder} {output_encoder} {pos_encoding} \n\n")
    return train_loader, test_loaders, data_processor
