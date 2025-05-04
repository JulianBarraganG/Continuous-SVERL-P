from copy import deepcopy
from tqdm import tqdm
from sys import stderr

from dataclasses import dataclass
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import compute_normalization
from .imputation_networks import get_imputation_networks
from .VAEAC import VAEAC

@dataclass
class TrainingArgs:
    epochs: int = 10
    validation_ratio: float = 0.2
    validations_per_epoch: int = 1
    validation_iwae_num_samples: int = 25
    num_imputations: int = 5
    use_last_checkpoint: bool = False


def extend_batch(batch, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    while batch.shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch = next(dataloader_iterator)
        if nw_batch.shape[0] + batch.shape[0] > batch_size:
            nw_batch = nw_batch[:batch_size - batch.shape[0]]
        batch = torch.cat([batch, nw_batch], 0)
    return batch


def extend_batch_tuple(batch, dataloader, batch_size):
    """
    The same as extend_batch, but here the batch is a list of tensors
    to be extended. All tensors are assumed to have the same first dimension.
    Returns the extended batch (i. e. list of extended tensors).
    """
    while batch[0].shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch = next(dataloader_iterator)
        if nw_batch[0].shape[0] + batch[0].shape[0] > batch_size:
            nw_batch = [nw_t[:batch_size - batch[0].shape[0]]
                        for nw_t in nw_batch]
        batch = [torch.cat([t, nw_t], 0) for t, nw_t in zip(batch, nw_batch)]
    return batch


def get_validation_iwae(val_dataloader, mask_generator, batch_size,
                        model, num_samples, verbose=False):
    """
    Compute mean IWAE log likelihood estimation of the validation set.
    Takes validation dataloader, mask generator, batch size, model (VAEAC)
    and number of IWAE latent samples per object.
    Returns one float - the estimation.
    """
    cum_size = 0
    avg_iwae = 0
    iterator = val_dataloader
    if verbose:
        iterator = tqdm(iterator)
    for batch in iterator:
        init_size = batch.shape[0]
        batch = extend_batch(batch, val_dataloader, batch_size)
        mask = mask_generator(batch)
        if next(model.parameters()).is_cuda:
            batch = batch.cuda()
            mask = mask.cuda()
        with torch.no_grad():
            iwae = model.batch_iwae(batch, mask, num_samples)[:init_size]
            avg_iwae = (avg_iwae * (cum_size / (cum_size + iwae.shape[0])) +
                        iwae.sum() / (cum_size + iwae.shape[0]))
            cum_size += iwae.shape[0]
        if verbose:
            iterator.set_description('Validation IWAE: %g' % avg_iwae)
    return float(avg_iwae)

def get_vaeac(args: TrainingArgs, one_hot_max_sizes: list, data: str | np.ndarray):
    """
    given a specification on state feature data types, and trajectory data file path,
    this function will train a vaeac model on the data, and return the vaeac model.
    parameters
    ----------
    args : TrainingArgs
    one_hot_max_sizes : list
    data : str | np.ndarray
    returns
    -------
    vaeac : vaeac
        the trained vaeac model as a pytorch module.
    """
    # read and normalize input data
    if isinstance(data, str):
        try:
            raw_data = np.loadtxt(data, delimiter=',')
        except FileNotFoundError:
            raise FileNotFoundError(
                    f"data file not found. please check the path: {data},"
                    + "and make sure you call from src directory."
                    )
    else:
        raw_data = data

    networks = get_imputation_networks(one_hot_max_sizes)

    args = TrainingArgs()

    raw_data = torch.from_numpy(raw_data).float()
    norm_mean, norm_std = compute_normalization(raw_data, one_hot_max_sizes)
    norm_std = torch.max(norm_std, torch.tensor(1e-9))
    data = (raw_data - norm_mean[None]) / norm_std[None]

    # default parameters which are not supposed to be changed from user interface
    use_cuda = torch.cuda.is_available()
    verbose = True
    # non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. it might be fixed now, but anyway there is no need
    # to have a lot of workers for dataloader over in-memory tabular data.
    num_workers = 0

    # design all necessary networks and learning parameters for the dataset

    # build vaeac on top of returned network, optimizer on top of vaeac,
    # extract optimization parameters and mask generator
    model = VAEAC(
        networks['reconstruction_log_prob'],
        networks['proposal_network'],
        networks['prior_network'],
        networks['generative_network'],
        networks['sampler'], 
        one_hot_max_sizes
    )
    if use_cuda:
        model = model.cuda()
    optimizer = networks['optimizer'](model.parameters())
    batch_size = networks['batch_size']
    mask_generator = networks['mask_generator']
    vlb_scale_factor = networks.get('vlb_scale_factor', 1)

    # train-validation split
    val_size = ceil(len(data) * args.validation_ratio)
    val_indices = np.random.choice(len(data), val_size, False)
    val_indices_set = set(val_indices)
    train_indices = [i for i in range(len(data)) if i not in val_indices_set]
    train_data = data[train_indices]
    val_data = data[val_indices]

    # initialize dataloaders
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=False)

    # number of batches after which it is time to do validation
    validation_batches = ceil(len(dataloader) / args.validations_per_epoch)

    # a list of validation IWAE estimates
    validation_iwae = []
    # a list of running variational lower bounds on the train set
    train_vlb = []
    # the length of two lists above is the same because the new
    # values are inserted into them at the validation checkpoints only

    # best model state according to the validation IWAE
    best_state = None

    # main train loop
    for epoch in range(args.epochs):

        iterator = dataloader
        avg_vlb = 0
        if verbose:
            print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
            iterator = tqdm(iterator)

        # one epoch
        for i, batch in enumerate(iterator):

            # the time to do a checkpoint is at start and end of the training
            # and after processing validation_batches batches
            if any([
                        i == 0 and epoch == 0,
                        i % validation_batches == validation_batches - 1,
                        i + 1 == len(dataloader)
                    ]):
                val_iwae = get_validation_iwae(val_dataloader, mask_generator,
                                               batch_size, model,
                                               args.validation_iwae_num_samples,
                                               verbose)
                validation_iwae.append(val_iwae)
                train_vlb.append(avg_vlb)

                # if current model validation IWAE is the best validation IWAE
                # over the history of training, the current state
                # is saved to best_state variable
                if max(validation_iwae[::-1]) <= val_iwae:
                    best_state = deepcopy({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'validation_iwae': validation_iwae,
                        'train_vlb': train_vlb,
                    })

                if verbose:
                    print(file=stderr)
                    print(file=stderr)

            # if batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch = extend_batch(batch, dataloader, batch_size)

            # generate mask and do an optimizer step over the mask and the batch
            mask = mask_generator(batch)
            optimizer.zero_grad()
            if use_cuda:
                batch = batch.cuda()
                mask = mask.cuda()
            vlb = model.batch_vlb(batch, mask).mean()
            (-vlb / vlb_scale_factor).backward()
            optimizer.step()

            # update running variational lower bound average
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
            if verbose:
                iterator.set_description('Train VLB: %g' % avg_vlb)

    # if use doesn't set use_last_checkpoint flag,
    # use the best model according to the validation IWAE
    if not args.use_last_checkpoint:
        model.load_state_dict(best_state['model_state_dict'])

    return model

