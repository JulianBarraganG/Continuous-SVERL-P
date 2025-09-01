from torch import nn
from torch.optim import Adam

from typing import Any

from .mask_generators import MCARGenerator
from .nn_utils import MemoryLayer, SkipConnection
from .prob_utils import CategoricalToOneHotLayer, GaussianCategoricalLoss, \
                       GaussianCategoricalSampler, SetGaussianSigmasToOne

def get_imputation_networks(one_hot_max_sizes: list[int],
                            nn_size_dict: dict[str, int],
                            ) -> dict[str, Any]:
    """
    This function builds neural networks for imputation given
    the list of one-hot max sizes of the dataset features.
    It returns a dictionary with those neural networks together with
    reconstruction log probability function, optimizer constructor,
    sampler from the generator output, mask generator, batch size,
    and scale factor for the stability of the variational lower bound
    optimization.

    Parameters
    ----------
    one_hot_max_sizes : list[int]
        List of maximum sizes for one-hot encoded features.
    nn_size_dict : dict[str, int]
        Dictionary containing neural network size parameters:
        - 'width': width of the hidden layers
        - 'depth': depth of the network (number of hidden layers)
        - 'latent_dim': dimensionality of the latent space
    Returns
    -------
    dict[str, Any]
        A dictionary containing the following keys:
            - 'batch_size': int, batch size for training
            - 'reconstruction_log_prob': callable, function to compute
              reconstruction log probability
            - 'sampler': callable, function to sample from the generator output
            - 'vlb_scale_factor': float, scale factor for the variational
              lower bound optimization
            - 'optimizer': callable, function to create an optimizer
            - 'mask_generator': callable, function to generate masks
            - 'proposal_network': nn.Module, proposal network for the VAEAC
            - 'prior_network': nn.Module, prior network for the VAEAC
            - 'generative_network': nn.Module, generative network for the VAEAC
    """

    width = nn_size_dict["width"]
    depth = nn_size_dict["depth"]
    latent_dim = nn_size_dict["latent_dim"]

    # Proposal network
    proposal_layers = [
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes),
                                 list(range(len(one_hot_max_sizes)))),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes) * 2,
                  width),
        nn.LeakyReLU(),
    ]
    for i in range(depth):
        proposal_layers.append(
            SkipConnection(
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    proposal_layers.append(
        nn.Linear(width, latent_dim * 2)
    )
    proposal_network = nn.Sequential(*proposal_layers)

    # Prior network
    prior_layers = [
        CategoricalToOneHotLayer(one_hot_max_sizes +
                                 [0] * len(one_hot_max_sizes)),
        MemoryLayer('#input'),
        nn.Linear(sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes),
                  width),
        nn.LeakyReLU(),
    ]
    for i in range(depth):
        prior_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer('#%d' % i),
                nn.Linear(width, width),
                nn.LeakyReLU(),
            )
        )
    prior_layers.extend([
        MemoryLayer('#%d' % depth),
        nn.Linear(width, latent_dim * 2),
    ])
    prior_network = nn.Sequential(*prior_layers)

    # Generative network
    generative_layers = [
        nn.Linear(latent_dim, width),
        nn.LeakyReLU(),
    ]
    for i in range(depth + 1):
        generative_layers.append(
            SkipConnection(
                # skip-connection from prior network to generative network
                MemoryLayer('#%d' % (depth - i), True),
                nn.Linear(width * 2, width),
                nn.LeakyReLU(),
            )
        )
    generative_layers.extend([
        MemoryLayer('#input', True),
        nn.Linear(width + sum(max(1, x) for x in one_hot_max_sizes) +
                  len(one_hot_max_sizes),
                  sum(max(2, x) for x in one_hot_max_sizes)),
        SetGaussianSigmasToOne(one_hot_max_sizes),
    ])
    generative_network = nn.Sequential(*generative_layers)

    return {
        'batch_size': 64,

        'reconstruction_log_prob': GaussianCategoricalLoss(one_hot_max_sizes),

        'sampler': GaussianCategoricalSampler(one_hot_max_sizes,
                                              sample_most_probable=True),

        'vlb_scale_factor': 1 / len(one_hot_max_sizes),

        'optimizer': lambda parameters: Adam(parameters, lr=3e-4),

        'mask_generator': MCARGenerator(0.2),

        'proposal_network': proposal_network,

        'prior_network': prior_network,

        'generative_network': generative_network,
    }
