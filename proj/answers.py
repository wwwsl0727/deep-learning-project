r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 128
    hypers['z_dim'] = 128
    hypers['x_sigma2'] = 0.9
    hypers['learn_rate'] = 0.001
    hypers['betas'] = (0.9,0.99)
    # ========================
    return hypers

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======

    hypers['batch_size'] = 64
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.1
    hypers['discriminator_optimizer'] = {'type': 'Adam', 'lr': 2e-4 , 'betas':(0.0,0.9)}
    hypers['generator_optimizer'] = {'type': 'Adam', 'lr': 2e-4,  'betas':(0.0,0.9) }
    # ========================
    return hypers


# ==============


