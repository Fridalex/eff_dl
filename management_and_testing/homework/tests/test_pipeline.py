import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
import numpy as np

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, generate_samples, train_epoch
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5

@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(device, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)
    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    
    num_epochs = 1
    if device == 'cpu':
        train_dataset = torch.utils.data.Subset(train_dataset, range(16))

    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i in range(num_epochs):
        losses = train_epoch(ddpm, dataloader, optim, device)
        print(f'Loss after epoch {i+1} is {np.asarray(losses)}')
        if device == 'cpu':
            generate_samples(ddpm, device, f'test_samples_cpu/{i+1:02d}.png')
        else:
            generate_samples(ddpm, device, f'test_samples_gpu/{i+1:02d}.png')
