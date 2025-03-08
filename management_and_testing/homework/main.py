import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import wandb
from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch, get_samples
from modeling.unet import UnetModel
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from hpamams import config
import sys

wandb.login(key='03473c2b8e9b6a50995c56a4492a3bcd7da7483f')

def main(device: str):
    wandb.init(config=config, project="effdl_hw1", name="first_try")
    num_epochs = config['num_epochs']

    ddpm = DiffusionModel(
        eps_model=UnetModel(config['in_channels'], config['out_channels'], hidden_size=config['hidden_size']),
        betas=config['betas_range'],
        num_timesteps=config['num_timesteps'],
    )
    ddpm.to(device)
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=config['learning_rate'])
    lr_scheduler = CosineAnnealingLR(optim, 200)
    wandb.watch(ddpm)
    wandb.log({'input_data' : next(iter(dataloader))})

    for i in range(1, num_epochs + 1):
        losses_train = train_epoch(ddpm, dataloader, optim, device)
        metrics = {'train_loss' : np.asarray(losses_train).mean(), 'lr' : lr_scheduler.get_last_lr()[0]}
        wandb.log(metrics, step=i)
        samples = get_samples(ddpm, device)
        image = wandb.Image(samples)
        metrics['Example_of_generation'] = image
        wandb.log(metrics, step=i)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
