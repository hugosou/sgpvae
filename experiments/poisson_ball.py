import argparse
import numpy as np
import pandas as pd
import tqdm
import torch

# Install modules from parent directory.
import sys
sys.path.append('../')
import sgpvae

from data.eeg import load
from sgpvae.utils.misc import str2bool, save



import sys

# caution: path[0] is reserved for script path (or '' in REPL)

import sys
import os
sys.path.append('../../RPGPFA/')
from unstructured_recognition_gpfa import load_gprpm
from utils import generate_2D_latent
from datetime import datetime
import pickle

torch.set_default_dtype(torch.float32)



def main(args):
    # Load EEG data.
    data_type = torch.float32
    torch.set_default_dtype(data_type)

    data_folder = '../results_gprpm/benchmark/poisson'
    random_id = str(int(10000 * torch.rand(1).numpy()))

    # Generate Data
    data_type = torch.float32
    torch.set_default_dtype(data_type)

    # GPUs ?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('GP-RPM on GPU')
    else:
        print('GP-RPM on CPU')

    # Dimension of the observations
    num_observation = 50
    dim_observation = 10
    len_observation = 50


    # Sampling Frequency
    F = 10

    # Length of Each sample [sec]
    T = int(len_observation / F)

    # Oscillation Speed
    omega = 0.5


    # Random initializations
    theta = 2 * np.pi * np.random.rand(num_observation)
    z0 = torch.tensor(np.array([np.cos(theta), np.sin(theta)]).T)
    zt, _ = generate_2D_latent(T, F, omega, z0)



    # True Latent
    true_latent_ext = zt[:, 1:, 0].unsqueeze(-1)


    # Rate intensity for Poisson Noise
    mean_rate_loc = torch.linspace(-1, 1, dim_observation).unsqueeze(0).unsqueeze(0)
    vari_th = 4
    scale_th = 1
    rate_model = (vari_th ** 2) * torch.exp(-(mean_rate_loc - true_latent_ext) ** 2 / scale_th ** 2)
    observation_samples = torch.poisson(rate_model)

    # Convert Observations
    observations = torch.tensor(observation_samples, dtype=data_type)

    y = observations.permute(1, 2, 0)
    x = torch.linspace(0, 1, y.shape[0], device=device).unsqueeze(-1)

    # Normalise observations.
    y_mean, y_std = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
    y = (y - y_mean) / y_std
    batch_size = len_observation

    x = torch.tensor(x, device=device)
    y = torch.tensor(y, device=device)

    dataset = sgpvae.utils.dataset.TupleDataset(x, y, missing=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    # Model construction.
    # GP prior kernel.
    kernel = sgpvae.kernels.RBFKernel(
        lengthscale=args.init_lengthscale, scale=args.init_scale)
    w_transform = torch.sigmoid if args.transform else None

    # Likelihood function.
    if args.likelihood == 'gprn':
        print('Using GPRN likelihood function.')
        likelihood = sgpvae.likelihoods.GPRNHomoGaussian(
            f_dim=args.f_dim, out_dim=y.shape[1], sigma=args.sigma,
            w_transform=w_transform)
        latent_dim = args.f_dim + args.f_dim * y.shape[1]

    elif args.likelihood == 'gprn-nn':
        print('Using GPRN-NN likelihood function.')
        likelihood = sgpvae.likelihoods.GPRNNNHomoGaussian(
            f_dim=args.f_dim, w_dim=args.w_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma,
            w_transform=w_transform)
        latent_dim = args.f_dim + args.f_dim * args.w_dim

    elif args.likelihood == 'nn-gprn':
        print('Using NN-GPRN likelihood function.')
        likelihood = sgpvae.likelihoods.NNGPRNHomoGaussian(
            f_dim=args.f_dim, w_dim=args.w_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma)
        latent_dim = args.f_dim + args.w_dim * y.shape[1]

    elif args.likelihood == 'linear':
        print('Using linear likelihood function.')
        likelihood = sgpvae.likelihoods.AffineHomoGaussian(
            in_dim=args.latent_dim, out_dim=y.shape[1], sigma=args.sigma)

    else:
        print('Using NN likelihood function.')
        likelihood = sgpvae.likelihoods.NNHomoGaussian(
            in_dim=args.latent_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma)
        latent_dim = args.latent_dim

    # Approximate likelihood function.
    if args.pinference_net == 'factornet':
        variational_dist = sgpvae.likelihoods.FactorNet(
            in_dim=y.shape[1], out_dim=latent_dim,
            h_dims=args.h_dims, min_sigma=args.min_sigma,
            init_sigma=args.initial_sigma)

    elif args.pinference_net == 'indexnet':
        variational_dist = sgpvae.likelihoods.IndexNet(
            in_dim=y.shape[1], out_dim=latent_dim,
            inter_dim=args.inter_dim, h_dims=args.h_dims,
            rho_dims=args.rho_dims, min_sigma=args.min_sigma,
            init_sigma=args.initial_sigma)

    elif args.pinference_net == 'pointnet':
        variational_dist = sgpvae.likelihoods.PointNet(
            out_dim=latent_dim, inter_dim=args.inter_dim,
            h_dims=args.h_dims, rho_dims=args.rho_dims,
            min_sigma=args.min_sigma, initial_sigma=args.initial_sigma)

    elif args.pinference_net == 'zeroimputation':
        variational_dist = sgpvae.likelihoods.NNHeteroGaussian(
            in_dim=y.shape[1], out_dim=latent_dim,
            hidden_dims=args.h_dims, min_sigma=args.min_sigma,
            init_sigma=args.initial_sigma)

    else:
        raise ValueError('{} is not a partial inference network.'.format(
            args.pinference_net))

    # Construct SGP-VAE model.
    if args.model == 'gpvae':
        model = sgpvae.models.GPVAE(
            likelihood, variational_dist, latent_dim, kernel,
            add_jitter=args.add_jitter)

    elif args.model == 'sgpvae':
        z_init = torch.linspace(
            0, x[-1].item(), steps=args.num_inducing, device=device).unsqueeze(1)

        likelihood = likelihood.to(device)
        variational_dist = variational_dist.to(device)

        model = sgpvae.models.SGPVAE(
            likelihood, variational_dist, args.latent_dim, kernel, z_init,
            add_jitter=args.add_jitter, fixed_inducing=args.fixed_inducing)

        print('LOOK HERE')
        print(likelihood.network.layers[0].weight.device)
        print(variational_dist.networks[0].network.layers[0].weight.device)

    elif args.model == 'vae':
        model = sgpvae.models.VAE(
            likelihood, variational_dist, args.latent_dim)

    else:
        raise ValueError('{} is not a model.'.format(args.model))

    # Model training.
    model.train(True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_tot = []
    time_tot = []
    clock_start = datetime.now()
    epoch_iter = tqdm.tqdm(range(args.epochs), desc='Epoch')
    for epoch in epoch_iter:
        losses = []
        for batch in loader:
            x_b, y_b, idx = batch

            optimiser.zero_grad()
            loss = torch.tensor([0], device=device, dtype=data_type)
            for cur_observation in range(num_observation):
                loss -= model.elbo(x_b, y_b[:, :, cur_observation], None, num_samples=1)
            loss.backward()
            optimiser.step()
            losses.append(loss.item() * dataset.x.shape[0])

        loss_tot.append(loss.item() * dataset.x.shape[0])
        epoch_iter.set_postfix(loss=np.mean(losses))
        delta_time = (datetime.now() - clock_start).total_seconds()
        time_tot.append(delta_time)

        if epoch % args.cache_freq == 0:
            elbo=0
            for cur_observation in range(num_observation):
                elbo += model.elbo(dataset.x, dataset.y[:, :, cur_observation], None, num_samples=100)
            elbo *= dataset.x.shape[0]

            tqdm.tqdm.write('ELBO: {:.3f}'.format(elbo))

    latent_mean_tot = torch.zeros(num_observation, x.shape[0])

    for cur_observation in range(num_observation):
        latent_mean_tot[cur_observation] = (model.qf(dataset.x, dataset.y[:, :, cur_observation])[0].loc).squeeze(0)

    latent_mean_tot = latent_mean_tot.to("cpu")

    results = {'latent_mean': latent_mean_tot, 'loss': loss_tot, 'time': time_tot}
    model_name = './../../results_gp_rpm/2sgpvae_poisson_' \
                 + 'id' + random_id  + datetime.now().strftime("%Y_%M_%d_%Hh%Mm%Ss") + '.pkl'
    with open(model_name, 'wb') as handle:
        pickle.dump(results, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Kernel.
    parser.add_argument('--init_lengthscale', default=0.05, type=float)
    parser.add_argument('--init_scale', default=1., type=float)

    # GPVAE.
    parser.add_argument('--model', default='sgpvae')
    parser.add_argument('--likelihood', default='nn', type=str)
    parser.add_argument('--pinference_net', default='factornet', type=str)
    parser.add_argument('--latent_dim', default=1, type=int)
    parser.add_argument('--f_dim', default=3, type=int)
    parser.add_argument('--w_dim', default=3, type=int)
    parser.add_argument('--decoder_dims', default=[50, 50], nargs='+',
                        type=int)
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--h_dims', default=[50, 50], nargs='+', type=int)
    parser.add_argument('--rho_dims', default=[20], nargs='+', type=int)
    parser.add_argument('--inter_dim', default=20, type=int)
    parser.add_argument('--num_inducing', default=20, type=int)
    parser.add_argument('--fixed_inducing', default=True, type=str2bool)
    parser.add_argument('--add_jitter', default=True,
                        type=sgpvae.utils.misc.str2bool)
    parser.add_argument('--min_sigma', default=1e-3, type=float)
    parser.add_argument('--initial_sigma', default=.1, type=float)
    parser.add_argument('--transform', default=False, type=str2bool)
    parser.add_argument('--elbo_subset', default=False, type=str2bool)

    # Training.
    parser.add_argument('--epochs', default=40000, type=int)
    parser.add_argument('--cache_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    # General.
    parser.add_argument('--save', default=False, type=str2bool)
    parser.add_argument('--results_dir', default='./_results/eeg/', type=str)

    args = parser.parse_args()
    main(args)
