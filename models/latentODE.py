import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
from utils.ode_fn import ODE_function


class LatentODE(nn.Module):
    '''new'''
    def __init__(self, args):
        super(LatentODE, self).__init__()
        self.x_size = args.x_size
        self.h_size = args.h_size
        self.device = torch.device(f"cuda:{args.gpu_num}")

        self.ode_function = ODE_function(self.h_size)
        self.encoder = Encoder(self.x_size, self.h_size, self.device)
        self.decoder = Decoder(self.x_size, self.h_size)

    def forward(self, trajs, times, context_idx, target_idx, training=True):
        both_idx = np.concatenate([context_idx, target_idx])  # 0 2 1 3
        temp_idx = np.argsort(both_idx.copy())  # 0 2 1 3
        both_idx.sort()  # 0 1 2 3
        times = times.to(self.device)
        if training:
            mu_all, sigma_all = self.encoder(times[both_idx], trajs[:, both_idx, :], self.ode_function)
            # mu_context, sigma_context = self.encoder(times[context_idx], trajs[:, context_idx, :],
            #                                          self.ode_function)
            mu_context = torch.zeros_like(mu_all)
            sigma_context = torch.ones_like(sigma_all)

            epsilon = torch.randn(sigma_all.size()).to(self.device)
            z = mu_all + sigma_all * epsilon

            x_mu, x_sigma = self.decoder(times[both_idx], z, self.ode_function)  # 0 1 2 3
            x_mu = x_mu[:, temp_idx, :]  # 0 2 1 3
            x_sigma = x_sigma[:, temp_idx, :]  # 0 2 1 3

            return x_mu, x_sigma, mu_all, sigma_all, mu_context, sigma_context
        else:
            # mu_context, sigma_context = self.encoder(times[context_idx], trajs[:, context_idx, :])
            mu_context, _ = self.encoder(times[context_idx], trajs[:, context_idx, :], self.ode_function)

            # epsilon = torch.randn(sigma_context.size()).to(self.device)
            # z = mu_context + sigma_context * epsilon
            z = mu_context

            x_mu, x_sigma = self.decoder(times[both_idx], z, self.ode_function)  # 1 2 3 4
            x_mu = x_mu[:, temp_idx, :]
            x_sigma = x_sigma[:, temp_idx, :]

            return x_mu, x_sigma


class Encoder(nn.Module):
    '''new'''
    def __init__(self, x_size, h_size, device):
        super(Encoder, self).__init__()
        self.h_size = h_size
        self.x_size = x_size
        self.device = device

        self.rnncell = nn.GRUCell(x_size, self.h_size)

        self.latent_mu = nn.Sequential(nn.Linear(self.h_size, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, self.h_size))

        self.latent_sigma = nn.Sequential(nn.Linear(self.h_size, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, self.h_size))

    def forward(self, times, trajs, ode_function):
        '''
        trajs: (Batch Size) x (Traj Len) x (Traj Dim)
        times: (Trja Len)
        '''
        h_vector = torch.zeros(trajs.size(0), self.h_size).to(self.device)
        traj_len = len(times)

        for i, _ in enumerate(times):
            # flow
            # if i != 0:
            #     h_vector = odeint(ode_function, h_vector,
            #                       torch.Tensor([times[i-1], times[i]]).to(self.device), method='rk4',
            #                       rtol=1e-3, atol=1e-4).permute(1, 0, 2)
            #     h_vector = h_vector[:, -1, :]
            # # jump
            # x = trajs[:, i, :]
            if i != 0:
                h_vector = odeint(ode_function, h_vector,
                                  torch.Tensor([times[traj_len-i], times[traj_len-1-i]]).to(self.device), method='rk4',
                                  rtol=1e-3, atol=1e-4).permute(1, 0, 2)
                h_vector = h_vector[:, -1, :]
            # jump
            x = trajs[:, traj_len-1-i, :]
            h_vector = self.rnncell(x, h_vector)

        mu = self.latent_mu(h_vector)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.latent_sigma(h_vector))

        return mu, sigma


class Decoder(nn.Module):
    '''new'''
    def __init__(self, x_size, h_size):
        super(Decoder, self).__init__()
        self.x_size = x_size
        self.h_size = h_size

        self.nn = nn.Sequential(nn.Linear(h_size, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU())

        self.x_mu = nn.Linear(128, self.x_size)
        self.x_sigma = nn.Linear(128, self.x_size)

    def forward(self, times, h_vector, ode_function):
        '''
        h_vector: (Batch size) x (Hidden dim)
        times: (Traj len)
        '''
        batch_size, _ = h_vector.size()
        traj_len = times.size()[0]

        # flow
        h_vectors = odeint(ode_function, h_vector, times, method='rk4', rtol=1e-3, atol=1e-4).permute(1, 0, 2)

        # decoding
        h_vectors = self.nn(h_vectors)
        mu = self.x_mu(h_vectors)
        sigma = 0.1 + 0.9 * F.softplus(self.x_sigma(h_vectors))  # BS*TL x x_dim
        mu = mu.view(batch_size, traj_len, self.x_size)  # BS x TL x x_dim
        sigma = sigma.view(batch_size, traj_len, self.x_size)  # BS x TL x x_dim

        return mu, sigma