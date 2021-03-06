import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class NeuralProcess_rev(nn.Module):
    def __init__(self, args):
        super(NeuralProcess_rev, self).__init__()
        self.x_size = args.x_size
        self.h_size = args.h_size
        self.z_size = args.h_size
        self.device = torch.device(f"cuda:{args.gpu_num}")

        self.lat_enc = LatentEncoder(self.x_size, self.h_size)
        self.det_enc = DeterministicEncoder(self.x_size, self.h_size)
        self.decoder = Decoder(self.x_size, self.h_size, self.z_size)

    def forward(self, trajs, times, context_idx, target_idx, training=True):
        both_idx = np.concatenate([context_idx, target_idx])
        if training:
            mu_all, sigma_all = self.lat_enc(times[both_idx], trajs[:, both_idx, :])
            mu_context, sigma_context = self.lat_enc(times[context_idx], trajs[:, context_idx, :])

            r_vec = self.det_enc(times[context_idx], trajs[:, context_idx, :])

            epsilon = torch.randn(sigma_all.size()).to(self.device)
            z = mu_all + sigma_all * epsilon

            x_mu, x_sigma = self.decoder(times[both_idx], r_vec, z)

            return x_mu, x_sigma, mu_all, sigma_all, mu_context, sigma_context
        else:
            # mu_context, sigma_context = self.lat_enc(times[context_idx], trajs[:, context_idx, :])
            mu_context, _ = self.lat_enc(times[context_idx], trajs[:, context_idx, :])

            r_vec = self.det_enc(times[context_idx], trajs[:, context_idx, :])

            # epsilon = torch.randn(sigma_context.size()).to(self.device)
            # z = mu_context + sigma_context * epsilon
            z = mu_context

            x_mu, x_sigma = self.decoder(times[both_idx], r_vec, z)

            return x_mu, x_sigma


class LatentEncoder(nn.Module):
    def __init__(self, x_size, h_size):
        super(LatentEncoder, self).__init__()
        self.h_size = h_size
        self.x_size = x_size

        self.encoder = nn.Sequential(nn.Linear(self.x_size + 1, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.h_size))

        self.latent_mu = nn.Sequential(nn.Linear(self.h_size, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, self.h_size))

        self.latent_sigma = nn.Sequential(nn.Linear(self.h_size, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, self.h_size))

    def forward(self, times, trajs):
        '''
        trajs: (Batch Size) x (Traj Len) x (Traj Dim)
        times: (Traj Len)
        '''
        batch_size, traj_len, _ = trajs.size()

        times = times.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)

        times_flat = times.view(batch_size * traj_len, 1)  # bs * tl x 1
        trajs_flat = trajs.contiguous().view(batch_size * traj_len, self.x_size)  # bs * tl x x_dim

        input_pairs = torch.cat((times_flat, trajs_flat), dim=1)  # bs*tl x (1+x_dim)
        r = self.encoder(input_pairs)  # bs*tl x r_dim
        r = r.view(batch_size, traj_len, self.h_size)  # bs x tl x r_dim
        r = r.mean(1)  # bs x r_dim

        mu = self.latent_mu(r)  # bs x r_dim
        sigma = 0.1 + 0.9 * torch.sigmoid(self.latent_sigma(r))  # bs x r_dim

        return mu, sigma


class DeterministicEncoder(nn.Module):
    def __init__(self, x_size, h_size):
        super(DeterministicEncoder, self).__init__()
        self.x_size = x_size
        self.h_size = h_size

        self.encoder = nn.Sequential(nn.Linear(self.x_size + 1, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.h_size))

    def forward(self, times, trajs):
        '''
        trajs: (Batch Size) x (Traj Len) x (Traj Dim)
        times: (Traj Len)
        '''
        batch_size, traj_len, _ = trajs.size()

        times = times.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)

        times_flat = times.view(batch_size * traj_len, 1)  # bs * tl x 1
        trajs_flat = trajs.contiguous().view(batch_size * traj_len, self.x_size)  # bs * tl x x_dim

        input_pairs = torch.cat((times_flat, trajs_flat), dim=1)  # bs * tl x (1 + x_dim)
        r = self.encoder(input_pairs)  # bs * tl x r_dim
        r = r.view(batch_size, traj_len, self.h_size)  # bs x tl x r_dim
        r = r.mean(1)  # bs x r_dim
        return r


class Decoder(nn.Module):
    def __init__(self, x_size, h_size, z_size):
        super(Decoder, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.z_size = z_size

        self.decoder = nn.Sequential(nn.Linear(self.h_size + self.z_size + 1, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU())

        self.x_mu = nn.Linear(128, self.x_size)
        self.x_sigma = nn.Linear(128, self.x_size)

    def forward(self, times, h_vector, z_vector):
        '''
        h_vector: (Batch Size) x (Hidden Dim)
        z_vector: (Batch Size) x (Hidden Dim)
        times: (Traj Len)
        '''
        batch_size, _ = z_vector.size()
        traj_len = times.size()[0]

        times = times.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)  # BS x TL x 1
        h_vector = h_vector.unsqueeze(1).repeat(1, traj_len, 1)
        z_vector = z_vector.unsqueeze(1).repeat(1, traj_len, 1)  # BS x TL x H

        times_flat = times.view(batch_size * traj_len, 1)  # BS*TL x 1
        h_flat = h_vector.view(batch_size * traj_len, self.h_size)
        z_flat = z_vector.view(batch_size * traj_len, self.h_size)  # BS*TL x H

        input_pairs = torch.cat((times_flat, h_flat, z_flat), dim=1)  # BS*TL x (1+H)

        h = self.decoder(input_pairs)  # BS*TL x 50
        mu = self.x_mu(h)  # BS*TL x x_dim
        sigma = 0.1 + 0.9 * F.softplus(self.x_sigma(h))  # BS*TL x x_dim
        mu = mu.view(batch_size, traj_len, self.x_size)  # BS x TL x x_dim
        sigma = sigma.view(batch_size, traj_len, self.x_size)  # BS x TL x x_dim

        return mu, sigma