import torch
from torch import nn
from torch.nn import functional as F


class NeuralProcess(nn.Module):
    def __init__(self, args):
        super(NeuralProcess, self).__init__()
        self.x_size = args.x_size
        self.h_size = args.h_size
        self.device = torch.device(f"cuda:{args.gpu_num}")

        self.encoder = Encoder(self.x_size, self.h_size)
        self.decoder = Decoder(self.x_size, self.h_size)

    def forward(self, trajs, times, training=True):
        if training:
            mu_all, sigma_all = self.encoder(times, trajs)
            mu_context, sigma_context = self.encoder(times[:25], trajs[:, :25, :])

            epsilon = torch.randn(sigma_all.size()).to(self.device)
            z = mu_all + sigma_all * epsilon

            x_mu, x_sigma = self.decoder(times, z)

            return x_mu, x_sigma, mu_all, sigma_all, mu_context, sigma_context
        else:
            mu_context, sigma_context = self.encoder(times[:25], trajs[:, :25, :])

            epsilon = torch.randn(sigma_context.size()).to(self.device)
            z = mu_context + sigma_context * epsilon

            x_mu, x_sigma = self.decoder(times, z)

            return x_mu, x_sigma


class Encoder(nn.Module):
    def __init__(self, x_size, h_size):
        super(Encoder, self).__init__()
        self.h_size = h_size
        self.x_size = x_size

        self.encoder = nn.Sequential(nn.Linear(self.x_size + 1, 40),
                                     nn.ReLU(),
                                     nn.Linear(40, 40),
                                     nn.ReLU(),
                                     nn.Linear(40, self.h_size))

        self.latent_mu = nn.Sequential(nn.Linear(self.h_size, 40),
                                       nn.ReLU(),
                                       nn.Linear(40, self.h_size))

        self.latent_sigma = nn.Sequential(nn.Linear(self.h_size, 40),
                                          nn.ReLU(),
                                          nn.Linear(40, self.h_size))

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


class Decoder(nn.Module):
    def __init__(self, x_size, h_size):
        super(Decoder, self).__init__()

        self.x_size = x_size
        self.h_size = h_size

        self.decoder = nn.Sequential(nn.Linear(self.h_size + 1, 40),
                                     nn.ReLU(),
                                     nn.Linear(40, 40),
                                     nn.ReLU(),
                                     nn.Linear(40, 40),
                                     nn.ReLU())

        self.x_mu = nn.Linear(40, self.x_size)
        self.x_sigma = nn.Linear(40, self.x_size)

    def forward(self, times, h_vector):
        '''
        h_vector: (Batch Size) x (Hidden Dim)
        times: (Traj Len)
        '''
        batch_size, _ = h_vector.size()
        traj_len = times.size()[0]

        times = times.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)  # BS x TL x 1
        h_vector = h_vector.unsqueeze(1).repeat(1, traj_len, 1)  # BS x TL x H

        times_flat = times.view(batch_size * traj_len, 1)  # BS*TL x 1
        h_flat = h_vector.view(batch_size * traj_len, self.h_size)  # BS*TL x H

        input_pairs = torch.cat((times_flat, h_flat), dim=1)  # BS*TL x (1+H)

        h = self.decoder(input_pairs)  # BS*TL x 50
        mu = self.x_mu(h)  # BS*TL x x_dim
        sigma = 0.1 + 0.9 * F.softplus(self.x_sigma(h))  # BS*TL x x_dim
        mu = mu.view(batch_size, traj_len, self.x_size)  # BS x TL x x_dim
        sigma = sigma.view(batch_size, traj_len, self.x_size)  # BS x TL x x_dim

        return mu, sigma