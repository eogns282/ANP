import os
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.Sinusoid.sinusoid import sinusoid_dataset, sinusoid_collate, SineData
from utils.ops import kl_divergence, log_normal_pdf
from utils.visualization import vis_sinusoid_traj


class Trainer_sinusoid:
    def __init__(self, args, model):
        self.exp_name = args.exp_name
        self.model = model
        self.task = args.task
        self.noisy_data = args.noisy_data
        self.num_epochs = args.epochs
        self.cv_idx = args.cv_idx
        self.num_data = args.num_data
        self.diverse = args.diverse_data
        self.batch_size = args.batch_size
        self.num_full = args.num_full_x

        self.sample_strategy = args.sample_strategy

        # self._data_indexing(self.cv_idx)
        self._load_dataloader_val()
        self._make_folder()

        self.optimizer = optim.Adam(nn.ParameterList(self.model.parameters()), lr=1e-3)
        self.best_loss = float('inf')
        self.best_mse = float('inf')
        self.best_kl = float('inf')
        self.device = torch.device(f"cuda:{args.gpu_num}")
        self.writer = SummaryWriter("./runs/{}".format(self.exp_name))

    # def _data_indexing(self, cv_idx):
    #     idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    #     data_idx_num = idx_list[cv_idx]
    #     i = data_idx_num[0]
    #     j = data_idx_num[1]
    #
    #     if self.num_data == 1000:
    #         test_range = range(200 * i, 200 * (i + 1))
    #         val_range = range(200 * j, 200 * (j + 1))
    #         train_range = set(range(1000)) - set(test_range) - set(val_range)
    #     elif self.num_data == 3000:
    #         test_range = range(600 * i, 600 * (i + 1))
    #         val_range = range(600 * j, 600 * (j + 1))
    #         train_range = set(range(3000)) - set(test_range) - set(val_range)
    #
    #     self.val_idx = np.array(val_range)
    #     self.train_idx = np.array(list(train_range))

    def _load_dataloader_train(self, batch_size):
        print('Loading a new training dataset')
        data_train = SineData(num_samples=1000)
        self.dl_train = DataLoader(dataset=data_train, shuffle=True, batch_size=batch_size)

    def _load_dataloader_val(self):
        # data_val = SineData(num_samples=300)
        f = open('datasets/Sinusoid/val_data.pickle', 'rb')
        data_val = pickle.load(f)
        f.close()
        self.dl_val = DataLoader(dataset=data_val, shuffle=False, batch_size=len(data_val))


    # def _load_dataloader(self, batch_size):
    #     if self.noisy_data:
    #         if self.num_data == 1000:
    #             data_path = './datasets/Sinusoid/sinusoid_new_noisy.csv'
    #         elif self.num_data == 3000:
    #             if self.diverse:
    #                 data_path = './datasets/Sinusoid/sinusoid_noisy_div_3000.csv'
    #             else:
    #                 data_path = './datasets/Sinusoid/sinusoid_noisy_3000.csv'
    #     else:
    #         data_path = './datasets/Sinusoid/sinusoid_new_noiseless.csv'
    #     # data_train = sinusoid_dataset(data_path=data_path, idx=self.train_idx)
    #     # data_val = sinusoid_dataset(data_path=data_path, idx=self.val_idx)
    #     # self.dl_train = DataLoader(dataset=data_train, shuffle=True, collate_fn=sinusoid_collate,
    #     #                            batch_size=batch_size)
    #     # self.dl_val = DataLoader(dataset=data_val, shuffle=False, collate_fn=sinusoid_collate,
    #     #                          batch_size=len(data_val))
    #
    #     data_train = SineData(num_samples=3000)
    #     data_val = SineData(num_samples=300)
    #     self.dl_train = DataLoader(dataset=data_train, shuffle=True, batch_size=batch_size)
    #     self.dl_val = DataLoader(dataset=data_val, shuffle=False, batch_size=len(data_val))

    def _make_folder(self):
        if not os.path.isdir("./save/{}".format(self.exp_name)):
            if not os.path.isdir("./save"):
                os.mkdir("./save")
            os.mkdir("./save/{}".format(self.exp_name))

        if not os.path.isdir("./results/{}".format(self.exp_name)):
            if not os.path.isdir("./results"):
                os.mkdir("./results")
            os.mkdir("./results/{}".format(self.exp_name))

    def _train_phase(self, epoch):
        print("Start training")
        self.model.train()
        self.train_loss = 0.
        self.train_instance_num = 0
        self.train_mse = 0.
        self.train_kl = 0.
        self._load_dataloader_train(self.batch_size)

        for i, b in enumerate(tqdm(self.dl_train)):
            # times = b["time"].float().to(self.device)
            # trajs = b["traj"].float().to(self.device)

            b = torch.stack(b, 2).squeeze()
            times = b[0, :, 1].float().to(self.device)
            trajs = b[:, :, 2:].float().to(self.device)

            # sampling num of context / target
            if self.task == 'extrapolation':
                if self.sample_strategy == 1:
                    num_context = int(self.num_full / 2)  # 50
                    context_idx = np.arange(0, num_context)
                    target_idx = np.arange(num_context, self.num_full)

                elif self.sample_strategy == 2:
                    num_context = int(int(self.num_full / 2) / 2)  # 25
                    num_target = int(int(self.num_full / 2) / 2)  # 25
                    all_idx = np.random.choice(self.num_full, num_context + num_target, replace=False)  # 50
                    all_idx.sort()
                    context_idx = all_idx[:num_context]
                    target_idx = all_idx[num_context:]

                elif self.sample_strategy == 3:
                    num_context = np.random.choice(np.arange(3, self.num_full), 1)[0]
                    num_target = np.random.choice(np.arange(0, self.num_full - num_context), 1)[0]
                    all_idx = np.random.choice(self.num_full, num_context + num_target, replace=False)
                    all_idx.sort()
                    context_idx = all_idx[:num_context]
                    target_idx = all_idx[num_context:]

                else:
                    print('Incorrect sampling strategy')

            elif self.task == 'interpolation':
                if self.sample_strategy == 1:
                    num_context = int(self.num_full / 2)  # 50
                    context_idx = np.arange(0, self.num_full, 2)
                    target_idx = np.arange(1, self.num_full, 2)

                elif self.sample_strategy == 2:
                    num_context = int(int(self.num_full/ 2) / 2)  # 25
                    num_target = int(int(self.num_full/ 2) / 2)  # 25
                    all_idx = np.random.choice(self.num_full, num_context + num_target, replace=False)  # 50
                    context_idx = all_idx[:num_context]  # 1 3
                    context_idx.sort()
                    target_idx = all_idx[num_context:]  # 2 4
                    target_idx.sort()

                elif self.sample_strategy == 3:
                    num_context = np.random.choice(np.arange(3, self.num_full), 1)[0]  # n~U(3, 100)
                    num_target = np.random.choice(np.arange(0, self.num_full - num_context), 1)[0]  # m~n+U(0, 100-n)
                    all_idx = np.random.choice(self.num_full, num_context + num_target, replace=False)  # m + n
                    context_idx = all_idx[:num_context]
                    context_idx.sort()
                    target_idx = all_idx[num_context:]
                    target_idx.sort()

                else:
                    print('Incorrect sampling strategy')

            else:
                print('Incorrect task condition!')



            both_idx = np.concatenate([context_idx, target_idx])  # 1 3 2 4

            self.optimizer.zero_grad()

            pred_mu, pred_sigma, mu_all, \
            sigma_all, mu_context, sigma_context = self.model(trajs, times, context_idx, target_idx, training=True)
            # 1 3 2 4: pred_mu pred_sigma mu_all sigma_all
            # 1 3: mu_context sigma_context

            # mse_loss = torch.mean(torch.pow(pred - trajs, 2))
            mse_loss = log_normal_pdf(trajs[:, both_idx, :], pred_mu, pred_sigma).mean()  # 1 3 2 4

            kl_loss = kl_divergence(mu_all, sigma_all, mu_context, sigma_context)

            # loss = mse_loss + kl_loss
            loss = -mse_loss + kl_loss

            loss.backward()
            self.optimizer.step()

            inst_num = torch.ones_like(pred_mu).sum().item()
            self.train_loss += loss.item() * inst_num
            self.train_instance_num += inst_num
            self.train_mse += mse_loss.item() * inst_num
            self.train_kl += kl_loss.item() * inst_num

        self.train_loss /= self.train_instance_num
        self.train_mse /= self.train_instance_num
        self.train_kl /= self.train_instance_num

        if (epoch + 1) % 2 == 0:
            vis_sinusoid_traj(trajs[0, context_idx, 0], times[context_idx],
                              trajs[0, target_idx, 0], times[target_idx],
                              pred_mu[0, :num_context, 0], pred_mu[0, num_context:, 0],
                              epoch, '/' + str(self.exp_name) + '/')

    def _validation_phase(self, epoch):
        print("Validation phase")
        self.model.eval()
        self.val_loss = 0.
        self.val_instance_num = 0
        self.val_mse = 0.
        self.val_kl = 0.

        with torch.no_grad():
            for i, b in enumerate(self.dl_val):
                # times = b["time"].float().to(self.device)
                # trajs = b["traj"].float().to(self.device)

                b = torch.stack(b, 2).squeeze()
                times = b[0, :, 1].float().to(self.device)
                trajs = b[:, :, 2:].float().to(self.device)

                # sampling num of context / target
                if self.task == 'extrapolation':
                    num_context = int(self.num_full / 2)  # 50
                    context_idx = np.arange(0, num_context, 2)
                    target_idx = np.arange(num_context, self.num_full, 2)
                elif self.task == 'interpolation':
                    num_context = int(self.num_full / 4)  # 50
                    context_idx = np.arange(0, self.num_full, 4)
                    target_idx = np.arange(2, self.num_full, 4)
                else:
                    print('Incorrect task condition!')

                both_idx = np.concatenate([context_idx, target_idx])
                pred_mu, pred_sigma, mu_all, sigma_all, mu_context, sigma_context = self.model(trajs, times,
                                                                                               context_idx, target_idx,
                                                                                               training=True)
                mse_loss = torch.mean(torch.pow(pred_mu - trajs[:, both_idx, :], 2))
                kl_loss = kl_divergence(mu_all, sigma_all, mu_context, sigma_context)

                loss = mse_loss + kl_loss

                inst_num = torch.ones_like(pred_mu).sum().item()
                self.val_loss += loss.item() * inst_num
                self.val_instance_num += inst_num
                self.val_mse += mse_loss.item() * inst_num
                self.val_kl += kl_loss.item() * inst_num

            self.val_loss /= self.val_instance_num
            self.val_mse /= self.val_instance_num
            self.val_kl /= self.val_instance_num

        if self.val_mse < self.best_mse:
            self.best_loss = self.val_loss
            self.best_mse = self.val_mse
            self.best_kl = self.val_kl
            self.best_epoch = epoch
            print('save best model')

            if epoch < 500:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'best_mse': self.best_mse,
                    'best_kl': self.best_kl,
                    'best_epoch': self.best_epoch,
                }, "./save/{}/500_best_model".format(self.exp_name))
            elif (epoch >= 500) and (epoch < 1000):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'best_mse': self.best_mse,
                    'best_kl': self.best_kl,
                    'best_epoch': self.best_epoch,
                }, "./save/{}/1000_best_model".format(self.exp_name))
            elif (epoch >= 1000) and (epoch < 1500):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'best_mse': self.best_mse,
                    'best_kl': self.best_kl,
                    'best_epoch': self.best_epoch,
                }, "./save/{}/1500_best_model".format(self.exp_name))

        if (epoch + 1) % 2 == 0:
            vis_sinusoid_traj(trajs[0, context_idx, 0], times[context_idx],
                              trajs[0, target_idx, 0], times[target_idx],
                              pred_mu[0, :num_context, 0], pred_mu[0, num_context:, 0],
                              epoch, '/' + str(self.exp_name) + '/', val=True)

    def _logger_scalar(self, epoch):
        self.writer.add_scalar('train_loss', self.train_loss, epoch)
        self.writer.add_scalar('val_loss', self.val_loss, epoch)
        self.writer.add_scalar('train_mse', self.train_mse, epoch)
        self.writer.add_scalar('train_kl', self.train_kl, epoch)
        self.writer.add_scalar('val_mse', self.val_mse, epoch)
        self.writer.add_scalar('val_kl', self.val_kl, epoch)

    def run(self):
        for epoch in range(self.num_epochs):
            self._train_phase(epoch)
            self._validation_phase(epoch)

            print(f"Loss at epoch {epoch}: train_loss={self.train_loss:.5f}, val_loss={self.val_loss:.5f}")
            print(f"Current best loss={self.best_loss:.5f}")
            print(f"Train mse(likelihood)={self.train_mse:.5f}, Train kl={self.train_kl:.5f}")
            print(f"Val mse={self.val_mse:.5f}, Val kl={self.val_kl:.5f}")

            self._logger_scalar(epoch)

            if (epoch + 1) % 10 == 0:
                print('save current model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'current_loss': self.val_loss,
                    'current_mse': self.val_mse,
                    'current_kl': self.val_kl,
                    'best_epoch': self.best_epoch
                }, "./save/{}/current_model".format(self.exp_name))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_mse': self.best_mse,
            'best_kl': self.best_kl,
            'best_epoch': self.best_epoch
        }, "./save/{}/final_model".format(self.exp_name))

        df_file_name = "./results/{}/sinusoid.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "Loss": [self.best_loss], "best_epoch": [self.best_epoch],
             'MSE': [self.best_mse], 'KL': [self.best_kl]})
        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)

        self.writer.close()