import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from datasets.Sinusoid.sinusoid import sinusoid_dataset, sinusoid_collate


class Evaluator_sinusoid:
    def __init__(self, args, model):
        self.device = torch.device(f"cuda:{args.gpu_num}")
        self.exp_name = args.exp_name
        self.model = model
        self.task = args.task
        self.noisy_data = args.noisy_data
        self.cv_idx = args.cv_idx
        self.num_data = args.num_data
        self.diverse = args.diverse_data
        self.num_full = args.num_full_x

        self.sample_strategy = args.sample_strategy

        # self._data_indexing(self.cv_idx)
        # self._load_dataloader()
        self._load_dataloader_test()
        self._load_checkpoint()
        self._make_folder()

    # def _data_indexing(self, cv_idx):
    #     idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    #     data_idx_num = idx_list[cv_idx]
    #     i = data_idx_num[0]
    #
    #     if self.num_data == 1000:
    #         test_range = range(200 * i, 200 * (i + 1))
    #     elif self.num_data == 3000:
    #         test_range = range(600 * i, 600 * (i + 1))
    #
    #     self.test_idx = np.array(test_range)
    #
    # def _load_dataloader(self):
    #     if self.noisy_data:
    #         if self.num_data == 1000:
    #             data_path = './datasets/Sinusoid/sinusoid_new_noisy.csv'
    #         else:
    #             if self.diverse:
    #                 data_path = './datasets/Sinusoid/sinusoid_noisy_div_3000.csv'
    #             else:
    #                 data_path = './datasets/Sinusoid/sinusoid_noisy_3000.csv'
    #     else:
    #         data_path = './datasets/Sinusoid/sinusoid_new_noiseless.csv'
    #     data_test = sinusoid_dataset(data_path=data_path, idx=self.test_idx)
    #     self.dl_test = DataLoader(dataset=data_test, shuffle=False, collate_fn=sinusoid_collate,
    #                               batch_size=len(data_test))

    def _load_dataloader_test(self):
        f = open('datasets/Sinusoid/test_data.pickle', 'rb')
        data_test = pickle.load(f)
        f.close()
        self.dl_test = DataLoader(dataset=data_test, shuffle=False, batch_size=len(data_test))

    def _load_checkpoint(self):
        try:
            checkpoint = torch.load('./save/{}/1500_best_model'.format(self.exp_name))
        except:
            try:
                checkpoint = torch.load('./save/{}/1000_best_model'.format(self.exp_name))
            except:
                checkpoint = torch.load('./save/{}/500_best_model'.format(self.exp_name))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_epoch = checkpoint['best_epoch']

    def _make_folder(self):
        if not os.path.isdir("./test_results/{}".format(self.exp_name)):
            if not os.path.isdir("./test_results"):
                os.mkdir("./test_results")
            os.mkdir("./test_results/{}".format(self.exp_name))

    def _test_phase(self):
        print("Test start")
        self.model.eval()
        self.test_loss = 0
        self.test_instance_num = 0
        self.test_mse = 0
        self.test_kl = 0
        self.mse_context = 0
        self.mse_target = 0

        with torch.no_grad():
            for i, b in enumerate(self.dl_test):
                # times = b["time"].float().to(self.device)
                # trajs = b["traj"].float().to(self.device)
                b = torch.stack(b, 2).squeeze()
                times = b[0, :, 1].float().to(self.device)
                trajs = b[:, :, 2:].float().to(self.device)

                # sampling num of context / target
                if self.task == 'extrapolation':
                    num_context = int(self.num_full / 4)  # 50
                    context_idx = np.arange(0, num_context, 2)
                    target_idx = np.arange(num_context, self.num_full, 2)

                elif self.task == 'interpolation':
                    num_context = int(self.num_full / 4)  # 50
                    context_idx = np.arange(0, self.num_full, 4)
                    target_idx = np.arange(2, self.num_full, 4)
                else:
                    print('Incorrect task condition!')

                both_idx = np.concatenate([context_idx, target_idx])


                pred_mu, pred_sigma = self.model(trajs, times, context_idx, target_idx,
                                                 training=False)
                mse_loss = torch.mean(torch.pow(pred_mu - trajs[:, both_idx, :], 2))
                self.mse_context += torch.mean(torch.pow(pred_mu[:, :len(context_idx), :] - trajs[:, context_idx, :], 2))
                self.mse_target += torch.mean(torch.pow(pred_mu[:, len(context_idx):, :] - trajs[:, target_idx, :], 2))
                kl_loss = 0.

                loss = mse_loss + kl_loss

                inst_num = torch.ones_like(pred_mu).sum().item()
                self.test_loss += loss.item() * inst_num
                self.test_instance_num += inst_num
                self.test_mse += mse_loss.item() * inst_num
                self.test_kl += kl_loss * inst_num

            self.test_loss /= self.test_instance_num
            self.test_mse /= self.test_instance_num
            self.test_kl /= self.test_instance_num


    def run(self):
        self._test_phase()

        print(f"Loss at epoch {self.best_epoch}: test_loss={self.test_loss:.7f}")
        print(f"Test mse={self.test_mse:.5f}, Test kl={self.test_kl:.5f}")

        df_file_name = "./test_results/{}/sinusoid.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "Loss": [self.test_loss], "best_epoch": [self.best_epoch],
             "MSE": [self.test_mse], "KL": [self.test_kl],
             "context_mse": [self.mse_context.item()], "target_mse": [self.mse_target.item()]})

        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)