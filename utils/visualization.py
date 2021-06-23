import os
import numpy as np
from matplotlib import pyplot as plt


def vis_sinusoid_traj(trajs_c, times_c, trajs_t, times_t, pred_c, pred_t, sigma_c, sigma_t, epoch, exp_name, val=None):
    trajs_c = trajs_c.cpu().detach().numpy()  # 1 3
    times_c = times_c.cpu().detach().numpy()  # 1 3
    trajs_t = trajs_t.cpu().detach().numpy()  # 2 4
    times_t = times_t.cpu().detach().numpy()  # 2 4
    pred_c = pred_c.cpu().detach().numpy()
    pred_t = pred_t.cpu().detach().numpy()
    sigma_c = sigma_c.cpu().detach().numpy()
    sigma_t = sigma_t.cpu().detach().numpy()

    times = np.concatenate([times_c, times_t])
    trajs = np.concatenate([trajs_c, trajs_t])
    pred = np.concatenate([pred_c, pred_t])
    sigma = np.concatenate([sigma_c, sigma_t])

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    trajs = trajs[sort_idx]
    pred = pred[sort_idx]
    sigma = sigma[sort_idx]

    plt.xlim(-2, 2)
    plt.ylim(-4, 4)
    plt.plot(times, trajs, color='black', label='Ground truth', alpha=0.6)
    plt.plot(times, pred, color='red', label='Predictive trajectory', alpha=0.6)
    # plt.axvline(x=4.9, linestyle=':')
    plt.fill_between(times, pred-sigma, pred+sigma, alpha=0.5, label='Confidence')

    plt.scatter(times_c, trajs_c, color='black', label='Context point', alpha=0.6)
    plt.scatter(times_t, trajs_t, color='red', label='Target point', alpha=0.6)


    plt.legend(fontsize='xx-small', bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=5)

    plt.title(f'Num contexts: {len(trajs_c)}, Num targets: {len(trajs_t)}', fontsize='xx-small', y=-0.1)

    epoch = str(epoch)
    epoch = (3-len(epoch)) * '0' + epoch

    path = './vis' + exp_name
    if not os.path.isdir(path):
        if not os.path.isdir('./vis'):
            os.mkdir('./vis')
        os.mkdir(path)

    if val is None:
        path += 'train/'
    else:
        path += 'val/'

    if not os.path.isdir(path):
        os.mkdir(path)

    plt.savefig(path + epoch + '.png', dpi=300)
    plt.close()