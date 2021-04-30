import os
from matplotlib import pyplot as plt


def vis_sinusoid_traj(trajs, times, pred, epoch, exp_name, val=None):
    trajs = trajs.cpu().detach().numpy()
    times = times.cpu().detach().numpy()
    plt.xlim(-0.2, 5.1)
    plt.ylim(-1.01, 1.01)
    plt.scatter(times[0], trajs[0])  # starting points
    plt.plot(times, trajs)
    plt.axvline(x=times[24], linestyle=':')

    pred = pred.cpu().detach().numpy()
    plt.plot(times, pred, color='red')

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