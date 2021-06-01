import os
from matplotlib import pyplot as plt


def vis_sinusoid_traj(trajs, times, pred, epoch, exp_name, val=None):
    trajs = trajs.cpu().detach().numpy()
    times = times.cpu().detach().numpy()
    plt.xlim(-0.2, 10.1)
    plt.ylim(-1.71, 1.71)
    plt.scatter(times[0], trajs[0])  # starting points
    plt.plot(times, trajs)
    plt.axvline(x=4.9, linestyle=':')

    pred = pred.cpu().detach().numpy()
    plt.plot(times, pred, color='red')
    plt.scatter(times, trajs, color='black')

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