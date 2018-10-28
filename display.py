import numpy as np
import os
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['savefig.dpi'] = 300


labels = np.linspace(-1, 1, 1000)

root = r'G:\Python\libs\binarynet\data'

def get_data(name):
    w = np.loadtxt(
        os.path.join(root, 'weights_{}.csv'.format(name)), delimiter=',')
    w = (w.T / np.sum(w, axis=1)).T * 100
    stat = np.loadtxt(
        os.path.join(root, 'stats_{}.csv'.format(name)), delimiter=',')
    return w[:26], stat[:26]

w_det, s_det = get_data('determ')
w_stoch, s_stoch = get_data('stoch')
w_none, s_none = get_data('none')

fig = plt.figure()
plt.plot(s_none[:, 0], 100 - 100 * s_none[:, 3], 'r', label='Train none')
plt.plot(s_none[:, 0], 100 - 100 *s_none[:, 5], 'r--', label='Test none')

plt.plot(s_det[:, 0], 100 - 100 *s_det[:, 3], 'g', label='Train Deterministic')
plt.plot(s_det[:, 0], 100 - 100 *s_det[:, 5], 'g--', label='Test Deterministic')

plt.plot(s_stoch[:, 0], 100 - 100 *s_stoch[:, 3], 'b', label='Train Stochastic')
plt.plot(s_stoch[:, 0], 100 - 100 *s_stoch[:, 5], 'b--', label='Test Stochastic')

plt.xlabel('Epoch')
plt.ylabel('Error %')
plt.legend()
plt.tight_layout()
fig.show()


fig = plt.figure()
plt.plot(labels, w_none[-1], 'r', label='none')
plt.plot(labels, w_det[-1], 'g', label='Deterministic')
plt.plot(labels, w_stoch[-1], 'b', label='Stochastic')
plt.xlabel('Weight Bin')
plt.ylabel('Percent %')
plt.legend()
plt.tight_layout()
fig.show()


max_ = max(max(np.max(w_stoch), np.max(w_none)), np.max(w_det))
for x, z, label in [
    (s_none, w_none, 'none'), (s_det, w_det, 'Deterministic'),
        (s_stoch, w_stoch, 'Stochastic')]:
    my_dpi = 96
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = x[:, 0][:, np.newaxis].repeat(len(labels), axis=1)
    Y = labels.T[np.newaxis, :].repeat(len(x[:, np.newaxis]), axis=0)

    surf = ax.plot_surface(
        X, Y, z, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)

    plt.xlabel('Epoch', linespacing=6.2)
    plt.ylabel('Weight Bin', linespacing=6.2)
    plt.title(label)
    ax.set_zlabel('Percent %', linespacing=6.2)
    ax.yaxis.labelpad = 13
    ax.xaxis.labelpad = 13
    ax.zaxis.labelpad = 13
    ax.set_zlim(0, max_)
    plt.tight_layout()
    fig.show()
plt.show()
