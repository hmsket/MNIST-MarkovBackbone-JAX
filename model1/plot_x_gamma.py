from jax import numpy as jnp

import matplotlib.pyplot as plt

from conv import Conv
from linear import Linear
from common import *



# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.903_te0.890_nh2_no2_s0_m0.8_e30_k5_b10_c0.001_t0.3,1.0'

hyparams = dir.split('_') # hyper parameters
labels = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
no = int(hyparams[4][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))

conv = Conv(nh, kernel_size)
linear = Linear(nh, no)

conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
linear_w = jnp.load(f'./params/{dir}/linear_w.npy')
linear_b = jnp.load(f'./params/{dir}/linear_b.npy')
params = [conv_w, conv_b, linear_w, linear_b]


def calc_x_gamma(params, x):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = softmax(tmp, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    tmp = linear.forward(linear_w, linear_b, tmp)
    x_gamma = softmax(tmp, axis=1)
    return x_gamma


fig = plt.figure()

if no == 2:
    ax = fig.add_subplot()
    ax.set_xlabel(r'$x_1^\gamma$')
    ax.set_ylabel(r'$x_2^\gamma$')
    ax.set_aspect('equal')
if no == 3:
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(r'$x_1^\gamma$')
    ax.set_ylabel(r'$x_2^\gamma$')
    ax.set_zlabel(r'$x_3^\gamma$')
    ax.view_init(elev=45, azim=60) # 3Dグラフの表示角度を変える

for label in labels:
    (train_x, train_y), (test_x, test_y) = get_mnist_dataset([label])
    x_gammas = calc_x_gamma(params, train_x)
    if no == 2:
        ax.plot(x_gammas[:,0], x_gammas[:,1], marker='.', ls='None', label=label)
    if no == 3:
        ax.plot(x_gammas[:,0], x_gammas[:,1], x_gammas[:,2], marker='.', ls='None', label=label)

plt.legend()
plt.show()
