from jax import numpy as jnp

from conv import Conv
from common import F

import matplotlib.pyplot as plt



# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.903_te0.890_nh2_no2_s0_m0.8_e30_k5_b10_c0.001_t0.3,1.0'

hyparams = dir.split('_') # hyper parameters
nums = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))
t = list(map(float, hyparams[0].split(',')))

conv = Conv(nh, kernel_size)

conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
params = [conv_w, conv_b]

F = F()
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset(nums)


def calc_x_beta(params, x):
    conv_w, conv_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    x_beta = F.softmax(tmp, t[0], axis=2)
    return x_beta


x_betas = calc_x_beta(params, train_x)
x_betas_sorted = jnp.sort(x_betas)
x_beta_mean = jnp.mean(x_betas_sorted, axis=0)

fig = plt.figure()

for i in range(nh):
    ax = fig.add_subplot(nh, 1, i+1)
    ax.hist(x_beta_mean[i], bins=100, range=[0.0, x_beta_mean[i].max()])
    ax.set_title(rf'$l = {i+1}$')

fig.suptitle(r'histogram of the average of $x^{\beta l}$')
plt.show()
