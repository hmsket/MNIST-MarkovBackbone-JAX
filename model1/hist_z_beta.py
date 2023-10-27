from jax import numpy as jnp

import matplotlib.pyplot as plt

from conv import Conv
from common import *



# 事前に学習したパラメータファイルを読み込む
dir = '0,1,2,3,4,5,6,7,8,9_tr0.775_te0.785_nh5_no10_s0_m0.1_e1000_k5_b10_c0.001_t1.0,1.0'

hyparams = dir.split('_') # hyper parameters
labels = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))

conv = Conv(nh, kernel_size)

conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
params = [conv_w, conv_b]


def calc_z_beta(params, x):
    conv_w, conv_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = softmax(tmp, axis=2)
    z_beta = conv.get_sum_prob_of_on_neuron(tmp)
    return z_beta


# 作成する図の行数列数を計算
col = min(nh, 5) # 最大でも５列
row = nh // col + min(1, nh % col) # あまりがあったら，もう１行

fig = plt.figure()

for i in range(nh):
    ax = fig.add_subplot(row, col, i+1)
    ax.set_title(f'z_beta_{i+1}')
    ax.set_xlim([0,1])
    for label in labels:
        (train_x, train_y), (test_x, test_y) = get_mnist_dataset([label])
        z_beta = calc_z_beta(params, train_x)
        ax.hist(z_beta[:,i], bins=100, label=label)

plt.legend(title='label', loc=(1.05, 0.0)) # locで，表示させる座標を指定
plt.show()
