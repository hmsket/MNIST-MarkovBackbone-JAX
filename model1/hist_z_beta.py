from jax import numpy as jnp

import matplotlib.pyplot as plt

from conv import Conv
from common import F



# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.968_te0.972_nh2_no2_s0_m0.8_e30_k5_b10'

hyparams = dir.split('_') # hyper parameters
nums = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))

conv = Conv(nh, kernel_size)

conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
params = [conv_w, conv_b]

F = F()


def calc_z_beta(params, x):
    conv_w, conv_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = F.softmax(tmp, t=1.0, axis=2)
    y = conv.get_sum_prob_of_on_neuron(tmp)
    return y


# 作成する図の行数列数を計算
col = min(nh, 5) # 最大でも５列
row = nh // col + min(1, nh % col) # あまりがあったら，もう１行

fig = plt.figure()

for i in range(nh):
    ax = fig.add_subplot(row, col, i+1)
    ax.set_title(f'z_beta_{i+1}')
    ax.set_xlim([0,1])
    for num in nums:
        (train_x, train_t), (test_x, test_t) = F.get_mnist_dataset([num])
        z_beta = calc_z_beta(params, train_x)
        ax.hist(z_beta[:,i], bins=100, label=num)

plt.legend(title='label', loc=(1.05, 0.0)) # locで，表示させる座標を指定
plt.show()
