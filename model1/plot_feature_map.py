from jax import numpy as jnp

import matplotlib.pyplot as plt

from conv import Conv
from common import F



# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.991_te0.988_nh5_no2_s0_m0.8_e30_k5_b10'

hyparams = dir.split('_') # hyper parameters
nums = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))

conv = Conv(nh, kernel_size)

conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
params = [conv_w, conv_b]

F = F()


def get_feature_map(params, x):
    conv_w, conv_b = params
    y = conv.forward(conv_w, conv_b, x)
    return y


# 作成する図の行数列数を指定
col = nh
row = 5

(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset(nums)
feature_map = get_feature_map(params, train_x[0:row])
feature_map_shape = [28-kernel_size[0]+1, 28-kernel_size[1]+1]

fig = plt.figure()

for i in range(row):
    for j in range(col):
        ax = fig.add_subplot(row, col, col*i+j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        cell = jnp.reshape(feature_map[i,j], feature_map_shape)
        ax.imshow(cell, cmap=plt.cm.gray_r)
    
plt.show()
