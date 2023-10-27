from jax import numpy as jnp

import matplotlib.pyplot as plt

from conv import Conv
from common import *



# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.991_te0.984_nh2_no2_s0_m0.1_e1000_k5_b10'

hyparams = dir.split('_') # hyper parameters
labels = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))

conv = Conv(nh, kernel_size)

conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
params = [conv_w, conv_b]


def get_area_of_max_weighted_sum(params, image):
    conv_w, conv_b = params
    hidden = conv.forward(conv_w, conv_b, image)
    idxs = jnp.argmax(hidden, axis=2)
    
    # 畳み込み後の行列の添字に対応する，畳み込み前の行列の添字を計算
    q = idxs // (28 - kernel_size[0] + 1)
    tmp_idx = idxs + q*(kernel_size[0] - 1)
    # ベクトルの添字に対応する，行列にreshapeしたときの座標を計算
    x = tmp_idx // 28
    y = tmp_idx % 28

    areas_shape = [nh, kernel_size[0]*kernel_size[1], image.shape[0]]
    areas = jnp.empty(areas_shape)
    
    for i in range(nh):
        for j in range(image.shape[0]):
            tmp = image[j, x[j,i]:x[j,i]+kernel_size[0], y[j,i]:y[j,i]+kernel_size[1]]
            tmp = jnp.reshape(tmp, kernel_size[0]*kernel_size[1])
            areas = areas.at[i,:,j].set(tmp)
    
    return areas


fig = plt.figure()

for i in range(len(labels)):
    (train_x, train_y), (test_x, test_y) = get_mnist_dataset([labels[i]])

    x = get_area_of_max_weighted_sum(params, train_x)

    corrs = jnp.empty([nh, train_x.shape[0]])

    for j in range(nh):
        corr = jnp.matmul(params[0][j], x[j])
        corrs = corrs.at[j].set(corr)

    projection_xs = jnp.empty([nh, train_x.shape[0], kernel_size[0]*kernel_size[1]])

    for j in range(nh):
        for k in range(train_x.shape[0]):
            tmp = corrs[j][k] * x[j,:,k]
            projection_xs = projection_xs.at[j,k,:].set(tmp)

    ave_pro_x = jnp.mean(projection_xs, axis=1) # average_projection_x

    for j in range(nh):
        x = nh % 3
        y = nh // 3
        ax = fig.add_subplot(len(labels), nh, i*nh+j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        cells_tmp = ave_pro_x[j]
        cells = jnp.reshape(cells_tmp, (kernel_size[0], conv.kernel_size[1]))

        ax.set_title(f'w_beta[{j}]: input "{labels[i]}"')
        ax.imshow(cells, cmap=plt.cm.gray_r)

plt.show()
