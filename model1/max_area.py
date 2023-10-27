from jax import numpy as jnp

from conv import Conv
from linear import Linear
from common import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches



# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.964_te0.967_nh2_no2_s0_m0.1_e641_k5_b10_c0.001_t1.0,1.0_overlap2'

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

(train_x, train_y), (test_x, test_y) = get_mnist_dataset(labels)


def predict(params, x):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = softmax(tmp, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    tmp = linear.forward(linear_w, linear_b, tmp)
    tmp = softmax(tmp, axis=1)
    idx = jnp.argmax(tmp)
    z = labels[idx]
    return z

def get_max_idx_of_conved_matrix(conv_params, x):
    conv_w, conv_b = conv_params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = softmax(tmp, axis=2)
    tmp = jnp.argmax(tmp, axis=2)
    # ０番目は「興奮しない」を担当する
    # 図の左上は，０番目ではなく，１番目に相当する
    tmp = tmp - 1
    idxs = jnp.reshape(tmp, [-1]) # ２次元から１次元に変換する
    return idxs

def conved_matrix_idx_2_input_matrix_xy(idx):
    mnist_size = 28 # MNIST: (28, 28)
    # 畳み込み後の行列の添字に対応する，畳み込み前の行列の添字を計算
    q = idx // (mnist_size - conv.kernel_size[0] + 1)
    tmp_idx = idx + q * (conv.kernel_size[0] - 1)
    # ベクトルの添字に対応する，ベクトルを行列にreshapeしたときの座標を計算
    x = tmp_idx // mnist_size
    y = tmp_idx % mnist_size
    return x, y


fig = plt.figure()
fig.suptitle('output / label')

# 畳み込みが最大となる領域の枠の色を，各ブリックごとに指定する
# brick_1: red
# brick_2: green
colors = ['r', 'g', 'b', 'c', 'm']

image_nums = range(0, 25) # MNISTの何枚目の画像を用いるか．これだと，0〜24枚目となる
for i in range(len(image_nums)):
    num = image_nums[i]
    image = train_x[num: num+1]
    idxs = get_max_idx_of_conved_matrix(params[0:2], image)

    ax = fig.add_subplot(5, 5, i+1) # 一度に表示する枚数を増やしたければ，ここを大きくする
    ax.set_xticks([])
    ax.set_yticks([])

    for j in range(len(idxs)):
        idx = idxs[j]
        if idx == -1: # 「興奮しない」が勝ったとき
            continue
        x, y = conved_matrix_idx_2_input_matrix_xy(idx)
        r = patches.Rectangle(xy=(y-0.5,x-0.5), width=conv.kernel_size[0], height=conv.kernel_size[1], fill=False, color=colors[j%5])
        ax.add_patch(r)

    z = predict(params, image)
    label = labels[jnp.argmax(train_y[num: num+1])]
    ax.set_title(f'{z} / {label}')
    ax.imshow(jnp.reshape(image, [28, 28]), cmap=plt.cm.gray_r)

plt.show()
