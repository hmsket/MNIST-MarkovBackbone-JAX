from jax import numpy as jnp

from conv import Conv
from common import F

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


dir = '6_7_tr0.709_te0.738_nh2_no2_s1_m0.8_e30_k5_b50'

hyparams = dir.split('_') # hyper parameters
num1 = int(hyparams[0])
num2 = int(hyparams[1])
nh = int(hyparams[4][2:])
kernel_size = (int(hyparams[9][1:]), int(hyparams[9][1:]))

conv = Conv(nh, kernel_size)
conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')

F = F()
# (train_x, train_t), (test_x, test_t) = F.get_mnist_dataset_of_one_num(num1)
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset_of_two_nums(num1, num2)


fig = plt.figure()

a = range(0, 9) # MNISTデータセットの0~9番目を用いる，プロットの関係上９枚まで．

for idx in range(len(a)):
    image = train_x[a[idx]]
    tmp_image = jnp.reshape(image, [1, image.shape[0], image.shape[1]]) # 自作関数を使うための，次元のつじつま合わせ
    hidden = conv.forward(conv_w, conv_b, tmp_image)
    tmp_idxs = jnp.argmax(hidden, axis=2)
    idxs = jnp.reshape(tmp_idxs, tmp_idxs.shape[1]) # ２次元から１次元にする

    colors = ['r', 'g', 'b', 'c', 'm'] # 領域を示す線の色を適当に５色用意する

    pos = idx+1
    ax = fig.add_subplot(3, 3, pos)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(len(idxs)):
        idx = idxs[i]
        # 畳み込み後の行列の添字に対応する，畳み込み前の行列の添字を計算
        q = idx // (image.shape[0] - conv.kernel_size[0] + 1)
        tmp_idx = idx + q * (conv.kernel_size[0] - 1)
        # ベクトルの添字に対応する，行列にreshapeしたときの座標を計算
        x = tmp_idx // image.shape[0]
        y = tmp_idx % image.shape[0]
        r = patches.Rectangle(xy=(y-0.5,x-0.5), width=conv.kernel_size[0], height=conv.kernel_size[1], fill=False, color=colors[i%5])
        ax.add_patch(r)

    ax.imshow(image, cmap=plt.cm.gray_r)

plt.show()
