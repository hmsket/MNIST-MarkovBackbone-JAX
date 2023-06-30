from jax import numpy as jnp

from conv import Conv
from common import F

from tqdm import tqdm
import matplotlib.pyplot as plt


F = F()

num = 7
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset_of_one_num(num)

dir = '6_7_tr0.968_te0.971_nh2_no2_s0_m0.8_e30_k5_b10'
conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')

nh = conv_w.shape[0]
ks = int(jnp.sqrt(conv_w.shape[1]))
kernel_size = (ks, ks)

conv = Conv(nh, kernel_size)


def get_area_of_max_weighted_sum(params, image, conv):
    conv_w, conv_b = params
    hidden = conv.forward(conv_w, conv_b, image)
    idxs = jnp.argmax(hidden, axis=2)
    
    # 畳み込み後の行列の添字に対応する，畳み込み前の行列の添字を計算
    q = idxs // (28 - conv.kernel_size[0] + 1)
    tmp_idx = idxs + q * (conv.kernel_size[0] - 1)
    # ベクトルの添字に対応する，行列にreshapeしたときの座標を計算
    x = tmp_idx // 28
    y = tmp_idx % 28

    array_shape = [nh, conv.kernel_size[0]*conv.kernel_size[1], image.shape[0]]
    array = jnp.empty(array_shape)

    for i in range(nh):
        for j in tqdm(range(image.shape[0])):
            tmp = image[j, x[j,i]:x[j,i]+conv.kernel_size[0], y[j,i]:y[j,i]+conv.kernel_size[1]]
            tmp = jnp.reshape(tmp, conv.kernel_size[0]*conv.kernel_size[1])
            array = array.at[i,:,j].set(tmp)
    return array


params = [conv_w, conv_b]
x = get_area_of_max_weighted_sum(params, train_x, conv)

corrs = jnp.empty([nh, train_x.shape[0]])

for i in range(nh):
    corr = jnp.matmul(params[0][i], x[i])
    corrs = corrs.at[i].set(corr)

projection_xs = jnp.empty([nh, train_x.shape[0], conv.kernel_size[0]*conv.kernel_size[1]])

for i in range(nh):
    for j in range(train_x.shape[0]):
        tmp = corrs[i][j] * x[i,:,j]
        projection_xs = projection_xs.at[i,j,:].set(tmp)

ave_pro_x = jnp.mean(projection_xs, axis=1) # average_projection_x

fig = plt.figure()

for i in range(nh):
    x = nh % 3
    y = nh // 3
    ax = fig.add_subplot(y+1, x+1, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    cells_tmp = ave_pro_x[i]
    cells = jnp.reshape(cells_tmp, (conv.kernel_size[0], conv.kernel_size[1]))

    ax.set_title(f'w_beta[{i}]')
    ax.imshow(cells, cmap=plt.cm.gray_r)

plt.show()
