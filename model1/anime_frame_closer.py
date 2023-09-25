import jax
from jax import numpy as jnp

from conv import Conv
from common import F

import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches



""" 準備 """
# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.903_te0.890_nh2_no2_s0_m0.8_e30_k5_b10_c0.001_t0.3,1.0'

# ハイパーパラメータの設定
hyparams = dir.split('_') # hyper parameters
nums = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))
t = list(map(float, hyparams[0].split(',')))

# インスタンス生成
conv = Conv(nh, kernel_size)
F = F()

# おもみフィルタとバイアスを読み込む
conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
params = [conv_w, conv_b]

# ２つの重みフィルタが重なっているピクセル数を記憶した行列ファイルを読み込む
o = jnp.load(f'./area_matrix/ks{kernel_size[0]}.npy')

# MNISTデータセットを読み込む
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset([7]) # 7

# MNIST画像を１枚取得
k = 26 # 重なっていないのを選んだ
image = train_x[k:k+1]


""" 関数の定義 """
@jax.jit
def calc_x_beta(params, x):
    conv_w, conv_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = F.softmax(tmp, t[0], axis=2)
    x_beta = jnp.reshape(tmp, [tmp.shape[1], tmp.shape[2]])
    return x_beta

@jax.jit
def get_max_idx_of_x_beta(x_beta):
    idxs = jnp.argmax(x_beta[:,1:], axis=1)
    return idxs

@jax.jit
def conved_matrix_idx_2_input_matrix_xy(idx):
    mnist_size = 28 # MNIST: (28, 28)
    # 畳み込み後の行列の添字に対応する，畳み込み前の行列の添字を計算
    q = idx // (mnist_size - conv.kernel_size[0] + 1)
    tmp_idx = idx + q * (conv.kernel_size[0] - 1)
    # ベクトルの添字に対応する，ベクトルを行列にreshapeしたときの座標を計算
    x = tmp_idx // mnist_size
    y = tmp_idx % mnist_size
    return x, y

# @jax.jit
def G(x_beta, o):
    sum = 0
    for i in range(nh):
        for j in range(i+1, nh):
            x1 = jnp.reshape(x_beta[i,1:], [-1, 1])
            x2 = jnp.reshape(x_beta[j,1:], [1, -1])
            tmp_x = jnp.matmul(x1, x2)
            tmp_x = jnp.reshape(tmp_x, [1, -1])
            tmp_o = jnp.reshape(o, [-1, 1])
            tmp = jnp.matmul(tmp_x, tmp_o)
            sum += tmp[0][0]
    return sum

@jax.jit
def update_x_beta(x_beta, grads, mu=0.1):
    x_beta = x_beta.at[0].set(x_beta[0]+mu*grads[0])
    x_beta = x_beta.at[1].set(x_beta[1]+mu*grads[1])
    divisor = jnp.reshape(jnp.sum(x_beta, axis=1), [-1, 1])
    x_beta = x_beta / divisor # normalize
    return x_beta


""" 関数の事前コンパイル """
calc_x_beta_jit = jax.jit(calc_x_beta)
get_max_idx_of_x_beta_jit = jax.jit(get_max_idx_of_x_beta)
conved_matrix_idx_2_input_matrix_xy_jit = jax.jit(conved_matrix_idx_2_input_matrix_xy)
grad_G = jax.grad(G, argnums=0)
grad_G_jit = jax.jit(grad_G)
update_x_beta_jit = jax.jit(update_x_beta)


""" アニメーション作成のための準備 """
fig = plt.figure()
ax = plt.axes()
ax.set_xticks([])
ax.set_yticks([])
frames = []
color_list = ['r', 'g']

N = 20 # num of iteration
cells = jnp.reshape(image, [image.shape[1], image.shape[2]]) # (28, 28)


""" x_betaの初期状態 """
x_beta = calc_x_beta_jit(params, image)


""" iter=0での枠を描画する """
ims = []
ims.append(fig.text(0.5, 1.03, f'iter: 0 / {N}', size=plt.rcParams["axes.titlesize"], ha="center", transform=ax.transAxes))
idxs = get_max_idx_of_x_beta_jit(x_beta)
for j, idx in enumerate(idxs):
    x, y = conved_matrix_idx_2_input_matrix_xy(idx)
    r = patches.Rectangle(xy=(y-0.5,x-0.5), width=conv.kernel_size[0], height=conv.kernel_size[1], fill=False, color=color_list[j%5])
    ims.append(ax.add_patch(r))
ims.append(ax.imshow(cells, cmap=plt.cm.gray_r))
frames.append(ims)


""" 枠を近づける """
for i in range(N):
    # x_betaを更新する
    grads = grad_G_jit(x_beta, o)
    x_beta = update_x_beta_jit(x_beta, grads)

    # 枠を描画する
    ims = []
    ims.append(fig.text(0.5, 1.03, f'iter: {i+1} / {N}', size=plt.rcParams["axes.titlesize"], ha="center", transform=ax.transAxes))
    idxs = get_max_idx_of_x_beta_jit(x_beta)
    for j, idx in enumerate(idxs):
        x, y = conved_matrix_idx_2_input_matrix_xy(idx)
        r = patches.Rectangle(xy=(y-0.5,x-0.5), width=conv.kernel_size[0], height=conv.kernel_size[1], fill=False, color=color_list[j%5])
        ims.append(ax.add_patch(r))
    ims.append(ax.imshow(cells, cmap=plt.cm.gray_r))
    frames.append(ims)


""" アニメーションの保存 """
ani = anm.ArtistAnimation(fig, frames, interval=300, repeat=False)
ani.save('./frame_closer.gif', writer='pillow')
