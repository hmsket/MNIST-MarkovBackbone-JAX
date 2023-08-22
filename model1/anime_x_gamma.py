import jax
from jax import random, numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.animation as anm

from conv import Conv
from linear import Linear
from common import F



""" ハイパーパラメータの設定 """
nums = [2, 6, 7]     # ２と６と７の三値分類
nh = 10              # 中間層のブリック数
no = len(nums)       # 出力ブリックの内部状態数
seed = 0             # シード値
mu = 0.8             # 学習係数
kernel_size = [5, 5] # カーネルサイズ
epochs = 50          # エポック数
n_batch = 10         # バッチサイズ


""" インスタンスを生成 """
conv = Conv(nh, kernel_size)
linear = Linear(nh, no)
F = F()


""" データセットの取得 """
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset(nums)


""" パラメータの初期生成 """
key, key1 = random.split(random.PRNGKey(seed))
conv_w, conv_b = conv.generate_params(key1)
key, key1 = random.split(key)
linear_w, linear_b = linear.generate_params(key1)
params = [conv_w, conv_b, linear_w, linear_b]


""" 関数の定義 """
def predict(params, x):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    tmp = linear.forward(linear_w, linear_b, tmp)
    y = F.softmax(tmp, axis=1)
    return y

@jax.jit
def loss_fn(params, x, t):
    y = predict(params, x)
    tmp_loss = -1 * jnp.sum(t*jnp.log(y+1e-7), axis=1) # cross entropy error
    loss = jnp.mean(tmp_loss)
    return loss

@jax.jit
def update_params(params, grads):
    params[0] = params[0] - mu * grads[0]
    params[1] = params[1] - mu * grads[1]
    params[2] = params[2] - mu * grads[2]
    params[3] = params[3] - mu * grads[3]
    return params


""" 関数の事前コンパイル """
loss_fn_jit = jax.jit(loss_fn)
grad_loss = jax.grad(loss_fn, argnums=0)
grad_loss_jit = jax.jit(grad_loss)
update_params_jit = jax.jit(update_params)  


""" アニメーション作成のための準備 """
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
        
frames = [] # アニメーションのフレームを格納する

color_list = ['r', 'g', 'b', 'c', 'm']


""" パラメータの学習 """
max_iter = len(train_t) // n_batch
for i in range(epochs):
    # データセットの順番をシャッフルする
    key, key1 = random.split(key)
    p = random.permutation(key1, len(train_t))
    train_x = train_x[p]
    train_t = train_t[p]

    for iter in range(max_iter):
        batch_x = train_x[iter*n_batch:(iter+1)*n_batch]
        batch_t = train_t[iter*n_batch:(iter+1)*n_batch]
        grads = grad_loss_jit(params, batch_x, batch_t)
        params = update_params_jit(params, grads)
    
    # アニメーションのフレームを作成する
    ims = []
    for k, num in enumerate(nums):
        (train_x_tmp, train_t_tmp), (test_x_tmp, test_t_tmp) = F.get_mnist_dataset([num])
        x_gammas = predict(params, train_x_tmp)
        if no == 2:
            ims.append(ax.plot(x_gammas[:,0], x_gammas[:,1], marker='.', ls='None', color=color_list[k], label=num))
        if no == 3:
            ims.append(ax.plot(x_gammas[:,0], x_gammas[:,1], x_gammas[:,2], marker='.', ls='None', color=color_list[k], label=num))

    # 絵をすべて重ね合わせて，１枚のframeにする
    frame = ims[0]
    for j in range(1, len(ims)):
        frame += ims[j]    
    
    frames.append(frame)


""" アニメーションの保存 """
ani = anm.ArtistAnimation(fig, frames, interval=100, repeat=False)
ani.save('./x_gamma.gif', writer='pillow')
