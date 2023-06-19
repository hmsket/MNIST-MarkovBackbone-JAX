import jax
from jax import random, numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from tqdm import tqdm

from conv import Conv
from linear import Linear
from common import F


num1, num2 = 3, 7
nh, no = 5, 1
seed = 0
mu = 0.0005
epochs = 10
kernel_size = (23, 23)
n_batch = 12183

F = F()

conv = Conv(nh, kernel_size)
linear = Linear(nh, no)

(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset_of_two_nums(num1, num2)

key, key1 = random.split(random.PRNGKey(seed))
conv_w, conv_b = conv.generate_params(key1)

key, key1 = random.split(key)
linear_w, linear_b = linear.generate_params(key1)

params = [conv_w, conv_b, linear_w, linear_b]


def predict(params, x):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    y = linear.forward(linear_w, linear_b, tmp)
    # tmp = linear.append_off_neuron(tmp)
    # y = F.softmax(tmp, axis=1)
    return y

def loss_fn(params, x, t):
    y = predict(params, x)
    # tmp_loss = -1 * jnp.sum(t*jnp.log(y+1e-7), axis=1) # cross entropy error
    tmp_loss = 1/2 * jnp.sum((t-y)*(t-y))
    loss = jnp.mean(tmp_loss)
    return loss

def update_params(params, grads):
    params[0] = params[0] - mu * grads[0]
    params[1] = params[1] - mu * grads[1]
    params[2] = params[2] - mu * grads[2]
    params[3] = params[3] - mu * grads[3]
    return params

def create_axes(fig):
    N = jnp.amin(jnp.asarray([nh, 9])) # Nを最大9にする
    width = int(jnp.ceil(jnp.sqrt(N)))
    height = int(jnp.ceil(N/width))
    axes = []
    for i in range(N):
        idx = 100*height + 10*width + (i+1)
        ax = fig.add_subplot(idx)
        axes.append(ax)
    return axes


""" conv_wのアニメーションを作りながら学習 """
fig = plt.figure()
axes = create_axes(fig)
frames = [] # frameを格納するリスト

grad_loss = jax.grad(loss_fn, argnums=0)
max_iter = len(train_t) // n_batch

# おもみの初期状態をアニメーションの１枚目に加える
ims = []
for i in range(len(axes)):
    ax = axes[i]
    ax.set_title(f'w_conv[{i}]')
    ax.set_xticks([])
    ax.set_yticks([])
    cells = jnp.reshape(params[0][i], kernel_size)
    im = ax.imshow(cells, cmap=plt.cm.gray_r)
    ims.append(im)
title = fig.text(0.5, 2.6, f'epoch: {0} / {epochs}, iter: {0} / {0}', size=plt.rcParams["axes.titlesize"], ha="center", transform=ax.transAxes)
ims.append(title)
frames.append(ims)

for i in range(epochs):
    print(f'epoch: {i+1} / {epochs}')
    plt.title(f'epoch: {i+1} / {epochs}')
    
    key, key1 = random.split(key)
    # train_xとtrain_tのインデックスを対応させたままシャッフルする
    p = random.permutation(key1, len(train_t))
    train_x = train_x[p]
    train_t = train_t[p]

    for iter in tqdm(range(max_iter)):
        batch_x = train_x[iter*n_batch:(iter+1)*n_batch]
        batch_t = train_t[iter*n_batch:(iter+1)*n_batch]
        grads = grad_loss(params, batch_x, batch_t)
        params = update_params(params, grads)

        # おもみアニメーションの作成
        ims = []
        for j in range(len(axes)):
            ax = axes[j]
            ax.set_title(f'w_conv[{j}]')
            ax.set_xticks([])
            ax.set_yticks([])
            cells = jnp.reshape(params[0][j], kernel_size)
            im = ax.imshow(cells, cmap=plt.cm.gray_r)
            ims.append(im)
        title = fig.text(0.5, 2.6, f'epoch: {i+1} / {epochs}, iter: {iter+1} / {max_iter}', size=plt.rcParams["axes.titlesize"], ha="center", transform=ax.transAxes)
        ims.append(title)
        frames.append(ims)
        
    # loss = loss_fn(params, train_x, train_t)
    # print(f'loss: {loss}')

ani = anm.ArtistAnimation(fig, frames, interval=100)
ani.save('./w_conv.gif', writer='pillow')

""" 検証 """
y = predict(params, train_x)
acc = F.test(y, train_t)
print(f'acc: {acc}')
