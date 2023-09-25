import jax
from jax import random, numpy as jnp

from conv import Conv
from linear import Linear
from common import F

import os


# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.989_te0.983_nh2_no2_s0_m0.1_e1000_k5_b10_c0.001'


""" ハイパーパラメータの読み込み """
hyparams = dir.split('_') # hyper parameters
nums = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
no = int(hyparams[4][2:])
seed = int(hyparams[5][1:])
mu = float(hyparams[6][1:])
pre_epochs = int(hyparams[7][1:])
kernel_size = [int(hyparams[8][1:]), int(hyparams[8][1:])]
n_batch = int(hyparams[9][1:])
c = float(hyparams[10][1:])


""" ハイパーパラメータの指定 """
key, key1 = random.split(random.PRNGKey(seed))
epochs = 1000


""" パラメータの読み込み """
conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
linear_w = jnp.load(f'./params/{dir}/linear_w.npy')
linear_b = jnp.load(f'./params/{dir}/linear_b.npy')
params = [conv_w, conv_b, linear_w, linear_b]


""" インスタンスを生成 """
conv = Conv(nh, kernel_size)
linear = Linear(nh, no)
F = F()


""" データセットの取得 """
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset(nums)


""" 関数の定義 """
def predict(params, x, t=1.0):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = F.softmax(tmp, t, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    tmp = linear.forward(linear_w, linear_b, tmp)
    y = F.softmax(tmp, t, axis=1)
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
    
    loss = loss_fn_jit(params, train_x, train_t)
    print(f'epoch: {pre_epochs+i+1} / {pre_epochs+epochs}, loss: {loss}')


""" 検証 """
# 学習データでの正解率測定
y = predict(params, train_x)
train_acc = F.test(y, train_t)
print(f'train_acc: {train_acc}')

# 検証データでの正解率測定
y = predict(params, test_x)
test_acc = F.test(y, test_t)
print(f'test_acc : {test_acc}')


""" 学習済みパラメータの保存 """
dir = './params'
if os.path.exists(dir) == False:
    os.mkdir(dir)

dir = f'./params/{",".join(map(str, nums))}_tr{train_acc:.3f}_te{test_acc:.3f}_nh{nh}_no{no}_s{seed}_m{mu}_e{pre_epochs+epochs}_k{kernel_size[0]}_b{n_batch}_c{c}'
if os.path.exists(dir) == False:
    os.mkdir(dir)

jnp.save(dir+'/conv_w', params[0])
jnp.save(dir+'/conv_b', params[1])
jnp.save(dir+'/linear_w', params[2])
jnp.save(dir+'/linear_b', params[3])
