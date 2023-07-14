import jax
from jax import random, numpy as jnp

import os
from tqdm import tqdm

from linear import Linear
from common import F


num1, num2 = 6, 7
NI, NH, NO = 28*28, 2, 1
nh, no = 5, 2
seed = 0
mu = 0.8
epochs = 1
kernel_size = (5, 5)
n_batch = 10

F = F()

linear1 = Linear(NI, NH, nh)
linear2 = Linear(NH, NO, no)

(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset_of_two_nums(num1, num2)

key, key1 = random.split(random.PRNGKey(seed))
linear1_w, linear1_b = linear1.generate_params(key1)

key, key1 = random.split(key)
linear2_w, linear2_b = linear2.generate_params(key1)

params = [linear1_w, linear1_b, linear2_w, linear2_b]


def predict(params, x):
    x = jnp.reshape(x, [-1, 28*28]) # reshape no wana !!
    linear1_w, linear1_b, linear2_w, linear2_b = params
    tmp = linear1.forward(linear1_w, linear1_b, x)
    tmp = linear1.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = linear1.get_sum_prob_of_on_neuron(tmp)
    tmp = jnp.transpose(tmp)
    tmp = linear2.forward(linear2_w, linear2_b, tmp)
    tmp = jnp.reshape(tmp, [-1, no]) # reshape no wana !!
    # tmp = linear.append_off_neuron(tmp)
    y = F.softmax(tmp, axis=1)
    return y

def loss_fn(params, x, t):
    y = predict(params, x)
    tmp_loss = -1 * jnp.sum(t*jnp.log(y+1e-7), axis=1)
    loss = jnp.mean(tmp_loss)
    return loss

def update_params(params, grads):
    params[0] = params[0] - mu * grads[0]
    params[1] = params[1] - mu * grads[1]
    params[2] = params[2] - mu * grads[2]
    params[3] = params[3] - mu * grads[3]
    return params


""" 学習 """
grad_loss = jax.grad(loss_fn, argnums=0)
max_iter = len(train_t) // n_batch

for i in range(epochs):
    print(f'epoch: {i+1} / {epochs}')
    
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
    
    loss = loss_fn(params, train_x, train_t)
    print(f'loss: {loss}')


""" 検証 """
# 学習データでの正解率測定
y = predict(params, train_x)
train_acc = F.test(y, train_t)
print(f'TRAIN_data_acc: {train_acc}')
# 検証データでの正解率測定
y = predict(params, test_x)
test_acc = F.test(y, test_t)
print(f'TEST_data_acc: {test_acc}')
