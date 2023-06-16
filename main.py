import jax
from jax import random, numpy as jnp
from tqdm import tqdm

from conv import Conv
from linear import Linear
from common import F


num1, num2 = 6, 7
nh, no = 5, 2
seed = 0
mu = 0.1
epochs = 5
kernel_size = (25, 25)
n_batch = 50

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
    tmp = linear.forward(linear_w, linear_b, tmp)
    # tmp = linear.append_off_neuron(tmp)
    y = F.softmax(tmp, axis=1)
    return y

def loss_fn(params, x, t):
    y = predict(params, x)
    tmp_loss = -1 * jnp.sum(t*jnp.log(y+1e-7), axis=1) # cross entropy error
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
    # 入力画像とラベルのインデックスを対応させたままシャッフルする
    p = random.permutation(key1, len(train_t))
    train_x = train_x[p]
    train_t = train_t[p]

    for iter in tqdm(range(max_iter)):
        batch_x = train_x[iter*n_batch:(iter+1)*n_batch]
        batch_t = train_t[iter*n_batch:(iter+1)*n_batch]
        grads = grad_loss(params, batch_x, batch_t)
        params = update_params(params, grads)

""" 検証 """
y = predict(params, train_x)
acc = F.test(y, train_t)
print(f'acc: {acc}')
